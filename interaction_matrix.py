import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
def plot_heatmap(self, annot: bool = False):
    """
    Mostra una heatmap della matrice D (ΔLOCO) con nomi delle feature.

    Parametri:
    - annot: se True, mostra i valori numerici nelle celle
    """


    if not hasattr(self, "D"):
        raise RuntimeError("Devi prima chiamare compute_matrix_D()")

    n_features = self.D.shape[0]

    # Usa self.feature_names se esiste, altrimenti nomi generici X0, X1, ...
    if hasattr(self, "feature_names") and len(self.feature_names) == n_features:
        feature_labels = self.feature_names
    else:
        feature_labels = [f"X{i}" for i in range(n_features)]

    plt.figure(figsize=(max(8, n_features * 0.6), max(6, n_features * 0.6)))
    sns.heatmap(
        self.D,
        cmap="coolwarm",
        center=0,
        annot=annot,
        fmt=".2f",
        xticklabels=feature_labels,
        yticklabels=feature_labels,
        cbar_kws={"label": "ΔLOCO"}
    )
    plt.title("Hi-Fi Diagnostic Matrix (ΔLOCO)", fontsize=14)
    plt.xlabel("Driver (Xᵢ)", fontsize=12)
    plt.ylabel("Feature rimossa (Xⱼ)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def sample_background(X_train: np.ndarray, background_samples: int = 50, random_state=None) -> np.ndarray:
    """
    Restituisce un sottoinsieme (background) campionato da X_train.
    Se background_samples > n_train allora si campiona con replacement.
    """
    rng = np.random.RandomState(random_state)
    n = X_train.shape[0]
    replace = background_samples > n
    idx = rng.choice(n, size=background_samples, replace=replace)
    return X_train[idx]


class HiFiDiagnosticMatrix:
    def __init__(self, model, X, y, loco_func,predict_func):
        """
        Parameters
        ----------
        model : object
            Modello già addestrato (es. regressore o classificatore)
        X : ndarray shape (n_samples, n_features)
            Dataset di input
        y : ndarray shape (n_samples,)
            Target
        loco_func : callable
            Funzione per calcolare LOCO: loco_func(X, y, driver_idx, context_indices)
            Deve restituire un valore scalare L_Z(driver->y)
        """
        self.model = model
        self.X = X
        self.y = y
        self.loco_func = loco_func
        self.predict_func = predict_func
        self.n_features = X.shape[1]
        self.loco_full = None

    def compute_matrix_D(self):
        """
        Costruisce la matrice D[j, i] = L_{Z\{j}}(X_i -> Y) - L_Z(X_i -> Y)
        """
        n_features = self.n_features
        D = np.zeros((n_features, n_features))

        # Calcoliamo i LOCO full per ogni driver
        full_context = list(range(n_features))
        self.loco_full = np.zeros(n_features)

        full_context_indices = list(range(n_features))

        print("Calcolo del LOCO di base (contesto completo)...")
        for i in tqdm(range(n_features), desc="Calcolo LOCO full"):
            context = [k for k in full_context_indices if k != i]
            self.loco_full[i] = self.loco_func(self.model, self.X, self.y, i, context)


        for i in tqdm(range(n_features), desc="Calcolo LOCO full"):
            X_bg = sample_background(self.X, 50, 42)
            preds_full = self.predict_func(self.model,self.X,self.y,full_context,X_bg)

            preds_without_driver = self.predict_func(self.model,self.X,self.y,[k for k in full_context if k!= i],X_bg)
            error_full = np.mean((self.y - preds_full) ** 2)

            error_reduced = np.mean((self.y - preds_without_driver) ** 2)

            print(error_full - error_reduced)

        # Costruiamo D
        for i in tqdm(range(n_features), desc="Costruzione matrice D"):
            for j in range(n_features):
                if j == i:
                    D[j, i] = 0  # Non calcoliamo l'effetto di rimuovere sé stesso
                    continue
                context_minus_j = [k for k in full_context if k != j and k!=i]
                loco_minus_j = self.loco_func(self.model,self.X, self.y, i, context_minus_j)
                D[j, i] = loco_minus_j - self.loco_full[i]

        self.D = D
        return D

    def compute_third_order_effects(self):
        """
        Esegue una scansione iterativa per catturare gli effetti del terzo ordine.
        Usa la matrice D per identificare le coppie più promettenti.
        """
        if not hasattr(self, "D"):
            raise RuntimeError("Devi prima chiamare compute_matrix_D()")

        n_features = self.n_features

        # Matrici per conservare i risultati del terzo ordine
        self.T_synergy = np.zeros((n_features, n_features, n_features))
        self.T_redundancy = np.zeros((n_features, n_features, n_features))

        print("\nScansione iterativa per interazioni del 3° ordine...")
        for i in tqdm(range(n_features), desc="Driver (3° ordine)"):
            # Trova il partner sinergico e ridondante più forte per il driver 'i'
            j_s = np.argmin(self.D[:, i])  # Partner sinergico (delta più negativo)
            j_r = np.argmax(self.D[:, i])  # Partner ridondante (delta più positivo)

            # --- A. Analisi della Sinergia del Terzo Ordine ---
            context_minus_js = [k for k in range(n_features) if k != i and k != j_s]
            loco_baseline_syn = self.loco_func(self.model, self.X, self.y, i, context_minus_js)

            for k in range(n_features):
                if k == i or k == j_s:
                    continue

                context_minus_js_k = [l for l in context_minus_js if l != k]
                loco_3rd_order = self.loco_func(self.model, self.X, self.y, i, context_minus_js_k)

                # Delta del terzo ordine: come cambia l'importanza di 'i' rimuovendo 'k',
                # dato che il suo miglior partner 'j_s' è già assente.
                delta_3rd = loco_3rd_order - loco_baseline_syn

                # Un delta negativo indica una sinergia a tre
                if delta_3rd < 0:
                    self.T_synergy[i, j_s, k] = -delta_3rd  # Salviamo come valore positivo

            # --- B. Analisi della Ridondanza del Terzo Ordine ---
            context_minus_jr = [k for k in range(n_features) if k != i and k != j_r]
            loco_baseline_red = self.loco_func(self.model, self.X, self.y, i, context_minus_jr)

            for k in range(n_features):
                if k == i or k == j_r:
                    continue

                context_minus_jr_k = [l for l in context_minus_jr if l != k]
                loco_3rd_order = self.loco_func(self.model, self.X, self.y, i, context_minus_jr_k)

                delta_3rd = loco_3rd_order - loco_baseline_red

                # Un delta positivo indica una ridondanza a tre
                if delta_3rd > 0:
                    self.T_redundancy[i, j_r, k] = delta_3rd

        print("Analisi del 3° ordine completata.")
        return self.T_synergy, self.T_redundancy

    def compute_redundancy_synergy(self, eps=0.0):
        """
        Calcola Ri e Si diagnostici:
          Ri = somma dei delta positivi (ridondanza)
          Si = somma del modulo dei delta negativi (sinergia)
        """
        if not hasattr(self, "D"):
            raise RuntimeError("Devi prima chiamare compute_matrix_D()")

        n_features = self.n_features
        R = np.zeros(n_features)
        S = np.zeros(n_features)

        for i in range(n_features):
            deltas = self.D[:, i]
            deltas = deltas[~np.isnan(deltas)]
            R[i] = np.sum(deltas[deltas > eps])       # ridondanza cumulativa
            S[i] = np.sum(-deltas[deltas < -eps])     # sinergia cumulativa (positiva)

        df_summary = pd.DataFrame({
            "Feature": np.arange(n_features),
            "R_redundancy": R,
            "S_synergy": S,
            "L_full": self.loco_full
        })

        self.R = R
        self.S = S
        self.df_summary = df_summary
        return df_summary

    def plot_heatmap(self, features_names, annot: bool = False):
        """
        Mostra una heatmap della matrice D (ΔLOCO) con nomi delle feature.

        Parametri:
        - annot: se True, mostra i valori numerici nelle celle
        """


        if not hasattr(self, "D"):
            raise RuntimeError("Devi prima chiamare compute_matrix_D()")

        n_features = self.D.shape[0]

        # Usa self.feature_names se esiste, altrimenti nomi generici X0, X1, ...
        feature_labels = features_names


        plt.figure(figsize=(max(8, n_features * 0.6), max(6, n_features * 0.6)))
        sns.heatmap(
            self.D,
            cmap="coolwarm",
            center=0,
            annot=annot,
            fmt=".2f",
            xticklabels=feature_labels,
            yticklabels=feature_labels,
            cbar_kws={"label": "ΔLOCO"}
        )
        plt.title("Hi-Fi Diagnostic Matrix (ΔLOCO)", fontsize=14)
        plt.xlabel("Driver (Xᵢ)", fontsize=12)
        plt.ylabel("Feature rimossa (Xⱼ)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def compute_synergy_redundancy(self, D, feature_names=None):
        """
        Calcola per ogni feature la somma dei contributi di sinergia e ridondanza
        dalla matrice diagnostica D (ΔLOCO).

        Parametri:
        - D: matrice numpy (n_features x n_features), dove D[j,i] = ΔLOCO(X_j -> X_i)
        - feature_names: lista opzionale con i nomi delle feature

        Restituisce:
        - DataFrame con colonne ["feature", "redundancy_sum", "synergy_sum", "net_effect"]
        """
        n_features = D.shape[0]
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(n_features)]

        redundancy_sum = np.sum(np.clip(D, a_min=0, a_max=None), axis=1)
        synergy_sum = - np.sum(np.clip(D, a_min=None, a_max=0), axis=1)
        net_effect = redundancy_sum + synergy_sum + self.loco_full

        df = pd.DataFrame({
            "feature": feature_names,
            "redundancy_sum": redundancy_sum,
            "synergy_sum": synergy_sum,
            "net_effect": net_effect,
            "u": self.loco_full
        })

        return df.sort_values(by="net_effect", ascending=False).reset_index(drop=True)


    def compute_full_decomposition(self, feature_names=None):
        """
        Calcola U, R e S integrando gli effetti del secondo e del terzo ordine.
        """
        if not hasattr(self, "D") or not hasattr(self, "T_synergy"):
            raise RuntimeError("Devi prima chiamare compute_matrix_D() e compute_third_order_effects()")

        n_features = self.n_features
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(n_features)]

        # 1. Contributo Unico (U) rimane il LOCO nel contesto completo
        U = self.loco_full

        n_features = self.n_features
        full_context = list(range(n_features))
        for i in tqdm(range(n_features)):

            self.loco_func(self.model,self.X, self.y, i, [k for k in full_context if k!=i])

        # 2. Effetti del Secondo Ordine (dalla matrice D)
        R_2nd_order = np.sum(np.maximum(0, self.D), axis=0)  # Somma delle colonne
        S_2nd_order = np.sum(np.abs(np.minimum(0, self.D)), axis=0)  # Somma delle colonne

        # 3. Effetti del Terzo Ordine (dai tensori T)
        # Per ogni driver, sommiamo tutti i suoi effetti del terzo ordine
        R_3rd_order = np.sum(self.T_redundancy, axis=(1, 2))
        S_3rd_order = np.sum(self.T_synergy, axis=(1, 2))

        # 4. Scomposizione Finale
        self.R = R_2nd_order + R_3rd_order
        self.S = S_2nd_order + S_3rd_order
        self.U = U

        df_summary = pd.DataFrame({
            "Feature": feature_names,
            "U_unique": self.U,
            "R_redundancy": self.R,
            "S_synergy": self.S,
            "Total_Importance": self.U + self.R + self.S
        })

        return df_summary.sort_values(by="Total_Importance", ascending=False).reset_index(drop=True)


