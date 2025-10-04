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
    import seaborn as sns
    import matplotlib.pyplot as plt

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


class HiFiDiagnosticMatrix:
    def __init__(self, model, X, y, loco_func):
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
        self.n_features = X.shape[1]

    def compute_matrix_D(self):
        """
        Costruisce la matrice D[j, i] = L_{Z\{j}}(X_i -> Y) - L_Z(X_i -> Y)
        """
        n_features = self.n_features
        D = np.zeros((n_features, n_features))

        # Calcoliamo i LOCO full per ogni driver
        full_context = list(range(n_features))
        loco_full = np.zeros(n_features)
        for i in tqdm(range(n_features), desc="Calcolo LOCO full"):

            loco_full[i] = self.loco_func(self.model,self.X, self.y, i, [k for k in full_context if k!=i])

        # Costruiamo D
        for i in tqdm(range(n_features), desc="Costruzione matrice D"):
            for j in range(n_features):
                if j == i:
                    D[j, i] = 0  # Non calcoliamo l'effetto di rimuovere sé stesso
                    continue
                context_minus_j = [k for k in full_context if k != j and k!=i]
                loco_minus_j = self.loco_func(self.model,self.X, self.y, i, context_minus_j)
                D[j, i] = loco_minus_j - loco_full[i]

        self.D = D
        self.loco_full = loco_full
        return D

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


