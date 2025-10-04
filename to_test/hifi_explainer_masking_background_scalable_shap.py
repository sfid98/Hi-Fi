# File: hifi_shapley_explainer.py
from math import comb, factorial

import numpy as np
import pandas as pd
import itertools


class HiFiShapleyExplainer:
    def __init__(self, model_factory, num_subset_samples=1000, background_samples=50):
        self.model_factory = model_factory
        self.num_subset_samples = num_subset_samples  # Numero di contesti casuali da testare
        self.background_samples = background_samples
        self.trained_model = None
        self.X_train = None
        self.y_train = None
        self.X_background = None
        self.feature_indices = None

    def fit(self, data: np.ndarray, target_idx: int):
        """Addestra il modello e crea il dataset di background."""
        print("Esecuzione del training una tantum del modello...")
        self.y_train = data[:, target_idx]
        self.feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        self.X_train = data[:, self.feature_indices]

        if self.X_train.shape[0] > self.background_samples:
            random_indices = np.random.choice(self.X_train.shape[0], self.background_samples, replace=False)
            self.X_background = self.X_train[random_indices]
        else:
            self.X_background = self.X_train

        self.trained_model = self.model_factory()
        self.trained_model.fit(self.X_train, self.y_train)
        print("Modello addestrato.")

    def _calculate_shapley_weight(self, subset_size: int, num_features: int) -> float:
        """
        Calcola il peso di Shapley per un sottoinsieme di una data dimensione.
        Questa versione è robusta e previene la divisione per zero.
        """
        k = subset_size
        M = num_features

        # se subset vuoto o contiene tutte le feature → peso 0
        if k < 0 or k >= M:
            return 0.0

        # Peso Shapley = 1 / C(M, k) * 1 / M? oppure combinazione inversa
        # La versione standard usata in KernelSHAP
        weight = 1.0 / comb(M, k)
        return weight

    def _get_expected_prediction(self, mapped_active_indices: list) -> np.ndarray:
        """Calcola la predizione attesa, integrando sulle feature di background."""
        if not mapped_active_indices:
            return np.full_like(self.y_train, np.mean(self.trained_model.predict(self.X_train)))

        num_predictions = self.X_train.shape[0]
        num_background_samples = self.X_background.shape[0]

        predictions_over_background = []
        for i in range(num_background_samples):
            X_masked = np.tile(self.X_background[i, :], (num_predictions, 1))
            mask = np.zeros(self.X_train.shape[1], dtype=bool)
            mask[mapped_active_indices] = True
            X_masked[:, mask] = self.X_train[:, mask]
            pred = self.trained_model.predict(X_masked)
            predictions_over_background.append(pred)

        return np.mean(predictions_over_background, axis=0)

    def _contribution_calculator(self, driver_idx: int, context_indices: list) -> float:
        """Calcola il contributo marginale medio (phi) del driver nel contesto dato."""
        mapped_driver_idx = self.feature_indices.index(driver_idx)
        mapped_context_indices = [self.feature_indices.index(i) for i in context_indices]

        preds_reduced = self._get_expected_prediction(mapped_context_indices)
        preds_full = self._get_expected_prediction([mapped_driver_idx] + mapped_context_indices)

        marginal_contribution = np.mean(preds_full - preds_reduced)
        return marginal_contribution

    def _analyze_driver_shapley(self, driver_idx: int):
        """
        Analizza un singolo driver usando la stima Shapley su un campione di subset.
        """
        available_indices = [idx for idx in self.feature_indices if idx != driver_idx]
        num_available = len(available_indices)

        contributions = []
        subsets_info = []
        weights = []  # Lista per memorizzare i pesi

        print(f"Campionamento di {self.num_subset_samples} contesti casuali...")
        for _ in range(self.num_subset_samples):
            # Scegli una dimensione 'k' casuale per il contesto (da 0 a N-1 features)
            k = np.random.randint(0, len(available_indices) + 1)

            # Scegli 'k' feature casuali per formare il contesto
            context_subset = list(np.random.choice(available_indices, k, replace=False))

            # Calcola il contributo marginale per questo contesto
            weight = self._calculate_shapley_weight(len(context_subset), num_available)
            weights.append(weight)

            phi = self._contribution_calculator(driver_idx, context_subset)

            contributions.append(phi)
            subsets_info.append(context_subset)

        # --- APPLICA LA TUA LOGICA DI SCOMPOSIZIONE ---
        if not contributions:
            return 0, 0, 0, 0

        contributions = np.array(contributions)

        L0 = np.sum(contributions * weights) / np.sum(weights)
        Lmin = np.min(contributions)
        Lmax = np.max(contributions)

        U = Lmin
        R = L0 - U
        S = Lmax - L0

        shap = np.sum(contributions)

        U = max(U, 0)
        R = max(R, 0)
        S = max(S, 0)

        return U, R, S, L0,shap

    def run_analysis(self):
        """Esegue l'analisi completa per tutti i driver."""
        if self.trained_model is None:
            raise RuntimeError("Devi prima chiamare il metodo .fit() sui dati.")

        driver_indices = self.feature_indices
        dU, dR, dS, dbiv, dshap = [], [], [], [],[]

        print(f"Avvio dell'analisi Hi-Fi-Shapley per {len(driver_indices)} drivers...")
        for i, driver_idx in enumerate(driver_indices):
            print(f"  Analisi del driver {i + 1}/{len(driver_indices)} (feature indice: {driver_idx})...")

            U, R, S, L0,shap = self._analyze_driver_shapley(driver_idx)

            dU.append(U)
            dR.append(R)
            dS.append(S)
            dshap.append(shap)
            dbiv.append(L0)

        final_results = {
            'unique': np.array(dU), 'redundancy': np.array(dR),
            'synergy': np.array(dS), 'pairwise': np.array(dbiv),
            'driver_indices': driver_indices,
            'dshap': dshap
        }
        print("Analisi completata.")
        return final_results