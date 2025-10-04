# File: hifi_explainer_scalable.py

import numpy as np
from sklearn.neighbors import NearestNeighbors


class HiFiExplainerScalable:
    def __init__(self, model_factory, n_surrogates=100, alpha=0.05):
        self.model_factory = model_factory
        self.n_surrogates = n_surrogates
        self.alpha = alpha
        self.trained_model = None
        self.X_train = None
        self.y_train = None
        self.feature_means = None
        self.data = None
        self.target_idx = None
        self.feature_indices = None
        self.X_background = None
        self.k_neighbors = None
        self.knn_index = None

    def fit(self, data: np.ndarray, target_idx: int, k_neighbors=10):
        """
        Addestra il modello una sola volta e memorizza i dati necessari per l'analisi.
        """
        print("Esecuzione del training una tantum del modello...")
        self.target_idx = target_idx
        self.y_train = data[:, target_idx]

        feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        self.X_train = data[:, feature_indices]
        self.feature_indices = feature_indices

        print(f"Costruzione dell'indice k-NN con k={k_neighbors}...")
        self.k_neighbors = k_neighbors
        # L'indice viene costruito sulle features standardizzate
        self.knn_index = NearestNeighbors(n_neighbors=k_neighbors)
        self.knn_index.fit(self.X_train)
        print("Indice k-NN costruito.")

        self.trained_model = self.model_factory()
        self.trained_model.fit(self.X_train, self.y_train)
        print("Modello addestrato.")

    def _get_knn_masked_data(self, active_indices_relative: list) -> np.ndarray:
        """
        Crea un dataset mascherato dove le colonne inattive sono riempite con la media
        dei k-vicini più prossimi, trovati usando solo le colonne attive.
        """
        # Se non ci sono feature attive, maschera tutto con la media globale (caso base)
        if not active_indices_relative:
            return np.tile(self.feature_means, (self.X_train.shape[0], 1))

        # --- CORREZIONE CHIAVE ---
        # 1. Costruisci un indice k-NN "al volo" usando SOLO le feature attive
        knn_subspace_index = NearestNeighbors(n_neighbors=self.k_neighbors, n_jobs=-1)  # n_jobs=-1 usa tutti i core
        knn_subspace_index.fit(self.X_train[:, active_indices_relative])

        # 2. Trova, per ogni riga, i suoi vicini basandosi solo su questo sottospazio
        _, neighbor_indices = knn_subspace_index.kneighbors(self.X_train[:, active_indices_relative])
        # --- FINE CORREZIONE ---

        # Inizia con una copia dei dati reali
        X_masked = self.X_train.copy()

        # Identifica le colonne da mascherare (quelle NON attive)
        num_features = self.X_train.shape[1]
        indices_to_mask = [i for i in range(num_features) if i not in active_indices_relative]

        # Per ogni colonna da mascherare, calcola la media locale basata sui vicini
        for col_idx in indices_to_mask:
            # Prendi i valori della colonna `col_idx` solo per le righe dei vicini
            neighbor_values = self.X_train[neighbor_indices, col_idx]
            # Calcola la media per ogni riga (axis=1)
            mean_of_neighbors = np.mean(neighbor_values, axis=1)
            # Riempi la colonna mascherata con queste medie condizionali
            X_masked[:, col_idx] = mean_of_neighbors

        return X_masked

    def _predict_with_mask(self, X_source: np.ndarray, mapped_active_indices: list) -> np.ndarray:
        if not mapped_active_indices:
            return np.full_like(self.y_train, np.mean(self.y_train))



        X_masked_reduced = self._get_knn_masked_data(mapped_active_indices)



        # La predizione finale è la media delle predizioni su tutti i campioni di background
        preds = self.trained_model.predict(X_masked_reduced)

        return preds

    def _loco_calculator_scalable(self, X_source: np.ndarray, driver_idx: int, context_indices: list) -> float:
        """
        Calcola il LOCO usando predizioni su dati mascherati.
        Ora accetta X_source per permettere l'uso di dati surrogato.
        """
        # X_source qui sono solo le features, non l'intera matrice di dati

        # Mappa gli indici originali a quelli di X_source
        mapped_driver_idx = self.feature_indices.index(driver_idx)
        mapped_context_indices = [self.feature_indices.index(i) for i in context_indices]

        # Errore del modello ridotto (solo contesto attivo)
        preds_reduced = self._predict_with_mask(X_source, mapped_context_indices)
        error_reduced = np.mean((self.y_train - preds_reduced) ** 2)

        # Errore del modello completo (driver + contesto attivi)
        preds_full = self._predict_with_mask(X_source, [mapped_driver_idx] + mapped_context_indices)
        error_full = np.mean((self.y_train - preds_full) ** 2)

        loco_value = error_reduced - error_full
        return loco_value

    def _calculate_hoi(self, driver_idx: int, available_indices: list, is_red: bool):
        """Implementazione della greedy search (logica principale)."""
        # La logica interna rimane la stessa, ma le chiamate cambiano.

        # Calcolo del pairwise iniziale
        initial_loco = self._loco_calculator_scalable(self.X_train, driver_idx, context_indices=[])

        mi_sequence = [initial_loco]
        selected_drivers_indices = []
        remaining_indices = available_indices.copy()

        should_continue = True
        while len(remaining_indices) > 0 and should_continue:
            deltas = []
            for candidate_idx in remaining_indices:
                current_context_indices = selected_drivers_indices + [candidate_idx]
                loco_val = self._loco_calculator_scalable(self.X_train, driver_idx, current_context_indices)
                deltas.append(loco_val)

            # ... (scelta del migliore come prima) ...
            best_delta_local_idx = np.argmin(deltas) if is_red else np.argmax(deltas)
            best_loco_value = deltas[best_delta_local_idx]
            winner_feature_idx = remaining_indices[best_delta_local_idx]

            # --- TEST STATISTICO CORRETTO ---
            if self.n_surrogates > 0:
                surrogate_locos = []
                for _ in range(self.n_surrogates):
                    # Crea una copia delle features e mescola la colonna del vincitore
                    X_surrogate = self.X_train.copy()
                    winner_mapped_idx = self.feature_indices.index(winner_feature_idx)
                    X_surrogate[:, winner_mapped_idx] = np.random.permutation(X_surrogate[:, winner_mapped_idx])

                    # CRITICITÀ 1 RISOLTA: Passa i dati surrogato al calcolatore di LOCO
                    loco_surr = self._loco_calculator_scalable(X_surrogate, driver_idx,
                                                               selected_drivers_indices + [winner_feature_idx])
                    surrogate_locos.append(loco_surr)

                # ... (calcolo p-value e stop come prima) ...
                comparison_function = np.less if is_red else np.greater
                p_val = np.sum(comparison_function(surrogate_locos, best_loco_value)) / (self.n_surrogates + 1)
                if p_val >= self.alpha:
                    should_continue = False

            if should_continue:
                # ... (aggiornamento stato come prima) ...
                mi_sequence.append(best_loco_value)
                selected_drivers_indices.append(winner_feature_idx)
                remaining_indices.remove(winner_feature_idx)

        n_drivers = len(mi_sequence)
        return selected_drivers_indices, mi_sequence, n_drivers

    # I metodi _calculate_hoi, _explain_driver, _hifi_decomposition, etc.
    # vanno copiati dalla classe HiFiExplainer originale, ma con una modifica chiave:
    # la chiamata al calcolatore di LOCO deve usare la nuova versione scalabile.

    # Esempio di come andrebbe modificato _calculate_hoi
    def explain_driver(self, data: np.ndarray, target_idx: int, driver_idx: int):
        """
        Helper function with the objective to apply the Hi-Fi method to the driver feature
        """
        y = data[:, target_idx]
        driver_x = data[:, driver_idx].reshape(-1, 1)

        num_features = data.shape[1]
        all_indices = list(range(num_features))
        # We keep track of all the possible candidates for the driver feature
        available_indices = [i for i in all_indices if i not in [target_idx, driver_idx]]

        # Redundancy calculator (is_red = True)
        drivers_red, mi_red, r_n = self._calculate_hoi(
            driver_idx, available_indices, is_red=True
        )

        # Synergy calculator (is_red=False)
        drivers_syn, mi_syn, s_n = self._calculate_hoi(
            driver_idx, available_indices, is_red=False
        )

        return drivers_red, drivers_syn, mi_red, mi_syn, r_n, s_n

    def _hifi_decomposition(self, mi_red_list: list, mi_syn_list: list, r_n_list: list, s_n_list: list):
        """
        Calculate the component U, R, S starting from the result of the greedy search
        """
        num_drivers = len(mi_red_list)
        dU, dR, dS, dbiv = [], [], [], []

        for k in range(num_drivers):
            # Extracting the result for the k driver
            mi_red = mi_red_list[k]
            mi_syn = mi_syn_list[k]
            r_n = r_n_list[k]
            s_n = s_n_list[k]


            # The pairwise predicting power LOCO is the first element of the sequence
            pairwise_loco = mi_red[0]
            dbiv.append(pairwise_loco)

            # (U) = is the latest value of the LOCO (the ones we the full z_min)
            unique_contribution = mi_red[r_n - 1]
            dU.append(unique_contribution)

            # (R) = Pairwise - U
            redundancy = pairwise_loco - unique_contribution
            dR.append(redundancy)

            #  (S) = Max LOCO (driver + z_max)  - Pairwise
            max_loco = mi_syn[s_n - 1]
            synergy = max_loco - pairwise_loco
            dS.append(synergy)

        return np.array(dU), np.array(dR), np.array(dS), np.array(dbiv)

    def _calculate_path_matrix(self, drivers_list: list, mi_list: list, driver_indices: list, is_red: bool):
        """
        Calculate the path matrix for the heatmap.
        """
        num_drivers = len(driver_indices)
        driver_map = {original_idx: matrix_idx for matrix_idx, original_idx in enumerate(driver_indices)}
        path_matrix = np.zeros((num_drivers, num_drivers))

        for i in range(num_drivers):
            driver_row_idx = i
            helpers = drivers_list[i] # This contains the z_min or z_max for the specific driver
            mi_sequence = mi_list[i] # This contains the list of increading or decreasing LOCO calculated by adding or removing a feature
            for j in range(len(helpers)):
                step = j + 1
                helper_original_idx = helpers[j]
                if helper_original_idx in driver_map:
                    helper_col_idx = driver_map[helper_original_idx]
                    if is_red:
                        delta = mi_sequence[step - 1] - mi_sequence[step]
                    else:
                        delta = mi_sequence[step] - mi_sequence[step - 1]

                    path_matrix[driver_row_idx, helper_col_idx] = delta

        return path_matrix


    def run_analysis(self, data: np.ndarray, target_idx: int):
        """
        Run Hi-Fi on the entire dataset analyzing each feature as a driver
        """
        num_features = data.shape[1]
        driver_indices = [i for i in range(num_features) if i != target_idx]

        all_drivers_red, all_drivers_syn = [], []
        all_mi_red, all_mi_syn = [], []
        all_r_n, all_s_n = [], []

        print(f"Starting Hi-Fi analysis for {len(driver_indices)} drivers...")
        for i, driver_idx in enumerate(driver_indices):
            print(f"Analysis of driver {i + 1}/{len(driver_indices)} (feature index: {driver_idx})...")

            # Start greedy search for the current driver
            drivers_red, drivers_syn, mi_red, mi_syn, r_n, s_n = self.explain_driver(
                data, target_idx, driver_idx
            )

            # Save the results required for the decomposition
            all_drivers_red.append(drivers_red)
            all_drivers_syn.append(drivers_syn)
            all_mi_red.append(mi_red)
            all_mi_syn.append(mi_syn)
            all_r_n.append(r_n)
            all_s_n.append(s_n)

        print("Calculating the decomposition...")
        # Start the decomposition
        dU, dR, dS, dbiv = self._hifi_decomposition(
            all_mi_red, all_mi_syn, all_r_n, all_s_n
        )

        neg_u_indices = dU < 0
        neg_biv_indices = dbiv < 0

        if np.any(neg_u_indices):
            print(f"Correzione di {np.sum(neg_u_indices)} valori di U negativi, trattandoli come zero.")
            dbiv[neg_biv_indices] = 0
            dR[neg_u_indices] = dbiv[neg_u_indices]  # R = pairwise - 0
            dU[neg_u_indices] = 0

        print("Heatmap matrix calculation...")
        redundancy_path = self._calculate_path_matrix(all_drivers_red, all_mi_red, driver_indices, is_red=True)
        synergy_path = self._calculate_path_matrix(all_drivers_syn, all_mi_syn, driver_indices, is_red=False)

        results = {
            'unique': dU,
            'redundancy': dR,
            'synergy': dS,
            'pairwise': dbiv,
            'driver_indices': driver_indices,
            'redundancy_path': redundancy_path,
            'synergy_path': synergy_path
        }
        print("Analysis completed")
        return results

    # Dentro la classe HiFiExplainerScalable

    def standard_loco(self):
        """
        Calcola il LOCO standard per ogni feature usando l'approccio scalabile
        basato su masking, senza riaddestrare il modello.
        """
        if self.trained_model is None:
            raise RuntimeError("Il modello deve essere addestrato prima con .fit()")

        print("Calcolo del LOCO standard (approssimato con masking)...")

        # 1. Calcola l'errore del modello completo (nessuna feature mascherata)
        # Per farlo, passiamo tutti gli indici delle feature come attivi.
        all_feature_indices = list(range(self.X_train.shape[1]))
        predictions_full = self._predict_with_mask(self.X_train, all_feature_indices)
        error_full = np.mean((self.y_train - predictions_full) ** 2)

        dLOC = []
        num_features = self.X_train.shape[1]

        # 2. Itera su ogni feature da "rimuovere" (mascherare)
        for i in range(num_features):
            # 3. Definisci le feature che devono rimanere attive (tutte tranne la i-esima)
            active_indices_reduced = [j for j in range(num_features) if j != i]

            # 4. Calcola la predizione e l'errore con una sola feature mascherata
            predictions_reduced = self._predict_with_mask(self.X_train, active_indices_reduced)
            error_reduced = np.mean((self.y_train - predictions_reduced) ** 2)

            # La formula del LOCO è errore_senza_feature - errore_con_feature
            loco_val = error_reduced - error_full
            dLOC.append(loco_val)

        print("LOCO standard calcolato.")
        return np.array(dLOC)

