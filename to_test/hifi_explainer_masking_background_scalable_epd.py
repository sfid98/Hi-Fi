# File: hifi_explainer_scalable.py

import numpy as np


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
        self.oracle_predictions = None

    def fit(self, data: np.ndarray, target_idx: int, background_samples=10):
        """
        Addestra il modello, crea il background dataset e calcola le predizioni
        del modello completo che useremo come riferimento ("oracolo").
        """
        print("Esecuzione del training una tantum del modello...")
        self.target_idx = target_idx
        self.y_train = data[:, target_idx]  # Il ground truth reale ci serve ancora per il training

        self.feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        self.X_train = data[:, self.feature_indices]

        # Crea il background dataset (se usi il masking con background)
        if self.X_train.shape[0] > background_samples:
            random_indices = np.random.choice(self.X_train.shape[0], background_samples, replace=False)
            self.X_background = self.X_train[random_indices]
        else:
            self.X_background = self.X_train

        # Addestra il modello
        self.trained_model = self.model_factory()
        self.trained_model.fit(self.X_train, self.y_train)
        print("Modello addestrato.")

        # --- NUOVA PARTE: CALCOLO DELLE PREDIZIONI "ORACOLO" ---
        print("Calcolo delle predizioni di riferimento (oracolo)...")
        # La predizione dell'oracolo è quella del modello addestrato su TUTTE le feature
        self.oracle_predictions = self.trained_model.predict(self.X_train)
        print("Predizioni di riferimento calcolate.")
        # --- FINE NUOVA PARTE ---


    def _predict_with_mask(self, X_source: np.ndarray, mapped_active_indices: list) -> np.ndarray:
        """
        Esegue una predizione usando il modello pre-addestrato.
        Le feature non in active_indices vengono mascherate con la loro media.

        OTTIMIZZAZIONE: Parte da X_source e sovrascrive, invece di usare np.tile.
        """
        if not mapped_active_indices:
            return np.full_like(self.y_train, np.mean(self.y_train))

        num_predictions = X_source.shape[0]
        num_background_samples = self.X_background.shape[0]

        # Crea N predizioni, una per ogni campione nel background dataset
        predictions_over_background = []
        for i in range(num_background_samples):
            # Inizia con una riga del background
            X_masked = np.tile(self.X_background[i, :], (num_predictions, 1))

            # Sovrascrivi le colonne attive con i valori reali da X_source
            mask = np.zeros(self.X_train.shape[1], dtype=bool)
            mask[mapped_active_indices] = True
            X_masked[:, mask] = X_source[:, mask]

            # Fai la predizione per questa versione mascherata
            pred = self.trained_model.predict(X_masked)
            predictions_over_background.append(pred)

        # La predizione finale è la media delle predizioni su tutti i campioni di background
        final_predictions = np.mean(predictions_over_background, axis=0)

        return final_predictions

    def _loco_calculator_model_centric(self, X_source: np.ndarray, driver_idx: int, context_indices: list) -> np.ndarray:
        """
        Calcola il LOCO locale basandosi sulla deviazione dalla predizione del
        modello completo ("oracolo"), invece che dal ground truth.
        """
        # 1. Predizione del modello ridotto (solo contesto attivo)
        mapped_driver_idx = self.feature_indices.index(driver_idx)
        mapped_context_indices = [self.feature_indices.index(i) for i in context_indices]

        preds_reduced = self._predict_with_mask(X_source, mapped_context_indices)  # o _predict_with_permutation, etc.
        # Errore quadratico RISPETTO ALL'ORACOLO
        error_reduced_local = np.mean((self.oracle_predictions - preds_reduced) ** 2)

        # 2. Predizione del modello completo (driver + contesto attivi)
        preds_full = self._predict_with_mask(X_source, [mapped_driver_idx] + mapped_context_indices)
        # Errore quadratico RISPETTO ALL'ORACOLO
        error_full_local = np.mean((self.oracle_predictions - preds_full) ** 2)

        # 3. Il LOCO locale misura la riduzione della deviazione dall'oracolo
        loco_value = error_reduced_local - error_full_local
        return loco_value


    def _calculate_hoi(self, driver_idx: int, available_indices: list, is_red: bool):
        """Implementazione della greedy search (logica principale)."""
        # La logica interna rimane la stessa, ma le chiamate cambiano.

        # Calcolo del pairwise iniziale
        initial_epd = self._loco_calculator_model_centric(self.X_train,driver_idx, context_indices=[])

        mi_sequence = [initial_epd]
        selected_drivers_indices = []
        remaining_indices = available_indices.copy()

        should_continue = True
        while len(remaining_indices) > 0 and should_continue:
            deltas = []
            for candidate_idx in remaining_indices:
                current_context_indices = selected_drivers_indices + [candidate_idx]
                epd_val = self._loco_calculator_model_centric(self.X_train,driver_idx, current_context_indices)
                deltas.append(epd_val)

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
                    loco_surr = self._loco_calculator_model_centric(X_surrogate, driver_idx,
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

