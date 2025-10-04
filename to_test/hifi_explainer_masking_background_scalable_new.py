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
        self.data = None
        self.target_idx = None
        self.feature_indices = None
        self.X_background = None

    def fit(self, data: np.ndarray, target_idx: int,background_samples=50):
        """
        Addestra il modello una sola volta e memorizza i dati necessari per l'analisi.
        """
        print("Esecuzione del training una tantum del modello...")
        self.target_idx = target_idx
        self.y_train = data[:, target_idx]

        feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        self.X_train = data[:, feature_indices]
        self.feature_indices = feature_indices

        # --- MODIFICA CHIAVE QUI ---
        # Invece di calcolare solo le medie, memorizziamo un campione di dati
        if self.X_train.shape[0] > background_samples:
            print(f"Creazione del dataset di background con {background_samples} campioni...")
            random_indices = np.random.choice(self.X_train.shape[0], background_samples, replace=False)
            self.X_background = self.X_train[random_indices]
        else:
            self.X_background = self.X_train  # Usa tutti i dati se sono pochi
        # --- FINE MODIFICA ---

        self.trained_model = self.model_factory()
        self.trained_model.fit(self.X_train, self.y_train)
        print("Modello addestrato.")

    # Dentro la tua classe HiFiExplainerScalable

    def _predict_with_mask(self, X_source: np.ndarray, mapped_active_indices: list) -> np.ndarray:
        """
        Esegue una predizione usando il modello pre-addestrato.
        Le feature non attive vengono mascherate campionando dal background dataset,
        e il risultato finale è la media delle predizioni.
        """
        # Caso base: se non ci sono feature attive, la predizione migliore è la media del target
        if not mapped_active_indices:
            return np.full_like(self.y_train, np.mean(self.y_train))

        num_source_samples = X_source.shape[0]
        num_background_samples = self.X_background.shape[0]

        # Lista per contenere i vettori di predizione per ogni campione di background
        predictions_over_background = []

        # 1. Itera su ogni riga del background dataset
        for i in range(num_background_samples):
            # 2. Inizia con una riga del background, ripetuta per ogni campione in X_source
            X_masked = np.tile(self.X_background[i, :], (num_source_samples, 1))

            # 3. Crea la maschera per le colonne attive (quelle da NON sostituire)
            mask = np.zeros(self.X_train.shape[1], dtype=bool)
            if mapped_active_indices:
                mask[mapped_active_indices] = True

            # 4. Sovrascrivi le colonne attive con i valori reali da X_source
            X_masked[:, mask] = X_source[:, mask]

            # 5. Esegui la predizione su questa versione mascherata e salvala
            pred = self.trained_model.predict(X_masked)
            predictions_over_background.append(pred)

        # 6. La predizione finale è la MEDIA delle predizioni calcolate
        final_predictions = np.mean(predictions_over_background, axis=0)

        return final_predictions

    def _loco_calculator_penalized(self, X_source, driver_idx: int, context_indices: list,
                                   lambda_penalty=0, sample_size=200) -> float:
        num_samples = self.X_train.shape[0]
        num_background_samples = self.X_background.shape[0]

        if len(context_indices) == 0:
            # Subcampionamento esempi
            num_samples = self.X_train.shape[0]
            if sample_size < num_samples:
                sampled_indices = np.random.choice(num_samples, sample_size, replace=False)
            else:
                sampled_indices = np.arange(num_samples)

            mapped_driver_idx = self.feature_indices.index(driver_idx)
            loco_per_sample = []

            for i in sampled_indices:
                # Replica background per imputazioni
                masked_full = np.tile(self.X_background, (1, 1))
                # Mantieni la feature driver
                masked_full[:, mapped_driver_idx] = X_source[i, mapped_driver_idx]
                # Predizioni
                preds_full = self.trained_model.predict(masked_full)
                # Errori rispetto al target
                errors = (self.y_train[i] - preds_full) ** 2
                # Media sulle imputazioni
                loco_per_sample.append(np.mean(errors))

        # Subcampionamento degli esempi
        if sample_size < num_samples:
            sampled_indices = np.random.choice(num_samples, sample_size, replace=False)
        else:
            sampled_indices = np.arange(num_samples)

        mapped_driver_idx = self.feature_indices.index(driver_idx)
        mapped_context_indices = [self.feature_indices.index(i) for i in context_indices]

        penalized_local_locos = []
        loco_per_imputation_mean_array = []

        for i in sampled_indices:
            # Costruisci in blocco tutte le imputazioni ridotte e complete
            masked_reduced = np.tile(self.X_background, (1, 1))
            masked_full = np.tile(self.X_background, (1, 1))

            if mapped_context_indices:
                masked_reduced[:, mapped_context_indices] = X_source[i, mapped_context_indices]
                masked_full[:, mapped_context_indices] = X_source[i, mapped_context_indices]

            masked_full[:, mapped_driver_idx] = X_source[i, mapped_driver_idx]

            # Predizioni vettorizzate
            preds_reduced = self.trained_model.predict(masked_reduced)
            preds_full = self.trained_model.predict(masked_full)

            # Calcolo degli errori
            error_reduced = (self.y_train[i] - preds_reduced) ** 2
            error_full = (self.y_train[i] - preds_full) ** 2

            loco_per_imputation = error_reduced - error_full
            mean_local_loco = np.mean(loco_per_imputation)
            loco_per_imputation_mean_array.append(mean_local_loco)

            variance_penalty = np.var(loco_per_imputation)  # penalità sulla parte ridotta

            penalized_loco = mean_local_loco - (lambda_penalty * variance_penalty)
            penalized_local_locos.append(penalized_loco)


        return np.mean(penalized_local_locos)

    def _loco_calculator_relative_r2(self, X_source, driver_idx: int, context_indices: list,
                                     sample_size=200, lambda_penalty=0.0) -> float:
        """
        Calcola il LOCO relativo basato su R² relativo (varianza spiegata),
        coerente con il paper originale, usando imputazioni di background.
        """

        num_samples = self.X_train.shape[0]
        num_background_samples = self.X_background.shape[0]

        # Subcampionamento degli esempi
        if sample_size < num_samples:
            sampled_indices = np.random.choice(num_samples, sample_size, replace=False)
        else:
            sampled_indices = np.arange(num_samples)

        mapped_driver_idx = self.feature_indices.index(driver_idx)
        mapped_context_indices = [self.feature_indices.index(i) for i in context_indices]

        loco_per_sample = []

        # Itera sugli esempi subcampionati
        for i in sampled_indices:
            # Replica background per imputazioni
            masked_reduced = np.tile(self.X_background, (1, 1))
            masked_full = np.tile(self.X_background, (1, 1))

            # --- Applica il contesto ---
            if mapped_context_indices:
                masked_reduced[:, mapped_context_indices] = X_source[i, mapped_context_indices]
                masked_full[:, mapped_context_indices] = X_source[i, mapped_context_indices]

            # Mantieni la feature driver
            masked_full[:, mapped_driver_idx] = X_source[i, mapped_driver_idx]

            # Predizioni
            preds_reduced = self.trained_model.predict(masked_reduced)
            preds_full = self.trained_model.predict(masked_full)

            # --- Calcolo del R² relativo ---
            var_y = np.var(self.y_train)
            # percentuale di varianza spiegata
            r2_per_imputation = 1 - ((self.y_train[i] - preds_full) ** 2 / var_y)

            # Media sulle imputazioni
            mean_r2 = np.mean(r2_per_imputation)

            # Penalità opzionale basata sulla varianza delle predizioni
            variance_penalty = np.var(preds_full)
            penalized_r2 = mean_r2 - lambda_penalty * variance_penalty

            loco_per_sample.append(penalized_r2)

        # L0 o LOCO globale come media dei campionamenti
        loco_global = np.mean(loco_per_sample)
        return loco_global

    def _calculate_hoi(self, driver_idx: int, available_indices: list, is_red: bool, tau: float = 1e-6, T: float = 0.1):
        """Implementazione della greedy search (logica principale)."""
        # La logica interna rimane la stessa, ma le chiamate cambiano.

        # Calcolo del pairwise iniziale
        initial_loco = self._loco_calculator_relative_r2(self.X_train, driver_idx, context_indices=[])

        #if initial_loco < tau:
            #initial_loco = 0

        mi_sequence = [initial_loco]
        selected_drivers_indices = []
        remaining_indices = available_indices.copy()

        should_continue = True
        while len(remaining_indices) > 0 and should_continue:
            deltas = []
            for candidate_idx in remaining_indices:
                current_context_indices = selected_drivers_indices + [candidate_idx]
                loco_val = self._loco_calculator_penalized(self.X_train, driver_idx, current_context_indices)
                deltas.append(loco_val)

            deltas = np.array(deltas)

            # --- SOFTMIN / SOFTMAX ---
            if is_red:
                # Softmin
                weights = np.exp(-deltas / T)
            else:
                # Softmax
                weights = np.exp(deltas / T)

            weights /= weights.sum()
            soft_value = np.dot(weights, deltas)

            # Trova la feature corrispondente più "probabile"
            best_delta_local_idx = np.argmax(weights)
            winner_feature_idx = remaining_indices[best_delta_local_idx]
            best_loco_value = soft_value


            # --- TEST STATISTICO CORRETTO ---
            if self.n_surrogates > 0:
                surrogate_locos = []
                for _ in range(self.n_surrogates):
                    # Crea una copia delle features e mescola la colonna del vincitore
                    X_surrogate = self.X_train.copy()
                    winner_mapped_idx = self.feature_indices.index(winner_feature_idx)
                    X_surrogate[:, winner_mapped_idx] = np.random.permutation(X_surrogate[:, winner_mapped_idx])

                    # CRITICITÀ 1 RISOLTA: Passa i dati surrogato al calcolatore di LOCO
                    loco_surr = self._loco_calculator_penalized(X_surrogate, driver_idx,
                                                               selected_drivers_indices + [winner_feature_idx])
                    surrogate_locos.append(loco_surr)

                # ... (calcolo p-value e stop come prima) ...
                comparison_function = np.less if is_red else np.greater
                p_val = np.sum(comparison_function(surrogate_locos, best_loco_value)) / (self.n_surrogates + 1)
                if p_val >= self.alpha:
                    should_continue = False

            if should_continue:
                # ... (aggiornamento stato come prima) ...
                if best_loco_value < tau:
                    best_loco_value = 0
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

