# File: hifi_explainer_scalable.py
import copy

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
        self.X_permuted = None

    def fit(self, data: np.ndarray, target_idx: int):
        """
        Addestra il modello una sola volta e memorizza i dati necessari per l'analisi.
        """
        print("Esecuzione del training una tantum del modello...")
        self.target_idx = target_idx
        self.y_train = data[:, target_idx]

        feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        self.X_train = data[:, feature_indices]
        self.feature_indices = feature_indices

        # CRITICITÀ 2 RISOLTA: Calcola le medie solo sulle features
        self.feature_means = np.mean(self.X_train, axis=0)
        self.X_permuted = self.X_train.copy()

        num_features = self.X_train.shape[1]

        for i in range(num_features):
            self.X_permuted[:, i] = np.zeros(self.X_permuted[:, i].shape[0])

        self.trained_model = self.model_factory()
        self.trained_model.fit(self.X_train, self.y_train)
        print("Modello addestrato.")

    def _predict_with_mask(self, X_source: np.ndarray, mapped_active_indices: list) -> np.ndarray:
        """
        Esegue una predizione usando il modello pre-addestrato.
        Le feature non in active_indices vengono mascherate con la loro media.

        OTTIMIZZAZIONE: Parte da X_source e sovrascrive, invece di usare np.tile.
        """
        if not mapped_active_indices:
            return np.full_like(self.y_train, np.mean(self.y_train))

        # Mappa gli indici originali (es. colonna 7) ai nuovi indici di X_train (es. colonna 6)
        #mapped_active_indices = [self.feature_indices.index(i) for i in active_indices]

        # Crea una maschera booleana per le colonne da mantenere
        mask = np.zeros(self.X_train.shape[1], dtype=bool)
        mask[mapped_active_indices] = True

        # Crea una copia di X_source e sovrascrivi le colonne inattive con la media
        X_masked = X_source.copy()
        #X_masked[:, ~mask] = self.feature_means[~mask]

        X_masked[:, ~mask] = self.X_permuted[:,~mask]


        return self.trained_model.predict(X_masked)

    # Dentro la tua classe HiFiExplainerIncremental
    import copy  # Assicurati di avere questo import all'inizio del file

    def _loco_calculator_incremental(self, X_source, driver_idx: int, context_indices: list) -> float:
        """
        Calcola il LOCO usando l'adattamento incrementale (partial_fit),
        gestendo correttamente sia il caso generale che quello pairwise.
        """
        # Mappa gli indici originali a quelli relativi di self.X_train
        mapped_driver_idx = self.feature_indices.index(driver_idx)
        mapped_context_indices = [self.feature_indices.index(i) for i in context_indices]

        # --- CASO 1: IL CONTESTO È VUOTO (Calcolo del Pairwise) ---
        # La tua osservazione è corretta: questo caso va gestito separatamente.
        if not mapped_context_indices:
            # La formula è L∅(X→Y) = ε(Y|∅) - ε(Y|X)

            # ε(Y|∅) è l'errore senza nessuna feature, che è la varianza del target.
            error_no_features = np.var(self.y_train)

            # Per calcolare ε(Y|X), addestriamo un modello temporaneo solo sul driver.
            # Usiamo un modello "fresco" per non essere influenzati dal training precedente.
            model_only_driver = self.model_factory()

            # Adattiamo il modello solo sul driver per alcune epoche.
              # Numero di epoche di adattamento (iperparametro)
            model_only_driver.fit(X_source[:, [mapped_driver_idx]], self.y_train)

            preds_only_driver = model_only_driver.predict(X_source[:, [mapped_driver_idx]])
            error_with_driver = np.var(self.y_train - preds_only_driver)

            loco_value = error_no_features - error_with_driver
            return loco_value

        # --- CASO 2: IL CONTESTO NON È VUOTO (Calcolo del LOCO Standard) ---
        else:
            # La formula è Lz(X→Y) = ε(Y|Z) - ε(Y|X,Z)

            # Calcolo di ε(Y|Z): Adattiamo un modello solo sul contesto.
            model_reduced = self.model_factory()

            model_reduced.fit(X_source[:, mapped_context_indices], self.y_train)
            preds_reduced = model_reduced.predict(X_source[:, mapped_context_indices])
            error_reduced =  np.var(self.y_train - preds_reduced)

            # Calcolo di ε(Y|X,Z): Adattiamo un modello sul driver + contesto.
            model_full = self.model_factory()
            full_indices = [mapped_driver_idx] + mapped_context_indices

            model_full.fit(X_source[:, full_indices], self.y_train)
            preds_full = model_full.predict(X_source[:, full_indices])
            error_full = np.var(self.y_train - preds_full)

            loco_value = error_reduced - error_full
            return loco_value
    def _calculate_hoi(self, driver_idx: int, available_indices: list, is_red: bool):
        """Implementazione della greedy search (logica principale)."""
        # La logica interna rimane la stessa, ma le chiamate cambiano.

        # Calcolo del pairwise iniziale
        initial_loco = self._loco_calculator_incremental(self.X_train, driver_idx, context_indices=[])

        mi_sequence = [initial_loco]
        selected_drivers_indices = []
        remaining_indices = available_indices.copy()

        should_continue = True
        while len(remaining_indices) > 0 and should_continue:
            deltas = []
            for candidate_idx in remaining_indices:
                current_context_indices = selected_drivers_indices + [candidate_idx]
                loco_val = self._loco_calculator_incremental(self.X_train, driver_idx, current_context_indices)
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
                    loco_surr = self._loco_calculator_incremental(X_surrogate, driver_idx,
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

        #if np.any(neg_u_indices):
            #print(f"Correzione di {np.sum(neg_u_indices)} valori di U negativi, trattandoli come zero.")
            #dbiv[neg_biv_indices] = 0
            #dR[neg_u_indices] = dbiv[neg_u_indices]  # R = pairwise - 0
            #dU[neg_u_indices] = 0

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

