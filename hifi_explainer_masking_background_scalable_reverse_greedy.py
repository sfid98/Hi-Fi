# File: hifi_explainer_scalable.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


class HiFiExplainerScalable:
    def __init__(self, model_factory, n_surrogates=10, alpha=0.05):
        self.y_test = None
        self.X_test = None
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
        self.background_samples = None
        self.feature_means = None
        self.feature_stds = None

    def fit(self, data: np.ndarray, target_idx: int,background_samples=150):
        self.target_idx = target_idx
        y_train = data[:, target_idx]

        feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        X_train = data[:, feature_indices]

        self.feature_indices = feature_indices

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.feature_means = np.mean(self.X_train, axis=0)
        self.feature_stds = np.std(self.X_train, axis=0)

        if self.X_train.shape[0] > background_samples:
            print(f"Creazione del dataset di background con {background_samples} campioni...")
            random_indices = np.random.choice(self.X_train.shape[0], background_samples, replace=False)
            self.X_background = self.X_train[random_indices]
        else:
            self.X_background = self.X_train

        self.trained_model = self.model_factory()
        self.trained_model.fit(self.X_train, self.y_train)
        print("Modello addestrato.")


    def fit_masking(self, data: np.ndarray, target_idx: int,
                    masking_rate=0.3, background_samples=50, nn_neighbors=5):


        self.target_idx = target_idx
        self.y_train = data[:, target_idx]


        self.feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        self.X_train = data[:, self.feature_indices]

        if self.X_train.shape[0] > background_samples:
            random_indices = np.random.choice(self.X_train.shape[0], background_samples, replace=False)
            self.X_background = self.X_train[random_indices]
        else:
            self.X_background = self.X_train

        X_corrupted = self.X_train.copy()
        n_samples, n_features = self.X_train.shape
        num_background_samples = self.X_background.shape[0]

        for i in range(n_samples):
            num_to_mask = max(1, int(masking_rate * n_features))
            cols_to_mask = np.random.choice(n_features, num_to_mask, replace=False)

            context_cols = [c for c in range(n_features) if c not in cols_to_mask]

            if context_cols:
                context_vals = X_corrupted[i, context_cols]
                bg_context = self.X_background[:, context_cols]
                dists = np.linalg.norm(bg_context - context_vals, axis=1)
                nearest_k = np.argsort(dists)[:nn_neighbors]
                bg_row_idx = np.random.choice(nearest_k)
            else:
                bg_row_idx = np.random.randint(0, num_background_samples)

            X_corrupted[i, cols_to_mask] = self.X_background[bg_row_idx, cols_to_mask]

        self.trained_model = self.model_factory()
        self.trained_model.fit(X_corrupted, self.y_train)



    def fit_robust_augmented(self, data: np.ndarray, target_idx: int,
                             num_augmented=3, masking_rate=0.3,
                             nn_neighbors=5, use_mixup=True, mixup_alpha=0.4,background_samples = 100):

        self.background_samples = background_samples

        self.target_idx = target_idx
        self.y_train = data[:, target_idx]

        self.feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        self.X_train = data[:, self.feature_indices]

        if self.X_train.shape[0] > self.background_samples:
            random_indices = np.random.choice(self.X_train.shape[0],
                                              self.background_samples, replace=False)
            self.X_background = self.X_train[random_indices]
        else:
            self.X_background = self.X_train

        X_aug = [self.X_train]
        y_aug = [self.y_train]

        for a in range(num_augmented):
            X_corrupted = self.X_train.copy()
            n_samples, n_features = self.X_train.shape
            num_background_samples = self.X_background.shape[0]

            for i in range(n_samples):
                num_to_mask = max(1, int(masking_rate * n_features))
                cols_to_mask = np.random.choice(n_features, num_to_mask, replace=False)
                context_cols = [c for c in range(n_features) if c not in cols_to_mask]

                if context_cols:
                    context_vals = X_corrupted[i, context_cols]
                    bg_context = self.X_background[:, context_cols]
                    dists = np.linalg.norm(bg_context - context_vals, axis=1)
                    nearest_k = np.argsort(dists)[:nn_neighbors]
                    bg_row_idx = np.random.choice(nearest_k)
                else:
                    bg_row_idx = np.random.randint(0, num_background_samples)

                X_corrupted[i, cols_to_mask] = self.X_background[bg_row_idx, cols_to_mask]

            X_aug.append(X_corrupted)
            y_aug.append(self.y_train)

        X_final = np.vstack(X_aug)
        y_final = np.hstack(y_aug)

        if use_mixup:
            n = X_final.shape[0]
            lam = np.random.beta(mixup_alpha, mixup_alpha, size=n)

            indices = np.random.permutation(n)
            X_mix = lam[:, None] * X_final + (1 - lam)[:, None] * X_final[indices]
            y_mix = lam * y_final + (1 - lam) * y_final[indices]

            X_final = np.vstack([X_final, X_mix])
            y_final = np.hstack([y_final, y_mix])

        # --- TRAINING ---
        self.trained_model = self.model_factory()
        self.trained_model.fit(X_final, y_final)
        print("Modello robusto addestrato con masking+augmentation.")

    def _predict_with_mask(self, X_source: np.ndarray, mapped_active_indices: list) -> np.ndarray:
        if not mapped_active_indices:
            return np.full_like(self.y_test, np.mean(self.y_test))

        num_source_samples = X_source.shape[0]
        num_background_samples = self.X_background.shape[0]

        predictions_over_background = []

        for i in range(num_background_samples):
            X_masked = np.tile(self.X_background[i, :], (num_source_samples, 1))

            mask = np.zeros(self.X_train.shape[1], dtype=bool)
            if mapped_active_indices:
                mask[mapped_active_indices] = True

            X_masked[:, mask] = X_source[:, mask]

            pred = self.trained_model.predict(X_masked)
            predictions_over_background.append(pred)

        final_predictions = np.mean(predictions_over_background, axis=0)

        return final_predictions

    # Aggiungi questo nuovo metodo alla tua classe HiFiExplainerScalable

    # Dentro la tua classe HiFiExplainerScalable

    def _predict_with_noise(self, X_source: np.ndarray, mapped_active_indices: list,
                            noise_level=1.0, num_samples=50) -> np.ndarray:
        """
        Esegue una predizione usando il modello pre-addestrato.
        Le feature non attive vengono "offuscate" con rumore gaussiano.
        La predizione finale è la MEDIA su 'num_samples' versioni rumorose.
        """
        # Se tutte le feature sono attive, non c'è nulla da perturbare
        if len(mapped_active_indices) == self.X_train.shape[1]:
            return self.trained_model.predict(X_source)

        num_source_samples, num_features = X_source.shape

        # Lista per raccogliere i vettori di predizione per ogni campione rumoroso
        predictions_over_noise = []

        # 1. Itera per il numero di campioni rumorosi desiderati
        for _ in range(num_samples):
            X_noisy = X_source.copy()

            # 2. Itera su tutte le colonne delle feature per decidere se aggiungere rumore
            for i in range(num_features):
                # Se una colonna NON è nella lista di quelle attive, aggiungiamo rumore
                if i not in mapped_active_indices:
                    noise_std = self.feature_stds[i] * noise_level
                    noise = np.random.normal(0, noise_std, num_source_samples)
                    X_noisy[:, i] += noise

            # 3. Esegui la predizione sulla matrice "offuscata" e salvala
            pred = self.trained_model.predict(X_noisy)
            predictions_over_noise.append(pred)

        # 4. La predizione finale è la MEDIA di tutte le predizioni calcolate
        final_predictions = np.mean(predictions_over_noise, axis=0)

        return final_predictions
    def _loco_calculator_scalable(self, X_source: np.ndarray, driver_idx: int, context_indices: list) -> float:
        mapped_driver_idx = self.feature_indices.index(driver_idx)
        mapped_context_indices = [self.feature_indices.index(i) for i in context_indices]

        preds_reduced = self._predict_with_mask(X_source, mapped_context_indices)
        error_reduced = np.mean((self.y_test - preds_reduced) ** 2)

        preds_full = self._predict_with_mask(X_source, [mapped_driver_idx] + mapped_context_indices)
        error_full = np.mean((self.y_test - preds_full) ** 2)

        loco_value = error_reduced - error_full
        return loco_value

    def _calculate_hoi_reverse(self, driver_idx: int, available_indices: list, is_red: bool, threshold=0.00009):
        # LOCO con tutte le features disponibili
        starting_point = self._loco_calculator_scalable(self.X_test, driver_idx, available_indices)

        remaining_indices = available_indices.copy()
        selected_indices = []
        loco_sequence = [starting_point]

        should_continue = True
        while remaining_indices and should_continue:
            deltas = []
            candidates = []

            for candidate_idx in remaining_indices:
                reduced_context = [i for i in remaining_indices if i != candidate_idx]
                loco_without = self._loco_calculator_scalable(self.X_test, driver_idx, reduced_context)
                delta = loco_without - starting_point  # confronto con LOCO iniziale


                # Se siamo nel caso ridondanza, prendiamo solo delta positivi
                if is_red and delta > 0:
                    deltas.append(delta)
                    candidates.append((candidate_idx, loco_without))

                # Se siamo nel caso sinergia, prendiamo solo delta negativi
                if not is_red and delta < 0:
                    deltas.append(delta)
                    candidates.append((candidate_idx, loco_without))

            if not deltas:
                break  # nessun candidato valido trovato

            if is_red:
                best_idx, best_loco = candidates[np.argmax(deltas)]
                best_delta = max(deltas)
            else:
                best_idx, best_loco = candidates[np.argmin(deltas)]
                best_delta = min(deltas)

            if abs(best_delta) < threshold:
                should_continue = False
            else:
                selected_indices.append(best_idx)
                loco_sequence.append(best_loco)
                remaining_indices.remove(best_idx)

        return selected_indices, loco_sequence, len(selected_indices)


    def explain_driver(self, data: np.ndarray, target_idx: int, driver_idx: int):
        y = data[:, target_idx]

        num_features = data.shape[1]
        all_indices = list(range(num_features))
        available_indices = [i for i in all_indices if i not in [target_idx, driver_idx]]

        red_set, loco_red_set, r_n = self._calculate_hoi_reverse(driver_idx, available_indices, is_red=True)
        syn_set, loco_syn_set, s_n = self._calculate_hoi_reverse(driver_idx, available_indices, is_red=False)

        return red_set, syn_set, loco_red_set, loco_syn_set, r_n, s_n


    def _hifi_decomposition(self, loco_red_set_list: list, loco_syn_set_list: list, r_n_list: list, s_n_list: list):

        num_drivers = len(loco_red_set_list)
        dU, dR, dS, dbiv = [], [], [], []


        for k in range(num_drivers):
            #l0 = self._loco_calculator_scalable(self.X_train, k + 1, context_indices=[])

            loco_red_set = loco_red_set_list[k]
            loco_syn_set = loco_syn_set_list[k]
            r_n = r_n_list[k]
            s_n = s_n_list[k]

            loco_full = loco_red_set[0]
            U = loco_syn_set[s_n - 1]
            dU.append(U)

            R = loco_red_set[r_n - 1] - loco_full
            S = loco_full - loco_syn_set[s_n - 1]

            dR.append(R)
            dS.append(S)
            dbiv.append(loco_full)
        return np.array(dU), np.array(dR), np.array(dS), np.array(dbiv)

    def _calculate_path_matrix(self, drivers_list: list, mi_list: list, driver_indices: list, is_red: bool):
        num_drivers = len(driver_indices)
        driver_map = {original_idx: matrix_idx for matrix_idx, original_idx in enumerate(driver_indices)}
        path_matrix = np.zeros((num_drivers, num_drivers))

        for i in range(num_drivers):
            driver_row_idx = i
            helpers = drivers_list[i]
            mi_sequence = mi_list[i]
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
        num_features = data.shape[1]
        driver_indices = [i for i in range(num_features) if i != target_idx]

        all_drivers_red, all_drivers_syn = [], []
        all_mi_red, all_mi_syn = [], []
        all_r_n, all_s_n = [], []

        print(f"Starting Hi-Fi analysis for {len(driver_indices)} drivers...")
        for i, driver_idx in enumerate(driver_indices):
            print(f"Analysis of driver {i + 1}/{len(driver_indices)} (feature index: {driver_idx})...")

            drivers_red, drivers_syn, mi_red, mi_syn, r_n, s_n = self.explain_driver(
                data, target_idx, driver_idx
            )

            all_drivers_red.append(drivers_red)
            all_drivers_syn.append(drivers_syn)
            all_mi_red.append(mi_red)
            all_mi_syn.append(mi_syn)
            all_r_n.append(r_n)
            all_s_n.append(s_n)

        print("Calculating the decomposition...")
        dU, dR, dS, dbiv = self._hifi_decomposition(
            all_mi_red, all_mi_syn, all_r_n, all_s_n
        )

        neg_u_indices = dU < 0
        neg_biv_indices = dbiv < 0

        if np.any(neg_u_indices):
            print(f"Correzione di {np.sum(neg_u_indices)} valori di U negativi, trattandoli come zero.")
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


    def standard_loco(self):
        if self.trained_model is None:
            raise RuntimeError("Il modello deve essere addestrato prima con .fit()")

        print("Calcolo del LOCO standard (approssimato con masking)...")

        all_feature_indices = list(range(self.X_train.shape[1]))
        predictions_full = self._predict_with_mask(self.X_train, all_feature_indices)
        error_full = np.mean((self.y_train - predictions_full) ** 2)

        dLOC = []
        num_features = self.X_train.shape[1]

        for i in range(num_features):
            active_indices_reduced = [j for j in range(num_features) if j != i]

            predictions_reduced = self._predict_with_mask(self.X_train, active_indices_reduced)
            error_reduced = np.mean((self.y_train - predictions_reduced) ** 2)

            loco_val = error_reduced - error_full
            dLOC.append(loco_val)

        print("LOCO standard calcolato.")
        return np.array(dLOC)

