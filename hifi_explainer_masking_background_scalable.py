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
        self.data = None
        self.target_idx = None
        self.feature_indices = None
        self.X_background = None
        self.background_samples = None

    def fit(self, data: np.ndarray, target_idx: int,background_samples=50):
        self.target_idx = target_idx
        self.y_train = data[:, target_idx]

        feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        self.X_train = data[:, feature_indices]
        self.feature_indices = feature_indices

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
            return np.full_like(self.y_train, np.mean(self.y_train))

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

    def _loco_calculator_scalable(self, X_source: np.ndarray, driver_idx: int, context_indices: list) -> float:
        mapped_driver_idx = self.feature_indices.index(driver_idx)
        mapped_context_indices = [self.feature_indices.index(i) for i in context_indices]

        preds_reduced = self._predict_with_mask(X_source, mapped_context_indices)
        error_reduced = np.mean((self.y_train - preds_reduced) ** 2)

        preds_full = self._predict_with_mask(X_source, [mapped_driver_idx] + mapped_context_indices)
        error_full = np.mean((self.y_train - preds_full) ** 2)

        loco_value = error_reduced - error_full
        return loco_value

    def _calculate_hoi(self, driver_idx: int, available_indices: list, is_red: bool):
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

            best_delta_local_idx = np.argmin(deltas) if is_red else np.argmax(deltas)
            best_loco_value = deltas[best_delta_local_idx]
            winner_feature_idx = remaining_indices[best_delta_local_idx]

            if self.n_surrogates > 0:
                surrogate_locos = []
                for _ in range(self.n_surrogates):
                    X_surrogate = self.X_train.copy()
                    winner_mapped_idx = self.feature_indices.index(winner_feature_idx)
                    X_surrogate[:, winner_mapped_idx] = np.random.permutation(X_surrogate[:, winner_mapped_idx])

                    loco_surr = self._loco_calculator_scalable(X_surrogate, driver_idx,
                                                               selected_drivers_indices + [winner_feature_idx])
                    surrogate_locos.append(loco_surr)

                comparison_function = np.less if is_red else np.greater
                p_val = np.sum(comparison_function(surrogate_locos, best_loco_value)) / (self.n_surrogates + 1)
                if p_val >= self.alpha:
                    should_continue = False

            if should_continue:
                mi_sequence.append(best_loco_value)
                selected_drivers_indices.append(winner_feature_idx)
                remaining_indices.remove(winner_feature_idx)

        n_drivers = len(mi_sequence)
        return selected_drivers_indices, mi_sequence, n_drivers

    def explain_driver(self, data: np.ndarray, target_idx: int, driver_idx: int):

        y = data[:, target_idx]
        driver_x = data[:, driver_idx].reshape(-1, 1)

        num_features = data.shape[1]
        all_indices = list(range(num_features))
        available_indices = [i for i in all_indices if i not in [target_idx, driver_idx]]

        drivers_red, mi_red, r_n = self._calculate_hoi(
            driver_idx, available_indices, is_red=True
        )

        drivers_syn, mi_syn, s_n = self._calculate_hoi(
            driver_idx, available_indices, is_red=False
        )

        return drivers_red, drivers_syn, mi_red, mi_syn, r_n, s_n

    def _hifi_decomposition(self, mi_red_list: list, mi_syn_list: list, r_n_list: list, s_n_list: list):

        num_drivers = len(mi_red_list)
        dU, dR, dS, dbiv = [], [], [], []

        for k in range(num_drivers):
            mi_red = mi_red_list[k]
            mi_syn = mi_syn_list[k]
            r_n = r_n_list[k]
            s_n = s_n_list[k]

            pairwise_loco = mi_red[0]
            dbiv.append(pairwise_loco)

            unique_contribution = mi_red[r_n - 1]
            dU.append(unique_contribution)

            redundancy = pairwise_loco - unique_contribution
            dR.append(redundancy)

            max_loco = mi_syn[s_n - 1]
            synergy = max_loco - pairwise_loco
            dS.append(synergy)

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

