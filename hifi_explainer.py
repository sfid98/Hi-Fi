import numpy as np
from utils import polynomial_regression_model

class HiFiExplainer:
    def __init__(self, model_function=polynomial_regression_model, n_surrogates=100, alpha=0.05):

        self.model = model_function
        self.n_surrogates = n_surrogates
        self.alpha = alpha

    def _hifi_loco_calculator(self, y: np.ndarray, driver_x: np.ndarray, context_z: np.ndarray) -> float:

        if driver_x.ndim == 1:
            driver_x = driver_x.reshape(-1, 1)


        if context_z is not None and context_z.shape[1] > 0:
            full_features = np.hstack((driver_x, context_z))
        else:
            full_features = driver_x

        residuals_full = self.model(y, full_features)
        error_full = np.var(residuals_full)

        if context_z is not None and context_z.shape[1] > 0:
            residuals_reduced = self.model(y, context_z)
            error_reduced = np.var(residuals_reduced)
        else:
            error_reduced = np.var(y)

        loco_value = error_reduced - error_full

        return loco_value

    def explain_driver(self, data: np.ndarray, target_idx: int, driver_idx: int):
        y = data[:, target_idx]
        driver_x = data[:, driver_idx].reshape(-1, 1)

        num_features = data.shape[1]
        all_indices = list(range(num_features))
        available_indices = [i for i in all_indices if i not in [target_idx, driver_idx]]

        initial_loco = self._hifi_loco_calculator(y, driver_x, context_z=None)

        drivers_red, mi_red, r_n = self._calculate_hoi(
            data, target_idx, driver_idx, available_indices, initial_loco, is_red=True
        )

        drivers_syn, mi_syn, s_n = self._calculate_hoi(
            data, target_idx, driver_idx, available_indices, initial_loco, is_red=False
        )

        return drivers_red, drivers_syn, mi_red, mi_syn, r_n, s_n

    def _calculate_hoi(self, data: np.ndarray, target_idx: int, driver_idx: int,
                       available_indices: list, initial_loco: float, is_red: bool):

        if is_red:
            choice_function = np.argmin
            comparison_function = np.less
        else:
            choice_function = np.argmax
            comparison_function = np.greater

        y = data[:, target_idx]
        driver_x = data[:, driver_idx].reshape(-1, 1)

        mi_sequence = [initial_loco]
        selected_drivers_indices = []
        remaining_indices = available_indices.copy()

        should_continue = True
        while len(remaining_indices) > 0 and should_continue:

            deltas = []
            for candidate_idx in remaining_indices:
                current_context_indices = selected_drivers_indices + [candidate_idx]
                context_z = data[:, current_context_indices]

                loco_val = self._hifi_loco_calculator(y, driver_x, context_z)
                deltas.append(loco_val)

            best_delta_local_idx = choice_function(deltas)
            best_loco_value = deltas[best_delta_local_idx]
            winner_feature_idx = remaining_indices[best_delta_local_idx]

            if self.n_surrogates > 0:
                current_context_with_winner_indices = selected_drivers_indices + [winner_feature_idx]
                context_with_winner = data[:, current_context_with_winner_indices]

                surrogate_locos = []
                for _ in range(self.n_surrogates):
                    context_surrogate = context_with_winner.copy()
                    context_surrogate[:, -1] = np.random.permutation(context_surrogate[:, -1])

                    loco_surr = self._hifi_loco_calculator(y, driver_x, context_surrogate)
                    surrogate_locos.append(loco_surr)

                p_val = np.sum(comparison_function(surrogate_locos, best_loco_value)) / (self.n_surrogates + 1)

                if p_val >= self.alpha:
                    should_continue = False

            if should_continue:
                mi_sequence.append(best_loco_value)
                selected_drivers_indices.append(winner_feature_idx)
                remaining_indices.remove(winner_feature_idx)

        n_drivers = len(mi_sequence)

        return selected_drivers_indices, mi_sequence, n_drivers

    def run_analysis(self, data: np.ndarray, target_idx: int):
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
        dU, dR, dS, dbiv = self._hifi_decomposition(
            all_mi_red, all_mi_syn, all_r_n, all_s_n
        )

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

    def standard_loco(self, data: np.ndarray, target_idx: int):
        print("Calculating standard LOCO...")
        y = data[:, target_idx]
        features = np.delete(data, target_idx, axis=1)
        driver_indices = [i for i in range(data.shape[1]) if i != target_idx]

        residuals_full = self.model(y, features)
        error_full = np.var(residuals_full)

        dLOC = []
        for i in range(features.shape[1]):
            features_reduced = np.delete(features, i, axis=1)
            residuals_reduced = self.model(y, features_reduced)
            error_reduced = np.var(residuals_reduced)

            loco_val = error_reduced - error_full
            dLOC.append(loco_val)

        print("LOCO standard calculated.")
        return np.array(dLOC)

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