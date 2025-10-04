# File: main.py

from hifi_explainer import HiFiExplainer
from hifi_explainer_masking_background_scalable_reverse_greedy import HiFiExplainerScalable
from utils import linear_regression_model, plot_decomposition, \
    plot_path_matrices, get_linear_model

import numpy as np


def generate_custom_toy_data(N=50000, random_seed=42):

    np.random.seed(random_seed)

    segnale_A = np.random.randn(N, 1)
    segnale_B = np.random.randn(N, 1)
    segnale_C = np.random.randn(N, 1)

    noise_y = 0.1 * np.random.randn(N, 1)
    Y = 2 * segnale_A + 3 * segnale_B + 1.5 * segnale_C + noise_y


    X1_ridondante = segnale_A + 0.2 * np.random.randn(N, 1)
    X2_ridondante = segnale_A + 0.2 * np.random.randn(N, 1)

    rumore_sinergico = np.random.randn(N, 1)
    X3_sinergica = segnale_B + rumore_sinergico
    X4_sinergica_helper = -rumore_sinergico

    X5_unica = segnale_C

    X = np.hstack([
        X1_ridondante,
        X2_ridondante,
        X3_sinergica,
        X4_sinergica_helper,
        X5_unica
    ])

    data = np.hstack([Y, X])
    feature_names = ['Y', 'X1_Redundant', 'X2_Redundant', 'X3_Synergistic', 'X4_SynergyHelper', 'X5_Unique']

    print("Dataset generato.")
    return data, feature_names




if __name__ == "__main__":
    data, feature_names = generate_custom_toy_data(N=5000)
    target_name = 'Y'
    target_idx = feature_names.index(target_name)

    explainer = HiFiExplainer(
        model_function=linear_regression_model,
        n_surrogates=100,
        alpha=0.05
    )

    explainer_scalable = HiFiExplainerScalable(
        model_factory=get_linear_model,
        n_surrogates=100,
        alpha=0.05
    )
    explainer_scalable.fit(data, target_idx)

    hifi_results = explainer_scalable.run_analysis(data, target_idx=target_idx)
    driver_feature_names = [name for name in feature_names if name != target_name]
    dloc_results = explainer_scalable.standard_loco()
    plot_decomposition(hifi_results, dloc_results, driver_feature_names)
    plot_path_matrices(hifi_results, driver_feature_names)






