# File: main.py

from hifi_explainer import HiFiExplainer
from to_test.hifi_explainer_masking_background_scalable_masking import HiFiExplainerScalable
from utils import linear_regression_model, plot_decomposition, \
    plot_path_matrices, get_linear_model, polynomial_regression_model

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

def generate_third_order_toy_data(N=50000, random_seed=42):
    np.random.seed(random_seed)

    # --- Sorgenti indipendenti ---
    a = np.random.randn(N, 1)
    b = np.random.randn(N, 1)
    c = np.random.randn(N, 1)
    d = np.random.randn(N, 1)
    e = np.random.randn(N, 1)

    # --- Ridondanza: X1 e X2 derivano dalla stessa sorgente (a) ---
    X1_ridondante = a + 0.2 * np.random.randn(N, 1)
    X2_ridondante = a + 0.2 * np.random.randn(N, 1)

    # --- Sinergia di 2° ordine: X3 e X4 creano un effetto solo insieme ---
    X3_sinergica = b + 0.3 * np.random.randn(N, 1)
    X4_sinergica_helper = -b + 0.3 * np.random.randn(N, 1)

    # --- Feature unica: X5 influenza Y da sola ---
    X5_unica = c + 0.1 * np.random.randn(N, 1)

    # --- Sinergia di 3° ordine: X6, X7, X8 devono coesistere per creare effetto ---
    X6 = d + 0.3 * np.random.randn(N, 1)
    X7 = e + 0.3 * np.random.randn(N, 1)
    X8 = np.random.randn(N, 1)  # completamente indipendente, ma parte dell'interazione

    # --- Target ---
    # Y = combinazione lineare + sinergie di secondo e terzo ordine + rumore
    noise_y = 0.1 * np.random.randn(N, 1)
    Y = (
        2.0 * a +                     # effetto dai ridondanti
        3.0 * X5_unica +              # unicità pura
        4.0 * (X3_sinergica * X4_sinergica_helper) +  # sinergia 2° ordine
        6.0 * (X6 * X7 * X8) +        # sinergia 3° ordine
        noise_y
    )

    # --- Costruzione dataset finale ---
    X = np.hstack([
        X1_ridondante,
        X2_ridondante,
        X3_sinergica,
        X4_sinergica_helper,
        X5_unica,
        X6,
        X7,
        X8
    ])

    data = np.hstack([Y, X])
    feature_names = [
        "Y",
        "X1_Redundant",
        "X2_Redundant",
        "X3_Synergistic",
        "X4_SynergyHelper",
        "X5_Unique",
        "X6_Synergy3_A",
        "X7_Synergy3_B",
        "X8_Synergy3_C",
    ]
    return data, feature_names



if __name__ == "__main__":
    data, feature_names = generate_third_order_toy_data(N=50000)
    target_name = 'Y'
    target_idx = feature_names.index(target_name)

    explainer = HiFiExplainer(
        model_function=polynomial_regression_model,
        n_surrogates=100,
        alpha=0.05
    )

    explainer_scalable = HiFiExplainerScalable(
        model_factory=get_linear_model,
        n_surrogates=100,
        alpha=0.05
    )
    #explainer_scalable.fit(data, target_idx)

    hifi_results = explainer.run_analysis(data, target_idx=target_idx)
    driver_feature_names = [name for name in feature_names if name != target_name]
    dloc_results = explainer.standard_loco(data, target_idx)
    plot_decomposition(hifi_results, dloc_results, driver_feature_names)
    plot_path_matrices(hifi_results, driver_feature_names)






