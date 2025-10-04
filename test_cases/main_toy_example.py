# File: main.py

import numpy as np
import matplotlib.pyplot as plt

from grad_hifi import X_data
from hifi_explainer import HiFiExplainer
from utils import polynomial_regression_model, linear_regression_model


def generate_toy_data(N=80000, random_seed=42):
    np.random.seed(random_seed)  # Per la riproducibilità

    mu_syn = [0, 0, 0]
    sigma_syn = [[1, 0.5, 0.3], [0.5, 1, -0.5], [0.3, -0.5, 1]]
    xsyn = np.random.multivariate_normal(mu_syn, sigma_syn, N)

    mu_red = [0, 0, 0]
    sigma_red = [[1, 0.5, 0.3], [0.5, 1, 0.5], [0.3, 0.5, 1]]
    xred = np.random.multivariate_normal(mu_red, sigma_red, N)

    xunique = np.random.randn(N, 1)
    xint = np.random.randn(N, 2)
    noise = 0.05 * np.random.randn(N, 1)


    X = np.zeros((N, 8))
    X[:, 0] = (xsyn[:, 0] + xred[:, 0] + xunique.flatten() + (xint[:, 0] * xint[:, 1]) + noise.flatten())
    X[:, 1:3] = xsyn[:, 1:3]
    X[:, 3:5] = xred[:, 1:3]
    X[:, 5] = xunique.flatten()
    X[:, 6:8] = xint

    return X


def generate_custom_toy_data(N=50000, random_seed=42):
    np.random.seed(random_seed)

    segnale_A = np.random.randn(N, 1)
    segnale_B = np.random.randn(N, 1)
    segnale_C = np.random.randn(N, 1)

    noise_y = 0.1 * np.random.randn(N, 1)
    Y = 2 * segnale_A + 3 * segnale_B + 4.5 * segnale_C + noise_y


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



def plot_decomposition(results, dloc, feature_names):
    dU, dR, dS, dbiv = results['unique'], results['redundancy'], results['synergy'], results['pairwise']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Grafico 1: Scomposizione Hi-Fi
    indices = np.arange(len(feature_names))
    axes[0].bar(indices, dU, label='unique', color='C0')
    axes[0].bar(indices, dR, bottom=dU, label='redundancy', color='C3')
    axes[0].bar(indices, dS, bottom=dU + dR, label='synergy', color='C2')

    axes[0].set_ylabel('Feature Importance')
    axes[0].set_title('Hi-Fi Decomposition')
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(feature_names)
    axes[0].legend()

    width = 0.35
    axes[1].bar(indices - width / 2, dloc, width, label='LOCO', color='C1')
    axes[1].bar(indices + width / 2, dbiv, width, label='pairwise', color='C0')

    axes[1].set_title('LOCO vs Pairwise')
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(feature_names)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def plot_path_matrices(results, feature_names):

    redundancy_path = results['redundancy_path']
    synergy_path = results['synergy_path']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    max_val = max(np.max(redundancy_path), np.max(synergy_path))

    im1 = axes[0].imshow(redundancy_path, cmap='hot_r', vmin=0, vmax=max_val)
    axes[0].set_title('Redundancy')
    axes[0].set_xticks(np.arange(len(feature_names)))
    axes[0].set_yticks(np.arange(len(feature_names)))
    axes[0].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[0].set_yticklabels(feature_names)

    # Heatmap per la Sinergia
    im2 = axes[1].imshow(synergy_path, cmap='hot_r', vmin=0, vmax=max_val)
    axes[1].set_title('Synergy')
    axes[1].set_xticks(np.arange(len(feature_names)))
    axes[1].set_yticks(np.arange(len(feature_names)))
    axes[1].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[1].set_yticklabels(feature_names)

    # Aggiungi una colorbar comune
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.6)

    fig.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     # 1. Genera i dati del toy example
#     X_data = generate_toy_data()
#     target_idx = 0
#     feature_names = [f'X{i}' for i in range(1, 8)]
#
#     # 2. Istanzia lo 'spiegatore' Hi-Fi
#     # Nota: usiamo pochi surrogati per rendere il test veloce
#     explainer = HiFiExplainer(model_function=polynomial_regression_model, n_surrogates=10, alpha=0.05)
#
#     # 3. Esegui l'analisi Hi-Fi completa
#     hifi_results = explainer.run_analysis(X_data, target_idx=target_idx)
#
#     # 4. Calcola il LOCO standard per confronto
#     dloc_results = explainer.standard_loco(X_data, target_idx=target_idx)
#
#     # 5. Visualizza i risultati
#     plot_decomposition(hifi_results, dloc_results, feature_names)

if __name__ == "__main__":
    #X_data = generate_toy_data()
    X_data = generate_custom_toy_data()
    target_idx = 0


    explainer = HiFiExplainer(model_function=linear_regression_model, n_surrogates=0)

    num_features = X_data.shape[1]
    driver_indices = [i for i in range(num_features) if i != target_idx]

    all_drivers_red, all_drivers_syn = [], []
    all_mi_red, all_mi_syn = [], []

    print("Avvio dell'analisi Hi-Fi per raccogliere i risultati grezzi...")
    for i, driver_idx in enumerate(driver_indices):
        print(f"  Analisi del driver {i + 1}/{len(driver_indices)} (feature indice: {driver_idx})...")
        drivers_red, drivers_syn, mi_red, mi_syn, _, _ = explainer.explain_driver(
            X_data, target_idx, driver_idx
        )
        all_drivers_red.append(drivers_red)
        all_drivers_syn.append(drivers_syn)
        all_mi_red.append(mi_red)
        all_mi_syn.append(mi_syn)

    r_n_hardcoded = [1, 1, 2, 2, 1, 1, 1]
    s_n_hardcoded = [2, 2, 1, 1, 1, 2, 2]

    print("Calcolo della scomposizione con r_n e s_n forzati...")
    dU, dR, dS, dbiv = explainer._hifi_decomposition(
        all_mi_red, all_mi_syn, r_n_hardcoded, s_n_hardcoded
    )

    redundancy_path = explainer._calculate_path_matrix(all_drivers_red, all_mi_red, driver_indices, is_red = True)
    synergy_path = explainer._calculate_path_matrix(all_drivers_syn, all_mi_syn, driver_indices, is_red = False)

    hifi_results = {
        'unique': dU, 'redundancy': dR, 'synergy': dS, 'pairwise': dbiv,
        'driver_indices': driver_indices,
        'redundancy_path': redundancy_path, 'synergy_path': synergy_path
    }

    dloc_results = explainer.standard_loco(X_data, target_idx=target_idx)

    feature_names = [f'X{i + 1}' for i in driver_indices]
    plot_decomposition(hifi_results, dloc_results, feature_names)
    plot_path_matrices(hifi_results, feature_names)



