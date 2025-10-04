import pandas as pd

from to_test.hifi_explainer_masking_background_scalable_shap import HiFiShapleyExplainer
from utils import preprocess_wine_data, plot_decomposition, get_linear_model


def load_wine_data(path='./'):
    """Load wine dataset."""
    print("Loading wine dataset...")
    try:
        data_red = pd.read_csv(path + 'winequality-red.csv', sep=';')
        data_white = pd.read_csv(path + 'winequality-white.csv', sep=';')
    except FileNotFoundError:
        print("\nError while loading the dataset!")
        return None, None


    data = pd.concat([data_red, data_white], ignore_index=True)
    data.columns = [col.replace(' ', '_') for col in data.columns]

    x_names = list(data.columns)
    x_data = data.values

    print("Dataset loaded.")
    return x_data, x_names


if __name__ == "__main__":

    raw_data, raw_feature_names = load_wine_data("/Users/stanislao/Desktop/Hi-Fi/datasets/wine+quality/")

    if raw_data is not None:
        target_name = 'quality'

        initial_alpha = 0.05

        processed_data, processed_names, target_idx, corrected_alpha = preprocess_wine_data(
            raw_data, raw_feature_names, target_name, alpha=initial_alpha
        )

        explainer = HiFiShapleyExplainer(
            model_factory=get_linear_model        )

        # 3. Addestra il modello e crea il background dataset
        explainer.fit(processed_data, target_idx=target_idx)

        # 4. Esegui l'analisi completa
        hifi_shap_results = explainer.run_analysis()

        # 5. Visualizza i risultati
        driver_feature_names = [name for name in processed_names if name != target_name]
        plot_decomposition(hifi_shap_results, [], driver_feature_names)
        print("\nEsecuzione completata. Decommenta la funzione di plot per visualizzare.")

        dU, dR, dS, dshap = hifi_shap_results['unique'], hifi_shap_results['redundancy'], hifi_shap_results['synergy'], hifi_shap_results['dshap']
        print("\nRisultati della decomposizione (U, R, S):")
        for i, name in enumerate(driver_feature_names):
            print(f"- {name}: U={dU[i]:.4f}, R={dR[i]:.4f}, S={dS[i]:.4f}, Shap={dshap[i]:.4f}")