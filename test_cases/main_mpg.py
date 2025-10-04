from pyexpat import features

import pandas as pd

from hifi_explainer import HiFiExplainer
from hifi_explainer_masking_background_scalable import HiFiExplainerScalable
from utils import preprocess_for_hifi, linear_regression_model, plot_decomposition, plot_path_matrices, \
    get_linear_model, decision_tree_regression_model, polynomial_regression_model, get_polynomial_model, get_xgboost, \
    plot_decomposition_1

if __name__ == "__main__":
    # URL del dataset
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    # I nomi delle colonne sono nel file .names
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    # Carica con pandas, gestendo gli spazi e i valori mancanti '?'
    df = pd.read_csv(url, names=column_names, na_values='?',
                     comment='\t', sep=" ", skipinitialspace=True)
    df = df.dropna()  # Rimuovi le righe con valori mancanti

    # Gestisci la feature 'Origin' (categorica) con one-hot encoding
    origin = df.pop('Origin')
    df['USA'] = (origin == 1) * 1.0
    df['Europe'] = (origin == 2) * 1.0
    df['Japan'] = (origin == 3) * 1.0

    features_name = [col for col in  column_names if col!="Origin"] + ["USA", "Europe", "Japan"]

    target_name = 'MPG'

    initial_alpha = 0.05
    processed_data, processed_names, target_idx, corrected_alpha = preprocess_for_hifi(
        df.to_numpy(), features_name, target_name, alpha=initial_alpha
    )

    explainer = HiFiExplainer(
        model_function=polynomial_regression_model,
        n_surrogates=1000,
        alpha=corrected_alpha
    )

    explainer_scalable = HiFiExplainerScalable(
        model_factory=get_linear_model,
        n_surrogates=100,
        alpha=corrected_alpha
    )

    explainer_scalable.fit_masking(processed_data, target_idx=target_idx)

    hifi_results = explainer_scalable.run_analysis(processed_data, target_idx=target_idx)
    driver_feature_names = [name for name in processed_names if name != target_name]
    #dloc_results = explainer.standard_loco(processed_data, target_idx=target_idx)
    dloc_results = explainer_scalable.standard_loco()
    plot_decomposition(hifi_results, dloc_results, driver_feature_names)
    plot_path_matrices(hifi_results, driver_feature_names)




