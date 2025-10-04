import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

from hifi_explainer import HiFiExplainer
from hifi_explainer_masking_background_scalable import HiFiExplainerScalable
from utils import preprocess_for_hifi, plot_decomposition, plot_path_matrices, get_linear_model, \
    linear_regression_model, get_polynomial_model, polynomial_regression_model, get_xgboost
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    housing = fetch_california_housing()

    # Unisci features e target in un'unica matrice
    data = np.c_[housing.data, housing.target]

    feature_names = list(housing.feature_names) + ['MedHouseVal']
    target_name = 'MedHouseVal'


    initial_alpha = 0.05
    processed_data, processed_names, target_idx, corrected_alpha = preprocess_for_hifi(
        data, feature_names, target_name, alpha=initial_alpha
    )

    poly_cols = ['Latitude', 'Longitude', 'MedInc']  # applica polinomi solo qui
    linear_cols = [c for c in processed_names if c not in poly_cols]
    explainer = HiFiExplainer(
        model_function=polynomial_regression_model,
        n_surrogates=100,
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
    dloc_results = explainer_scalable.standard_loco()
    #dloc_results = explainer.standard_loco(processed_data,target_idx)

    plot_decomposition(hifi_results, dloc_results, driver_feature_names)
    plot_path_matrices(hifi_results, driver_feature_names)