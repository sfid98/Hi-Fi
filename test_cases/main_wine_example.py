import pandas as pd

from hifi_explainer_masking_background_scalable_reverse_greedy import HiFiExplainerScalable
from utils import linear_regression_model, preprocess_wine_data, plot_decomposition, plot_path_matrices, \
    get_linear_model, get_xgboost, get_sgd_model, linear_regression_model_partial_fit, get_ridge_model


def load_wine_data(path='./'):
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

        # Use a simple Linear Regression for this dataset
        explainer = HiFiExplainerScalable(model_factory=get_ridge_model, n_surrogates=20)

        explainer.fit(processed_data, target_idx)

        # Hi-Fi analysis
        hifi_results = explainer.run_analysis(processed_data, target_idx=target_idx)

        # Calculate the loco
        #dloc_results = explainer.standard_loco(processed_data, target_idx)
        dloc_results = explainer.standard_loco()


        driver_feature_names = [name for i, name in enumerate(processed_names) if i != target_idx]

        plot_decomposition(hifi_results, dloc_results, driver_feature_names)
        plot_path_matrices(hifi_results, driver_feature_names)


