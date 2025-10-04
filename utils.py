import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge

def get_linear_model():
    return LinearRegression()


# --- NUOVA FUNZIONE DA AGGIUNGERE ---
def get_ridge_model(alpha=2.0, random_seed=42):
    """
    Restituisce un modello di regressione Ridge (lineare con regolarizzazione L2).
    L'alpha controlla la forza della regolarizzazione.
    """
    return Ridge(alpha=alpha, random_state=random_seed)

def get_decision_tree():
    return DecisionTreeRegressor(max_depth=4, random_state=42)

def get_poly_features(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    X_poly = poly.fit_transform(X)
    return X_poly

def get_sgd_model(random_seed=42):

    return SGDRegressor(
        loss='squared_error',
        penalty=None,
        learning_rate='constant',
        eta0=0.01,
        random_state=random_seed
    )

def get_polynomial_model(degree=2):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
        ('regressor', LinearRegression())
    ])


def get_xgboost():
    return XGBRegressor(random_state=42)


def polynomial_regression_model(y: np.ndarray, X: np.ndarray) -> np.ndarray:

    if X.ndim == 1:
        X = X.reshape(-1, 1)


    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    predictions = model.predict(X_poly)
    residuals = y - predictions

    return residuals


def linear_regression_model(y, X):
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions
    return residuals


def linear_regression_model_partial_fit(y, X, random_seed=42):
    model =SGDRegressor(
        loss='squared_error',
        penalty=None,
        learning_rate='constant',
        eta0=0.0001,
        max_iter=10000
    )
    for _ in range(10):
        model.partial_fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions
    return residuals

def decision_tree_regression_model(y, X):
    model = DecisionTreeRegressor()
    model.fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions
    return residuals

def preprocess_for_hifi(data: np.ndarray, feature_names: list, target_name: str, alpha: float):
    print("Preprocessing...")

    df = pd.DataFrame(data, columns=feature_names)

    df.dropna(inplace=True)

    y = df[target_name]
    X = df.drop(columns=[target_name])

    X = X.sort_index(axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    df_processed = pd.concat([y, X_scaled], axis=1)

    num_tests = X.shape[1]
    corrected_alpha = alpha / num_tests

    processed_data_np = df_processed.values
    processed_feature_names = list(df_processed.columns)
    new_target_idx = 0

    return processed_data_np, processed_feature_names, new_target_idx, corrected_alpha

def preprocess_wine_data(data: np.ndarray, feature_names: list, target_name: str, alpha: float):

    df = pd.DataFrame(data, columns=feature_names)

    df.dropna(inplace=True)

    y = df[target_name]
    X = df.drop(columns=[target_name, "alcohol"])


    X = X.sort_index(axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    df_processed = pd.concat([y, X_scaled], axis=1)

    num_tests = X.shape[1]
    corrected_alpha = alpha / num_tests

    processed_data_np = df_processed.values
    processed_feature_names = list(df_processed.columns)
    new_target_idx = 0

    return processed_data_np, processed_feature_names, new_target_idx, corrected_alpha

def plot_decomposition(results, dloc, feature_names):
    dU, dR, dS, dbiv = results['unique'], results['redundancy'], results['synergy'], results['pairwise']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    indices = np.arange(len(feature_names))
    axes[0].bar(indices, dU, label='unique', color='C0')
    axes[0].bar(indices, dR, bottom=dU, label='redundancy', color='C3')
    axes[0].bar(indices, dS, bottom=dU + dR, label='synergy', color='C2')

    axes[0].set_ylabel('Feature Importance')
    axes[0].set_title('Hi-Fi Decomposition')
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[0].legend()

    width = 0.35
    axes[1].bar(indices - width / 2, dloc, width, label='LOCO', color='C1')
    axes[1].bar(indices + width / 2, dbiv, width, label='pairwise', color='C0')

    axes[1].set_title('LOCO vs Pairwise')
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_decomposition_1(results, dloc, feature_names):

    dU, dR, dS, dbiv = results['unique'], results['redundancy'], results['synergy'], results['pairwise']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analisi della Decomposizione Hi-Fi', fontsize=16)

    indices = np.arange(len(feature_names))

    axes[0, 0].bar(indices, dU, label='Unique (U)', color='C0')
    axes[0, 0].set_title('Contributo Unico')
    axes[0, 0].set_ylabel('Importanza della Feature')
    axes[0, 0].set_xticks(indices)
    axes[0, 0].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    axes[0, 1].bar(indices, dR, label='Redundancy (R)', color='C3')
    axes[0, 1].set_title('Contributo di Ridondanza')
    axes[0, 1].set_xticks(indices)
    axes[0, 1].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1, 0].bar(indices, dS, label='Synergy (S)', color='C2')
    axes[1, 0].set_title('Contributo di Sinergia')
    axes[1, 0].set_ylabel('Importanza della Feature')
    axes[1, 0].set_xticks(indices)
    axes[1, 0].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    width = 0.35
    axes[1, 1].bar(indices - width / 2, dloc, width, label='LOCO', color='C1')
    axes[1, 1].bar(indices + width / 2, dbiv, width, label='Pairwise', color='C0')
    axes[1, 1].set_title('LOCO vs Pairwise')
    axes[1, 1].set_xticks(indices)
    axes[1, 1].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
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

    im2 = axes[1].imshow(synergy_path, cmap='hot_r', vmin=0, vmax=max_val)
    axes[1].set_title('Synergy')
    axes[1].set_xticks(np.arange(len(feature_names)))
    axes[1].set_yticks(np.arange(len(feature_names)))
    axes[1].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[1].set_yticklabels(feature_names)

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.6)

    fig.tight_layout()
    plt.show()
