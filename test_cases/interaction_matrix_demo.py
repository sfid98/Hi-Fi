# Esempio ipotetico
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from interaction_matrix import HiFiDiagnosticMatrix
from utils import preprocess_wine_data, preprocess_for_hifi
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

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

import numpy as np

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


def analyze_wine():
    raw_data, raw_feature_names = load_wine_data("/Users/stanislao/Desktop/Hi-Fi/datasets/wine+quality/")
    initial_alpha = 0.05
    target_name = 'quality'

    processed_data, processed_names, target_idx, corrected_alpha = preprocess_wine_data(
        raw_data, raw_feature_names, target_name, alpha=initial_alpha
    )

    return processed_data, processed_names, target_idx, target_name

def analyze_toy():
    processed_data, processed_names = generate_custom_toy_data(N=50000)
    #processed_data, processed_names  = generate_third_order_toy_data(N=50000)
    target_name = 'Y'
    target_idx = 0
    return processed_data, processed_names, target_idx, target_name

def analyze_california():
    housing = fetch_california_housing()

    # Unisci features e target in un'unica matrice
    data = np.c_[housing.data, housing.target]

    feature_names = list(housing.feature_names) + ['MedHouseVal']
    target_name = 'MedHouseVal'


    initial_alpha = 0.05
    processed_data, processed_names, target_idx, corrected_alpha = preprocess_for_hifi(
        data, feature_names, target_name, alpha=initial_alpha
    )
    return processed_data, processed_names, target_idx, target_name


#processed_data, processed_names, target_idx, target_name = analyze_wine()

processed_data, processed_names, target_idx, target_name = analyze_toy()
#processed_data, processed_names, target_idx, target_name = analyze_california()


y_train = processed_data[:, target_idx]

feature_indices = [i for i in range(processed_data.shape[1]) if i != target_idx]
X_train = processed_data[:, feature_indices]

#X, y = data.data, data.target



poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
X_poly = poly.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train, y_train)


#model = LinearRegression()
#model.fit(X_train, y_train)




def sample_background(X_train: np.ndarray, background_samples: int = 50, random_state=None) -> np.ndarray:
    """
    Restituisce un sottoinsieme (background) campionato da X_train.
    Se background_samples > n_train allora si campiona con replacement.
    """
    rng = np.random.RandomState(random_state)
    n = X_train.shape[0]
    replace = background_samples > n
    idx = rng.choice(n, size=background_samples, replace=replace)
    return X_train[idx]


def _predict_with_mask(model,
                       X_source: np.ndarray,
                       y: np.ndarray,
                       mapped_active_indices: list,
                       X_background: np.ndarray) -> np.ndarray:
    n_source, n_features = X_source.shape
    n_bg = X_background.shape[0]

    # Caso semplice: se non ci sono feature attive ritorna la predizione costante (mean(y))
    if not mapped_active_indices:
        return np.full(shape=(n_source,), fill_value=np.mean(y))

    # costruiamo in loop (memoria-friendly). Se vuoi più velocità usa la versione vectorizzata sotto.
    preds_acc = np.empty((n_bg, n_source))
    mask_bool = np.zeros(n_features, dtype=bool)
    mask_bool[mapped_active_indices] = True

    for i in range(n_bg):
        bg_row = X_background[i]                     # shape (n_features,)
        X_masked = np.tile(bg_row, (n_source, 1))    # shape (n_source, n_features)
        # Sostituisci le colonne "attive" con quelle reali di X_source
        X_masked[:, mask_bool] = X_source[:, mask_bool]
        preds = model.predict(X_masked)              # shape (n_source,)
        preds_acc[i, :] = preds

    final_preds = preds_acc.mean(axis=0)             # media sulle n_bg predizioni
    return final_preds


def _predict_with_mask_1(model,
                       X_source: np.ndarray,
                       y: np.ndarray,
                       mapped_active_indices: list,
                       X_background: np.ndarray) -> np.ndarray:
    n_source, n_features = X_source.shape
    n_bg = X_background.shape[0]

    # Caso semplice: se non ci sono feature attive ritorna la predizione costante (mean(y))
    if not mapped_active_indices:
        return np.full(shape=(n_source,), fill_value=np.mean(y))

    # costruiamo in loop (memoria-friendly). Se vuoi più velocità usa la versione vectorizzata sotto.
    mask_bool = np.zeros(n_features, dtype=bool)
    mask_bool[mapped_active_indices] = True

    X_masked = np.zeros((X_source.shape[0],X_source.shape[1]))
    X_masked[:, mask_bool] = X_source[:, mask_bool]
    preds = model.predict(X_masked)              # shape (n_source,)

    return preds

def _predict_with_mask_vectorized(model,
                                  X_source: np.ndarray,
                                  y: np.ndarray,
                                  mapped_active_indices: list,
                                  X_background: np.ndarray,
                                  batch_predict_size: int = 100000) -> np.ndarray:
    """
    Versione vectorizzata che costruisce in blocco tutte le matrici masked e richiama model.predict
    una sola volta (o in pochi chunk per evitare OOM).
    Utile se n_bg * n_source non è troppo grande.
    """
    n_source, n_features = X_source.shape
    n_bg = X_background.shape[0]

    if not mapped_active_indices:
        return np.full(shape=(n_source,), fill_value=np.mean(y))

    mask_bool = np.zeros(n_features, dtype=bool)
    mask_bool[mapped_active_indices] = True

    # creeremo una grande matrice di shape (n_bg * n_source, n_features)
    total_rows = n_bg * n_source
    preds_all = np.empty((total_rows,), dtype=float)

    # processiamo in chunk per non saturare memoria
    start = 0
    while start < n_bg:
        # chunk di background
        end = min(n_bg, start + 1000)   # chunk di righe background (tuneable)
        bg_chunk = X_background[start:end]   # shape (chunk_size, n_features)
        chunk_size = bg_chunk.shape[0]
        # costruisci matrice replicando ogni bg_row per tutte le source rows
        # shape (chunk_size * n_source, n_features)
        X_chunk = np.repeat(bg_chunk, repeats=n_source, axis=0)
        # replace colonne attive
        X_chunk[:, mask_bool] = np.tile(X_source[:, mask_bool], (chunk_size, 1))
        # predici in sottobatch
        n_rows = X_chunk.shape[0]
        step = max(1, batch_predict_size // n_features)
        preds_chunk = []
        for s in range(0, n_rows, step):
            e = min(n_rows, s + step)
            preds_chunk.append(model.predict(X_chunk[s:e]))
        preds_chunk = np.concatenate(preds_chunk, axis=0)  # length = n_rows
        # inserisci nell'array globale
        idx0 = start * n_source
        idx1 = idx0 + n_rows
        preds_all[idx0:idx1] = preds_chunk
        start = end

    # ora preds_all è organizzato in blocchi di size n_source
    preds_all = preds_all.reshape((n_bg, n_source))
    final_preds = preds_all.mean(axis=0)
    return final_preds


def loco_func(model,
              X_train: np.ndarray,
              y_train: np.ndarray,
              driver_idx: int,
              context_indices: list,
              background_samples: int = 50,
              random_state: int = None,
              use_vectorized: bool = False) -> float:
    """
    Calcola LOCO per un driver dato il contesto, usando mask-from-background approach.
    - driver_idx: indice della feature 'driver' (0..d-1)
    - context_indices: lista di indici da considerare attivi (non mascherati)
    Ritorna loco_value = error_reduced - error_full
    """
    # Usa lo stesso background per entrambi i calcoli per ridurre varianza
    X_bg = sample_background(X_train, background_samples, random_state)

    mapped_context = list(context_indices)  # assumiamo indici "globali" su X_train

    # preds reduced (context only)
    if use_vectorized:
        preds_reduced = _predict_with_mask_vectorized(model, X_train, y_train, mapped_context, X_bg)
    else:
        preds_reduced = _predict_with_mask(model, X_train, y_train, mapped_context, X_bg)
    error_reduced = np.mean((y_train - preds_reduced) ** 2)

    # preds full (driver + context)
    mapped_full = [driver_idx] + mapped_context
    if use_vectorized:
        preds_full = _predict_with_mask_vectorized(model, X_train, y_train, mapped_full, X_bg)
    else:
        preds_full = _predict_with_mask(model, X_train, y_train, mapped_full, X_bg)
    error_full = np.mean((y_train - preds_full) ** 2)

    loco_value = error_reduced - error_full
    return loco_value


def loco_func(model,
              X_train: np.ndarray,
              y_train: np.ndarray,
              driver_idx: int,
              context_indices: list,
              background_samples: int = 50,
              random_state: int = None,
              use_vectorized: bool = False) -> float:
    """
    Calcola LOCO per un driver dato il contesto, usando mask-from-background approach.
    - driver_idx: indice della feature 'driver' (0..d-1)
    - context_indices: lista di indici da considerare attivi (non mascherati)
    Ritorna loco_value = error_reduced - error_full
    """
    # Usa lo stesso background per entrambi i calcoli per ridurre varianza
    X_bg = sample_background(X_train, background_samples, random_state)

    mapped_context = list(context_indices)  # assumiamo indici "globali" su X_train

    # preds reduced (context only)
    if use_vectorized:
        preds_reduced = _predict_with_mask_vectorized(model, X_train, y_train, mapped_context, X_bg)
    else:
        preds_reduced = _predict_with_mask_1(model, X_train, y_train, mapped_context, X_bg)
    error_reduced = np.mean((y_train - preds_reduced) ** 2)

    # preds full (driver + context)
    mapped_full = [driver_idx] + mapped_context
    if use_vectorized:
        preds_full = _predict_with_mask_vectorized(model, X_train, y_train, mapped_full, X_bg)
    else:
        preds_full = _predict_with_mask_1(model, X_train, y_train, mapped_full, X_bg)
    error_full = np.mean((y_train - preds_full) ** 2)

    loco_value = error_reduced - error_full
    return loco_value




feature_indices = [i for i in range(processed_data.shape[1]) if i != target_idx]

hifi_diag = HiFiDiagnosticMatrix(model, X_train, y_train,
                                 loco_func=lambda model_, X_, y_, driver_idx, ctx:
                                 loco_func(model_, X_, y_, driver_idx, ctx),
                                 predict_func = lambda model_ , X_source_,y_,mapped_active_indices_,X_background_:
                                _predict_with_mask(model,X_source_,y_,mapped_active_indices_,X_background_))
D = hifi_diag.compute_matrix_D()
summary = hifi_diag.compute_redundancy_synergy()
print(summary)
feature_names = [names for names in processed_names if names != target_name]

T_syn, T_red = hifi_diag.compute_third_order_effects()
max_syn_val = np.max(T_syn)
i, j, k = np.unravel_index(np.argmax(T_syn), T_syn.shape)

print(f"\n--- Analisi del 3° Ordine ---")
print(f"Interazione Sinergica a 3 più forte trovata (valore: {max_syn_val:.4f}):")
print(f"Driver: {feature_names[i]}, Partner 1: {feature_names[j]}, Partner 2: {feature_names[k]}")


hifi_diag.plot_heatmap([names for names in processed_names if names != target_name])


result = hifi_diag.compute_synergy_redundancy(D, feature_names)
print(result)

summary_full = hifi_diag.compute_full_decomposition(feature_names)

print("\n--- Scomposizione Finale (con effetti del 2° e 3° ordine) ---")
print(summary_full)
