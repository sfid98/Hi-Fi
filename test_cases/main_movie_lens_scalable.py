from to_test.hifi_explainer_masking_scalable import HiFiExplainerScalable
from utils import get_linear_model, preprocess_for_hifi, plot_path_matrices, plot_decomposition, get_xgboost
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

def load_movielens_data():
    """
    Carica, unisce e pre-elabora il dataset MovieLens 1M.
    """
    print("Caricamento del dataset MovieLens 1M...")
    try:
        # Carica i dati - notare il separatore '::'
        users = pd.read_csv('../datasets/ml-1m/users.dat', sep='::', header=None, engine='python',
                            names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        ratings = pd.read_csv('../datasets/ml-1m/ratings.dat', sep='::', header=None, engine='python',
                              names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        movies = pd.read_csv('../datasets/ml-1m/movies.dat', sep='::', header=None, engine='python',
                             names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    except FileNotFoundError:
        print("\n!!! ERRORE: File del dataset non trovati. !!!")
        print("Assicurati di aver scaricato e decompresso 'ml-1m.zip' in questa cartella.")
        return None, None

    # Unisci i dati in un unico DataFrame
    df = pd.merge(pd.merge(ratings, users), movies)

    # --- Feature Engineering ---
    # 1. Converti 'Gender' in binario (0/1)
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'M' else 0)

    # 2. One-hot encoding dei generi
    genres_dummies = df['Genres'].str.get_dummies('|')
    df = pd.concat([df, genres_dummies], axis=1)

    # 3. Seleziona le colonne finali per il modello
    feature_cols = ['Age', 'Gender', 'Occupation'] + list(genres_dummies.columns)
    target_col = 'Rating'

    final_df = df[[target_col] + feature_cols]

    print(f"Dataset creato con {final_df.shape[0]} righe e {final_df.shape[1] - 1} features.")
    return final_df.values, list(final_df.columns)

if __name__ == "__main__":
    # 1. Carica e prepara i dati
    raw_data, feature_names = load_movielens_data()

    if raw_data is not None:
        # Usa un campione per velocizzare
        sample_size = 10000
        sample_indices = np.random.choice(raw_data.shape[0], sample_size, replace=False)
        raw_data = raw_data[sample_indices]

        target_name = 'Rating'
        processed_data, processed_names, target_idx, corrected_alpha = preprocess_for_hifi(
            raw_data, feature_names, target_name, alpha=0.05
        )

        # 2. Istanzia lo spiegatore SCALABILE
        explainer = HiFiExplainerScalable(
            model_factory=get_xgboost,
            n_surrogates=100,
            alpha=corrected_alpha
        )

        regressor = DecisionTreeRegressor(max_depth=4, random_state=42)

        model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1)

        X_train, X_test, y_train, y_test = train_test_split(processed_data[:,1:], processed_data[:,0], test_size=0.3, random_state=42)

        linear_model = get_linear_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        print("Mse: ", mse)
        # 3. Addestra il modello UNA SOLA VOLTA
        explainer.fit(processed_data, target_idx=target_idx)

        # 4. Esegui l'analisi Hi-Fi (che ora userà solo forward-pass)
        # Nota: La logica completa di _calculate_hoi e run_analysis va implementata
        # nella classe HiFiExplainerScalable per far funzionare questa riga.
        # hifi_results = explainer.run_analysis()

        # Esempio di come usare il calcolatore di LOCO scalabile
        hifi_results = explainer.run_analysis(processed_data, target_idx=target_idx)

        print(hifi_results)

        # 5. Calcola il LOCO standard

        # 6. Prepara le etichette per i grafici
        driver_feature_names = [name for name in processed_names if name != target_name]

        # 7. Visualizza i risultati (DECOMMENTA QUANDO HAI COPIATO LE FUNZIONI)
        print("\nPer visualizzare i grafici, copia le funzioni di plot e decommenta le righe seguenti.")
        dloc_results = explainer.standard_loco()

        plot_decomposition(hifi_results, dloc_results, driver_feature_names)

        plot_path_matrices(hifi_results, driver_feature_names)