# File: main_movie_lens.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hifi_explainer import HiFiExplainer
from utils import linear_regression_model, preprocess_for_hifi, plot_decomposition, plot_path_matrices


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
    raw_data, feature_names = load_movielens_data()

    if raw_data is not None:
        # **CONSIGLIO**: Per il primo test, usa un campione più piccolo!
        sample_size = 50000
        sample_indices = np.random.choice(raw_data.shape[0], sample_size, replace=False)
        raw_data = raw_data[sample_indices]
        # print(f"\nUtilizzo di un campione di {sample_size} righe per un test veloce.")

        # 2. Pre-elabora i dati (riordina, standardizza, etc.)
        target_name = 'Rating'
        initial_alpha = 0.05
        processed_data, processed_names, target_idx, corrected_alpha = preprocess_for_hifi(
            raw_data, feature_names, target_name, alpha=initial_alpha
        )

        # 3. Istanzia lo spiegatore Hi-Fi
        explainer = HiFiExplainer(
            model_function=linear_regression_model,
            n_surrogates=100,  # Usa un numero adeguato di surrogati
            alpha=corrected_alpha
        )

        # 4. Esegui l'analisi completa
        hifi_results = explainer.run_analysis(processed_data, target_idx=target_idx)

        # 5. Calcola il LOCO standard
        dloc_results = explainer.standard_loco(processed_data, target_idx=target_idx)

        # 6. Prepara le etichette per i grafici
        driver_feature_names = [name for name in processed_names if name != target_name]

        # 7. Visualizza i risultati (DECOMMENTA QUANDO HAI COPIATO LE FUNZIONI)
        print("\nPer visualizzare i grafici, copia le funzioni di plot e decommenta le righe seguenti.")
        plot_decomposition(hifi_results, dloc_results, driver_feature_names)
        plot_path_matrices(hifi_results, driver_feature_names)