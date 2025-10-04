# File: analizza_mpg_con_shap.py

import pandas as pd
import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt

from utils import preprocess_wine_data


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


print("Analisi con SHAP sul dataset Auto MPG - Inizio")

raw_data, raw_feature_names = load_wine_data("../datasets/wine+quality/")

target_name = 'quality'

initial_alpha = 0.05

processed_data, processed_names, target_idx, corrected_alpha = preprocess_wine_data(
    raw_data, raw_feature_names, target_name, alpha=initial_alpha
)

X = processed_data[:, 1:]
y = processed_data[:,0]

X = pd.DataFrame(data=X,columns=processed_names[1:])
# =============================================================================
# PASSO 2: ADDESTRAMENTO DEL MODELLO "ORACOLO"
# =============================================================================
print("\nAddestramento del modello oracolo (XGBoost)...")
model = xgboost.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X, y)
print("Modello addestrato.")

# =============================================================================
# PASSO 3: CALCOLO DEI VALORI SHAP
# =============================================================================
print("\nCalcolo dei valori SHAP in corso...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print("Calcolo SHAP completato.")

# =============================================================================
# PASSO 4: VISUALIZZAZIONE E INTERPRETAZIONE DEI RISULTATI
# =============================================================================
print("\nVisualizzazione dei risultati dell'analisi SHAP...")

# --- GRAFICO 1: IMPORTANZA GLOBALE DELLE FEATURE ---
# Questo è il tuo benchmark per la classifica di importanza generale.
print("\nGrafico 1: Importanza Globale delle Feature")
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Importanza Globale delle Feature (SHAP) - Auto MPG")
plt.show()

# --- GRAFICO 2: IMPATTO DELLE FEATURE (BEESWARM PLOT) ---
# Questo grafico ti mostra la direzione dell'impatto (valori alti/bassi).
print("\nGrafico 2: Impatto e Direzione delle Feature (Beeswarm)")
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.show()

# --- GRAFICO 3: ANALISI DELLE INTERAZIONI ---
# Questo è il tuo benchmark per le sinergie. Cerca i grafici fuori dalla
# diagonale con la separazione di colore più netta.
print("\nCalcolo delle interazioni SHAP in corso... (potrebbe richiedere più tempo)")
shap_interaction_values = explainer.shap_interaction_values(X)
print("Calcolo interazioni completato.")

print("\nGrafico 3: Principali Interazioni tra Feature (Sinergie)")
plt.figure()
shap.summary_plot(shap_interaction_values, X, show=False)
plt.show()

print("\nAnalisi con SHAP completata.")