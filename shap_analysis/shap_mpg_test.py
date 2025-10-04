# File: analizza_mpg_con_shap.py

import pandas as pd
import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt

print("Analisi con SHAP sul dataset Auto MPG - Inizio")

# =============================================================================
# PASSO 1: CARICAMENTO E PREPARAZIONE DEI DATI
# =============================================================================
print("\nCaricamento del dataset Auto MPG...")

# URL del dataset da UCI Machine Learning Repository
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# I nomi delle colonne sono specificati nella documentazione del dataset
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# Usiamo pandas per caricare i dati, gestendo alcune particolarità del file
# sep=" " indica che i dati sono separati da spazi
# skipinitialspace=True gestisce spazi multipli
# na_values='?' dice a pandas di trattare i punti interrogativi come valori mancanti
df = pd.read_csv(url, names=column_names, na_values='?',
                 comment='\t', sep=" ", skipinitialspace=True)

# Rimuovi le poche righe che hanno valori mancanti (in particolare per 'Horsepower')
df = df.dropna()

# La feature 'Origin' è categorica (1: USA, 2: Europe, 3: Japan).
# La trasformiamo in features binarie (one-hot encoding) per darla in pasto al modello.
origin = df.pop('Origin')
df['USA'] = (origin == 1) * 1.0
df['Europe'] = (origin == 2) * 1.0
df['Japan'] = (origin == 3) * 1.0

# Separa le features (X) dal target (y)
X = df.drop('MPG', axis=1)
y = df['MPG']
print("Dataset caricato e preparato.")
print("Features utilizzate:", list(X.columns))

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