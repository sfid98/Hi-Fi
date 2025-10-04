# File: analizza_con_shap.py

import pandas as pd
import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

print("Analisi con SHAP - Inizio")

# =============================================================================
# PASSO 1: CARICAMENTO E PREPARAZIONE DEI DATI
# =============================================================================
print("\nCaricamento del dataset California Housing...")

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# I nomi delle colonne sono nel file .names
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# Carica con pandas, gestendo gli spazi e i valori mancanti '?'
df = pd.read_csv(url, names=column_names, na_values='?',
                 comment='\t', sep=" ", skipinitialspace=True)
df = df.dropna()  # Rimuovi le righe con valori mancanti

housing = fetch_california_housing()
# Usiamo un DataFrame di Pandas per una migliore visualizzazione con SHAP
#X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
print("Dataset caricato.")

# =============================================================================
# PASSO 2: ADDESTRAMENTO DEL MODELLO "ORACOLO"
# =============================================================================
print("\nAddestramento del modello oracolo (XGBoost)...")
# Usiamo XGBoost perché è potente e l'integrazione con TreeExplainer di SHAP è velocissima
model = xgboost.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X, y)
print("Modello addestrato.")

# =============================================================================
# PASSO 3: CALCOLO DEI VALORI SHAP
# =============================================================================
print("\nCalcolo dei valori SHAP in corso... (potrebbe richiedere un po' di tempo)")
# Usiamo TreeExplainer, che è ottimizzato per i modelli ad albero come XGBoost
explainer = shap.TreeExplainer(model)

# Calcola i valori SHAP per ogni feature e ogni campione
shap_values = explainer.shap_values(X)

# Calcola i valori di interazione SHAP (opzionale, ma utile per sinergia)
shap_interaction_values = explainer.shap_interaction_values(X)
print("Calcolo SHAP completato.")

# =============================================================================
# PASSO 4: VISUALIZZAZIONE E INTERPRETAZIONE DEI RISULTATI
# =============================================================================
print("\nVisualizzazione dei risultati dell'analisi SHAP...")

# --- GRAFICO 1: IMPORTANZA GLOBALE DELLE FEATURE ---
# Questo grafico ti dà la classifica oggettiva dell'importanza di ogni feature.
# È il benchmark principale per il tuo ranking di Hi-Fi (U+R+S).
print("\nGrafico 1: Importanza Globale delle Feature")
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Importanza Globale delle Feature (SHAP)")
plt.show()

# --- GRAFICO 2: IMPATTO DELLE FEATURE (BEESWARM PLOT) ---
# Mostra non solo l'importanza, ma anche la direzione dell'effetto.
# Punti rossi = valore alto della feature, Punti blu = valore basso.
print("\nGrafico 2: Impatto e Direzione delle Feature (Beeswarm)")
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.show()

# --- GRAFICO 3: ANALISI DELLE INTERAZIONI ---
# Questo grafico mostra le interazioni più forti. L'asse y mostra la feature
# principale, e il colore dei punti mostra il valore di una seconda feature
# che interagisce con essa.
# È il benchmark per la tua heatmap di sinergia di Hi-Fi.
print("\nGrafico 3: Principali Interazioni tra Feature (Sinergie)")
plt.figure()
shap.summary_plot(shap_interaction_values, X, show=False)
plt.show()

print("\nAnalisi con SHAP completata.")