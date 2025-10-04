# Assicurati di avere Captum installato: pip install captum
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from captum.attr import IntegratedGradients  # <-- Importiamo IntegratedGradients

# (Il codice per generate_data, la definizione e l'addestramento del modello
# sono identici allo script precedente. Assumiamo che il 'model' sia già addestrato)

# --- COPIA/INCOLLA QUI LE PARTI PRECEDENTI ---
# 1. Funzione generate_data()
# 2. Creazione dati X, y, scaler, X_scaled
# 3. Definizione del modello (con nn.GELU) e addestramento
# ... assumiamo che 'model' sia addestrato e 'X' sia il tensore normalizzato.
# -----------------------------------------------

def generate_data(N=10000, random_seed=42):
    """Crea il dataset artificiale con relazioni note."""
    np.random.seed(random_seed)
    sA = np.random.randn(N, 1); sB = np.random.randn(N, 1); sC = np.random.randn(N, 1)
    Y = 2 * sA + 3 * sB + 1.5 * sC + 0.1 * np.random.randn(N, 1)
    X1 = sA + 0.2 * np.random.randn(N, 1); X2 = sA + 0.2 * np.random.randn(N, 1)
    noise_syn = np.random.randn(N, 1); X3 = sB + noise_syn; X4 = -noise_syn
    X5 = sC
    X = np.hstack([X1, X2, X3, X4, X5])
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# -- PASSO 1: Preparazione dei dati e del modello --

X, y = generate_data()
feature_names = ['X1_redundant', 'X2_redundant', 'X3_synergistic', 'X4_synergistic', 'X5_unique']

# --- MODIFICA: Normalizzazione dei dati ---
scaler = StandardScaler()
X_scaled_np = scaler.fit_transform(X.numpy())
X = torch.FloatTensor(X_scaled_np)

n_features = X.shape[1]
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# --- MODIFICA: Usiamo nn.GELU() ---
model = nn.Sequential(
    nn.Linear(n_features, 32),
    nn.GELU(), # <-- Funzione di attivazione più moderna e stabile
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -- PASSO 2: Addestramento del modello --

print("Inizio addestramento...")
# --- MODIFICA: Aumentiamo le epoche ---
num_epochs = 30
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0: # Stampa il progresso ogni 5 epoche
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print("Addestramento completato.")


# --- NUOVO ALGORITMO IG-URS ---

print("\n" + "-" * 40)
print("Inizio calcolo URS con Integrated Gradients...")

# Inizializza l'algoritmo IG
ig = IntegratedGradients(model)
n_features = X.shape[1]

# Seleziona un subset di dati per la spiegazione globale
ref_data = X[:500]  # Spieghiamo 100 campioni per una stima stabile

# Contenitori per i risultati medi
avg_uniqueness = torch.zeros(n_features)
avg_redundancy = torch.zeros(n_features)
avg_synergy = torch.zeros(n_features)

# Baseline di zeri
#baseline = torch.zeros_like(ref_data[0]).unsqueeze(0)
baseline = X.mean(dim=0).unsqueeze(0)

for input_sample in tqdm(ref_data):
    input_sample = input_sample.unsqueeze(0)  # Deve avere una dimensione batch

    # PASSO 1: Calcola l'Unicità (U)
    # Attribuzioni standard di IG rispetto a un baseline di zeri
    uniqueness_scores = ig.attribute(input_sample, baselines=baseline, n_steps=50).squeeze(0).abs().squeeze(0)
    avg_uniqueness += uniqueness_scores

    # Contenitori per R e S di questo campione
    sample_redundancy = torch.zeros(n_features)
    sample_synergy = torch.zeros(n_features)

    # PASSO 2 & 3: Calcola Interazioni, R e S
    for i in range(n_features):
        interaction_sum = 0
        for j in range(n_features):
            if i == j:
                continue

            # Crea il baseline ibrido con la feature j "attiva"
            hybrid_baseline = baseline.clone()
            hybrid_baseline[0, j] = input_sample[0, j]

            # Calcola IG per la feature i, dato j
            ig_i_given_j = ig.attribute(input_sample, baselines=hybrid_baseline, target=None, n_steps=50).squeeze(0)

            # L'interazione è la differenza
            #interaction = ig_i_given_j[i] - uniqueness_scores[i]
            interaction = ig_i_given_j[i]

            if interaction > 0:
                sample_synergy[i] += interaction
            else:
                sample_redundancy[i] += abs(interaction)

    avg_redundancy += sample_redundancy
    avg_synergy += sample_synergy

# Media dei risultati
avg_uniqueness /= len(ref_data)
avg_redundancy /= len(ref_data)
avg_synergy /= len(ref_data)

# Visualizza i risultati
results_df = pd.DataFrame({
    'Uniqueness (U)': avg_uniqueness.detach().numpy(),
    'Redundancy (R)': avg_redundancy.detach().numpy(),
    'Synergy (S)': avg_synergy.detach().numpy()
}, index=feature_names)

print(results_df)
results_df['Total Importance'] = results_df.sum(axis=1)
results_perc = results_df.loc[:, ['Uniqueness (U)', 'Redundancy (R)', 'Synergy (S)']].div(
    results_df['Total Importance'], axis=0)

print("\nContribuzione Percentuale di U, R, S (metodo IG):")
print((results_perc * 100).round(1))
print("-" * 40)