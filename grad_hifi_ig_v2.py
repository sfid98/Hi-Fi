import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from captum.attr import IntegratedGradients

# (Assumiamo che il modello sia già addestrato e i dati 'X' preparati e normalizzati)
# ...
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
#model = nn.Sequential(
#    nn.Linear(n_features, 32),
#    nn.ReLU(), # <-- Funzione di attivazione più moderna e stabile
#    nn.Linear(32, 1)
#)

model = nn.Linear(n_features, 1, bias=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -- PASSO 2: Addestramento del modello --

print("Inizio addestramento...")
# --- MODIFICA: Aumentiamo le epoche ---
num_epochs = 50
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

print("\n" + "-" * 40)
print("Inizio calcolo URS con Campionamento di Contesti...")

ig = IntegratedGradients(model)
n_features = X.shape[1]
ref_data = X[:100]  # Usiamo meno campioni perché il calcolo è più lento
M_samples = 50  # Numero di contesti casuali da campionare per feature

# Contenitori per i risultati medi globali
avg_uniqueness = torch.zeros(n_features)
# Matrice per accumulare le interazioni: interaction_effects[i, j] = effetto di j su i
interaction_effects = torch.zeros(n_features, n_features)

baseline = X.mean(dim=0).unsqueeze(0)

for input_sample in tqdm(ref_data, desc="Analizzando i campioni"):
    input_sample = input_sample.unsqueeze(0)

    # Calcola l'attribuzione di base (usata sia per U che per calcolare il delta)
    base_attributions = ig.attribute(input_sample, baselines=baseline, n_steps=50).squeeze(0)
    avg_uniqueness += base_attributions.abs()

    # Ciclo su ogni feature 'i' per calcolare come le altre la influenzano
    for i in range(n_features):
        feature_indices = list(range(n_features))
        other_features = [j for j in feature_indices if j != i]

        # Campiona M contesti
        for _ in range(M_samples):
            # Crea un contesto C: sottoinsieme casuale delle altre feature
            context_size = np.random.randint(1, n_features)  # Dimensione del contesto
            context = np.random.choice(other_features, size=context_size, replace=False)

            # Crea il baseline ibrido
            hybrid_baseline = baseline.clone()
            hybrid_baseline[0, context] = input_sample[0, context]

            # Calcola IG di i dato il contesto C
            ig_i_given_c = ig.attribute(input_sample, baselines=hybrid_baseline, n_steps=50).squeeze(0)

            # Calcola l'effetto dell'intero contesto su i
            delta = ig_i_given_c[i] - base_attributions[i]
            # Assegna questo effetto a ogni feature j che era nel contesto
            # Dividiamo per la dimensione del contesto per non favorire le feature che appaiono
            # in contesti più grandi
            if len(context) > 0:
                interaction_effects[i, context] += delta / len(context)

# Media dei risultati sul numero di campioni di riferimento
avg_uniqueness /= len(ref_data)
interaction_effects /= len(ref_data)

# Calcola R e S totali per ogni feature j (come influenza ALTRE feature)
total_synergy = torch.zeros(n_features)
total_redundancy = torch.zeros(n_features)

for j in range(n_features):
    # L'effetto di j è la somma di come ha influenzato tutte le altre feature i
    effects_of_j = interaction_effects[:, j]
    total_synergy[j] = torch.clamp(effects_of_j, min=0).sum()
    total_redundancy[j] = torch.abs(torch.clamp(effects_of_j, max=0).sum())

results_df = pd.DataFrame({
    'Uniqueness (U)': avg_uniqueness.detach().numpy(),
    'Redundancy (R)': total_redundancy.detach().numpy(),
    'Synergy (S)': total_synergy.detach().numpy()
}, index=feature_names)

results_df['Total Importance'] = results_df.sum(axis=1)
results_perc = results_df.loc[:, ['Uniqueness (U)', 'Redundancy (R)', 'Synergy (S)']].div(
    results_df['Total Importance'], axis=0)

print(results_df)

print("\nContribuzione Percentuale di U, R, S (Campionamento di Contesti):")
print((results_perc * 100).round(1).fillna(0))
print("-" * 40)