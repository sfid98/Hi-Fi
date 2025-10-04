import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler # <-- IMPORT AGGIUNTO

# -- PASSO 0: Funzione per la generazione dei dati (invariata) --

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

# --- MODIFICA: Verifichiamo la performance del modello ---
model.eval()
with torch.no_grad():
    y_pred = model(X)
    final_mse = criterion(y_pred, y)
print(f"Mean Squared Error finale sul training set: {final_mse.item():.4f}")
print("-" * 40)

# Il resto dello script per il calcolo di URS rimane invariato...
# (Passi 3 e 4 sono identici a prima)
def get_global_urs_scores(model, data_loader):
    model.eval()
    n_features = model[0].in_features
    total_uniqueness = torch.zeros(n_features)
    total_redundancy = torch.zeros(n_features)
    total_synergy = torch.zeros(n_features)
    avg_hessian = torch.zeros((n_features, n_features))
    num_samples = 0
    def model_output_fn_single(x): return model(x.unsqueeze(0)).squeeze()
    print("Calcolo dei punteggi URS sul set di riferimento...")
    for inputs, _ in tqdm(data_loader):
        for i in range(inputs.shape[0]):
            input_sample = inputs[i]
            H = torch.autograd.functional.hessian(model_output_fn_single, input_sample, create_graph=False)
            avg_hessian += H
            diag = torch.abs(torch.diag(H)); H_off_diag = H - torch.diag(torch.diag(H))
            synergy_matrix = torch.clamp(H_off_diag, min=0)
            redundancy_matrix = torch.abs(torch.clamp(H_off_diag, max=0))
            total_uniqueness += diag; total_synergy += torch.sum(synergy_matrix, dim=1)
            total_redundancy += torch.sum(redundancy_matrix, dim=1)
            num_samples += 1
    global_uniqueness = total_uniqueness / num_samples
    global_redundancy = total_redundancy / num_samples
    global_synergy = total_synergy / num_samples
    global_hessian = avg_hessian / num_samples
    return global_uniqueness, global_redundancy, global_synergy, global_hessian

ref_dataset = TensorDataset(X[:500], y[:500]); ref_loader = DataLoader(ref_dataset, batch_size=50)
u, r, s, H_global = get_global_urs_scores(model, ref_loader)

results_df = pd.DataFrame({'Uniqueness (U)': u.numpy(), 'Redundancy (R)': r.numpy(), 'Synergy (S)': s.numpy()}, index=feature_names)
results_df['Total Importance'] = results_df.sum(axis=1)
# Calcoliamo il contributo percentuale di U, R, S per ogni feature
results_perc = results_df.loc[:, ['Uniqueness (U)', 'Redundancy (R)', 'Synergy (S)']].div(results_df['Total Importance'], axis=0)

print("\nContribuzione Percentuale di U, R, S per Feature:")
print((results_perc * 100).round(1))
print("-" * 40)

def find_subsets_global(feature_name, global_hessian, threshold_ratio=0.1):
    feature_idx = feature_names.index(feature_name)
    synergistic_partners, redundant_partners = [], []
    threshold = global_hessian.abs().mean() * threshold_ratio
    for j in range(len(feature_names)):
        if feature_idx == j: continue
        interaction_strength = global_hessian[feature_idx, j].item()
        if interaction_strength > threshold: synergistic_partners.append((feature_names[j], round(interaction_strength, 4)))
        elif interaction_strength < -threshold: redundant_partners.append((feature_names[j], round(interaction_strength, 4)))
    synergistic_partners.sort(key=lambda x: x[1], reverse=True)
    redundant_partners.sort(key=lambda x: abs(x[1]), reverse=True)
    return synergistic_partners, redundant_partners

print("Analisi dei Sottoinsiemi (basata sull'Hessiana Globale):")
for name in feature_names:
    synergistic, redundant = find_subsets_global(name, H_global)
    print(f"Feature: {name}"); print(f"  - Partner Sinergici: {synergistic if synergistic else 'Nessuno'}"); print(f"  - Partner Ridondanti: {redundant if redundant else 'Nessuno'}\n")