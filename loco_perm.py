
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler

# (Assumiamo che il modello 'model' sia già addestrato e i dati 'X', 'y' siano disponibili)
# (Assicurati di usare un set di validazione che il modello non ha visto durante l'addestramento
# per stime più affidabili, ma per questo esempio useremo il training set)
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

print("\n" + "-" * 40)
print("Inizio calcolo URS con Permutation Feature Importance...")

# Usiamo l'intero dataset come set di validazione per questo esempio
validation_X, validation_y = X, y

# -- PASSO 1: Calcolo della Loss di Baseline --
model.eval()
with torch.no_grad():
    y_pred = model(validation_X)
    criterion = nn.MSELoss()
    baseline_loss = criterion(y_pred, validation_y).item()

print(f"Loss di Baseline: {baseline_loss:.4f}")

# -- PASSO 2: Calcolo dell'Importanza Individuale Totale --
total_importance = {}
for i in range(n_features):
    X_permuted = validation_X.clone()
    # Permuta la colonna i
    perm_indices = torch.randperm(X_permuted.size(0))
    X_permuted[:, i] = X_permuted[perm_indices, i]

    with torch.no_grad():
        y_pred_perm = model(X_permuted)
        perm_loss = criterion(y_pred_perm, validation_y).item()

    total_importance[i] = perm_loss - baseline_loss

print("\nImportanza Individuale Totale (Aumento della Loss):")
for i, name in enumerate(feature_names):
    print(f"  {name}: {total_importance[i]:.4f}")

# -- PASSO 3: Calcolo delle Interazioni (Sinergia e Ridondanza) --
interaction_effects = {}
for i in range(n_features):
    for j in range(i + 1, n_features):
        X_permuted_pair = validation_X.clone()
        # Permuta entrambe le colonne i e j
        perm_indices_i = torch.randperm(X_permuted_pair.size(0))
        perm_indices_j = torch.randperm(X_permuted_pair.size(0))
        X_permuted_pair[:, i] = X_permuted_pair[perm_indices_i, i]
        X_permuted_pair[:, j] = X_permuted_pair[perm_indices_j, j]

        with torch.no_grad():
            y_pred_pair = model(X_permuted_pair)
            pair_loss = criterion(y_pred_pair, validation_y).item()

        pair_importance = pair_loss - baseline_loss
        interaction = pair_importance - (total_importance[i] + total_importance[j])
        interaction_effects[(feature_names[i], feature_names[j])] = interaction

print("\nAnalisi delle Interazioni tra Coppie:")
# Ordiniamo per forza dell'interazione
sorted_interactions = sorted(interaction_effects.items(), key=lambda item: abs(item[1]), reverse=True)

for (feat_i, feat_j), value in sorted_interactions:
    if value > 0.01:  # Soglia per la sinergia
        print(f"  - Sinergia Forte tra {feat_i} e {feat_j}: {value:.4f}")
    elif value < -0.01:  # Soglia per la ridondanza
        print(f"  - Ridondanza Forte tra {feat_i} e {feat_j}: {value:.4f}")