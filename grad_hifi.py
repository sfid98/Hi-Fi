import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. DEFINIZIONE DELLE COMPONENTI DELL'ARCHITETTURA
# =============================================================================

class GatingNetwork(nn.Module):
    """
    La rete "intelligente" che impara a generare le maschere di dropout.
    Output: Logits per la maschera di dropout.
    """

    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(num_features, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_features)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        # Restituisce i logits, che verranno usati per campionare la maschera
        return self.layer2(x)


class PredictiveModel(nn.Module):
    """
    Il modello principale che impara a fare le predizioni.
    """

    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1, bias=True)

    def forward(self, x):
        return self.linear(x)


# =============================================================================
# 2. GENERAZIONE DATI (Useremo il nostro Toy Example)
# =============================================================================

def generate_data(N=10000, random_seed=42):
    """Crea il dataset artificiale con relazioni note."""
    np.random.seed(random_seed)
    sA = np.random.randn(N, 1);
    sB = np.random.randn(N, 1);
    sC = np.random.randn(N, 1)
    Y = 2 * sA + 3 * sB + 1.5 * sC + 0.1 * np.random.randn(N, 1)
    X1 = sA + 0.2 * np.random.randn(N, 1);
    X2 = sA + 0.2 * np.random.randn(N, 1)
    noise_syn = np.random.randn(N, 1);
    X3 = sB + noise_syn;
    X4 = -noise_syn
    X5 = sC
    X = np.hstack([X1, X2, X3, X4, X5])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# =============================================================================
# --- NUOVA FUNZIONE AUSILIARIA PER L'ANALISI ---
# =============================================================================
def calculate_gradient_magnitude(model, loss_fn, X_data, y_data, driver_idx, context_indices):
    """
    Calcola la magnitudine media del gradiente per un driver in un dato contesto.
    """
    # Crea la maschera per il dropout controllato
    mask_indices = [driver_idx] + context_indices
    mask = torch.zeros(X_data.shape[1])
    mask[mask_indices] = 1.0

    X_masked = X_data * mask
    X_masked.requires_grad_(True)

    # Calcolo del gradiente
    predictions = model(X_masked)
    loss = loss_fn(predictions, y_data)
    gradients = torch.autograd.grad(loss, X_masked)[0]

    # Restituisce la magnitudine media del gradiente del solo driver
    grad_magnitude = gradients[:, driver_idx].abs().mean().item()
    return grad_magnitude


# =============================================================================
# --- NUOVA FUNZIONE PER LA SCOMPOSIZIONE FINALE ---
# =============================================================================
def run_hifi_analysis(model, loss_fn, X_data, y_data, contribution_matrix, feature_names,mode=None):
    """
    Esegue la scomposizione U, R, S usando la matrice dei contributi appresa.
    """
    print("\n--- Avvio dell'analisi di scomposizione Hi-Fi ---")
    num_features = X_data.shape[1]
    results = {}

    for i in range(num_features):
        driver_idx = i
        driver_name = feature_names[i]

        # 1. Calcola L0 (importanza del driver da solo)
        L0 = calculate_gradient_magnitude(model, loss_fn, X_data, y_data, driver_idx, context_indices=[])

        #L0 = contribution_matrix[:,i].mean().item()

        # 2. Identifica i partner dalla matrice C
        contrib_column = contribution_matrix[:, driver_idx]

        # Escludi il driver stesso dalla ricerca dei partner
        available_partners = [j for j in range(num_features) if j != driver_idx]
        if not available_partners:
            L_max, L_min = L0, L0
        else:
            partner_contribs = contrib_column[available_partners]
            j_max_local_idx = torch.argmax(partner_contribs)
            j_min_local_idx = torch.argmin(partner_contribs)

            j_max = available_partners[j_max_local_idx]
            j_min = available_partners[j_min_local_idx]

            # 3. Calcola L_max e L_min
            L_max = calculate_gradient_magnitude(model, loss_fn, X_data, y_data, driver_idx, context_indices=[j_max])
            L_min = calculate_gradient_magnitude(model, loss_fn, X_data, y_data, driver_idx, context_indices=[j_min])

        # 4. Calcola U, R, S
        U = L_min
        R = L0 - L_min
        S = L_max - L0

        if mode == "clipping":
            U, R, S = max(U, 0), max(R, 0), max(S, 0)
        elif mode == "abs":
            U, R, S = abs(U), abs(R), abs(S)

        results[driver_name] = {'U': U, 'R': R, 'S': S}
        print(f"Driver '{driver_name}': U={U:.4f}, R={R:.4f}, S={S:.4f}")

    return results

# =============================================================================
# 3. IL CICLO DI ADDESTRAMENTO
# =============================================================================

# --- Parametri ---
NUM_FEATURES = 5
LEARNING_RATE_MAIN = 0.001
LEARNING_RATE_GATING = 0.0001
LEARNING_RATE_CONTRIB = 0.1  # Alpha per la media mobile
EPOCHS = 50
BATCH_SIZE = 128

# --- Inizializzazione ---
main_model = PredictiveModel(NUM_FEATURES)
gating_network = GatingNetwork(NUM_FEATURES)

# La matrice dei contributi non è un layer, ma un parametro che aggiorniamo manualmente
contribution_matrix = torch.zeros(NUM_FEATURES, NUM_FEATURES, requires_grad=False)
contribution_count_matrix = torch.zeros(NUM_FEATURES, NUM_FEATURES, requires_grad=False)


optimizer_main = optim.Adam(main_model.parameters(), lr=LEARNING_RATE_MAIN)
optimizer_gating = optim.Adam(gating_network.parameters(), lr=LEARNING_RATE_GATING)

loss_fn_pred = nn.MSELoss()

# --- Genera Dati ---
X_data, y_data = generate_data()
dataset = torch.utils.data.TensorDataset(X_data, y_data)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Loop di Training ---
for epoch in range(0):
    for X_batch, y_batch in loader:

        # --- FASE A: ADDESTRAMENTO PREDITTIVO STANDARD ---
        optimizer_main.zero_grad()
        predictions = main_model(X_batch)
        loss_pred = loss_fn_pred(predictions, y_batch)
        loss_pred.backward()
        optimizer_main.step()

        # --- FASE B: APPRENDIMENTO DELLE INTERAZIONI ---
        optimizer_gating.zero_grad()

        # 1. L'esploratore `g` propone una strategia di masking
        # Usiamo Gumbel-Softmax per rendere il campionamento della maschera differenziabile
        mask_logits = gating_network(X_batch)
        y_soft = torch.sigmoid(mask_logits)
        y_hard = torch.bernoulli(y_soft)
        mask = y_hard - y_soft.detach() + y_soft

        # 2. Applica la maschera
        X_masked = X_batch * mask

        # 3. Calcola l'importanza (gradiente) usando la maschera proposta
        X_masked.requires_grad_(True)
        predictions_masked = main_model(X_masked)
        loss_masked = loss_fn_pred(predictions_masked, y_batch)

        # Calcola i gradienti della loss rispetto all'input mascherato
        gradients = torch.autograd.grad(loss_masked, X_masked, create_graph=True)[0]

        # 4. Aggiorna la Matrice dei Contributi C (manualmente)
        # Scegliamo un driver casuale per ogni esempio nel batch
        drivers = torch.randint(0, NUM_FEATURES, (X_batch.size(0),))

        with torch.no_grad():
            for i in range(X_batch.size(0)):  # Itera su ogni esempio del batch
                driver_idx = drivers[i].item()
                grad_magnitude = gradients[i, driver_idx]

                # Trova le feature nel contesto (quelle attive oltre al driver)
                context_indices = mask[i].nonzero().flatten()

                for context_idx in context_indices:
                    if context_idx != driver_idx:
                        # Aggiorna C con una media mobile esponenziale
                        contribution_count_matrix[context_idx, driver_idx] += 1
                        contribution_matrix[context_idx, driver_idx] += grad_magnitude

                        current_val = contribution_matrix[context_idx, driver_idx]
                        contribution_matrix[context_idx, driver_idx] = (1 - LEARNING_RATE_CONTRIB) * current_val + LEARNING_RATE_CONTRIB * grad_magnitude

        # 5. Addestra l'Esploratore `g`
        # Premiamo il gating network per aver trovato interazioni forti (gradienti alti)
        loss_interp = -gradients.abs().mean()
        loss_interp.backward()
        optimizer_gating.step()

    print(
        f"Epoca {epoch + 1}/{EPOCHS}, Loss Predittiva: {loss_pred.item():.4f}, Loss Interpretabilità: {loss_interp.item():.4f}")

for epoch in range(EPOCHS):
    for X_batch, y_batch in loader:

        # --- FASE A: ADDESTRAMENTO PREDITTIVO STANDARD ---
        optimizer_main.zero_grad()
        predictions = main_model(X_batch)
        loss_pred = loss_fn_pred(predictions, y_batch)
        loss_pred.backward()
        optimizer_main.step()

        # --- FASE B: APPRENDIMENTO DELLE INTERAZIONI (Diretto e Robusto) ---
        # Per ogni esempio nel batch, eseguiamo un sondaggio
        for i in range(X_batch.size(0)):
            # Prendi un singolo campione
            x_sample = X_batch[i:i + 1]
            y_sample = y_batch[i:i + 1]

            # 1. Scegli un driver e un contesto casuali
            driver_idx = np.random.randint(0, NUM_FEATURES)

            num_context_features = np.random.randint(0, NUM_FEATURES)
            available_partners = [j for j in range(NUM_FEATURES) if j != driver_idx]
            context_indices = np.random.choice(available_partners, num_context_features, replace=False)

            # 2. Applica la maschera e calcola il gradiente del driver
            mask_indices = [driver_idx] + list(context_indices)
            mask = torch.zeros(NUM_FEATURES)
            mask[mask_indices] = 1.0

            x_masked = x_sample * mask
            x_masked.requires_grad_(True)

            pred_masked = main_model(x_masked)
            loss_masked = loss_fn_pred(pred_masked, y_sample)

            # Calcola il gradiente solo per questo esempio
            gradients = torch.autograd.grad(loss_masked, x_masked)[0]
            grad_magnitude = gradients[0, driver_idx].abs().item()

            # 3. Aggiorna le matrici di somma e conteggio
            with torch.no_grad():
                for ctx_idx in context_indices:
                    contribution_matrix[ctx_idx, driver_idx] += grad_magnitude
                    contribution_count_matrix[ctx_idx, driver_idx] += 1

    print(f"Epoca {epoch + 1}/{EPOCHS}, Loss Predittiva: {loss_pred.item():.4f}")

# =============================================================================
# 4. ANALISI DEI RISULTATI FINALI
# =============================================================================

epsilon = 1e-8
final_contribution_matrix = contribution_matrix / (contribution_count_matrix + epsilon)
# --- FINE MODIFICA 3 ---
print("\n--- Analisi della Matrice dei Contributi Finale ---")
feature_names = ['X1_Red', 'X2_Red', 'X3_Syn', 'X4_SynHelp', 'X5_Unique']
df_contrib = pd.DataFrame(final_contribution_matrix.numpy(), index=feature_names, columns=feature_names)
print(df_contrib)

# Visualizza la heatmap della matrice C
plt.figure(figsize=(8, 6))
sns.heatmap(df_contrib, annot=True, cmap="viridis", fmt=".3f")
plt.title("Matrice dei Contributi Appresa (C)")
plt.xlabel("Driver")
plt.ylabel("Contesto")
plt.show()

# --- ESEGUI LA NUOVA ANALISI DI SCOMPOSIZIONE ---
hifi_results = run_hifi_analysis(main_model, loss_fn_pred, X_data, y_data, final_contribution_matrix, feature_names)

# --- VISUALIZZA I RISULTATI FINALI (U, R, S) ---
results_df = pd.DataFrame(hifi_results).T
results_df[['U', 'R', 'S']].plot(kind='bar', stacked=True, figsize=(10, 6), color=['#1f77b4', '#d62728', '#2ca02c'])
plt.title("Scomposizione Hi-Fi (Appresa durante il Training)")
plt.ylabel("Importanza della Feature")
plt.xticks(rotation=45, ha="right")
plt.legend(["Unique", "Redundancy", "Synergy"])
plt.tight_layout()
plt.show()
