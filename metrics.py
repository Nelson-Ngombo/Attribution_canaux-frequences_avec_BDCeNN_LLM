# metrics.py
import numpy as np

# --- Fonctions de coût et conflits (existantes) ---

def compute_cost(x, W):
    """
    Coût total binaire (co-canal uniquement) :
    J = somme W[i][j] si x[i] == x[j]
    """
    N = len(x)
    cost = 0
    for i in range(N):
        for j in range(i+1, N):
            if x[i] == x[j]:
                cost += W[i][j]
    return cost

def count_conflicts(x, W):
    """
    Nombre de conflits co-canal (W[i][j] > 0 et même canal)
    """
    N = len(x)
    conflicts = 0
    for i in range(N):
        for j in range(i+1, N):
            if W[i][j] > 0 and x[i] == x[j]:
                conflicts += 1
    return conflicts

def compute_metrics(x, W):
    """Retourne coût binaire, conflits co-canal, canaux utilisés."""
    cost = compute_cost(x, W)
    conflicts = count_conflicts(x, W)
    used_channels = len(set(x))
    return {"cost": cost, "conflicts": conflicts, "used_channels": used_channels}

# --- interference des canaux ---

def create_channel_interference_matrix(K, decay=0.5, cutoff=2):
    """
    Matrice d'interférence entre canaux M (K x K).
    M[k,l] = 1.0 si même canal,
            decay^distance si distance <= cutoff,
            0 sinon.
    """
    M = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            d = abs(k - l)
            if d == 0:
                M[k, l] = 1.0
            elif d <= cutoff:
                M[k, l] = decay ** d
            else:
                M[k, l] = 0.0
    return M

def compute_spectrum_energy(x, W, M):
    """
    Coût global (co-canal + canaux adjacents) :
    J = somme W[i][j] * M[x[i]][x[j]]
    """
    N = len(x)
    energy = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if W[i, j] > 0:
                energy += W[i, j] * M[x[i]][x[j]]
    return energy

def count_spectrum_conflicts(x, W, M, threshold=0.0):
    """
    Nombre de conflits spectraux : paires avec W[i][j] > 0 et M[x[i]][x[j]] > threshold.
    """
    N = len(x)
    conflicts = 0
    for i in range(N):
        for j in range(i+1, N):
            if W[i, j] > 0 and M[x[i]][x[j]] > threshold:
                conflicts += 1
    return conflicts