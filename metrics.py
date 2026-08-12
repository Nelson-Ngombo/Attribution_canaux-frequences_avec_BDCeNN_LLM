# metrics.py
import numpy as np

# ============================================================
# 1. Métriques pour les conflits co-canal (même canal)
# ============================================================

def compute_cochannel_cost(x, W):
    """
    Coût total pour les conflits co-canal uniquement :
    J = somme W[i][j] si x[i] == x[j]
    """
    N = len(x)
    cost = 0
    for i in range(N):
        for j in range(i+1, N):
            if x[i] == x[j]:
                cost += W[i][j]
    return cost

def count_cochannel_conflicts(x, W):
    """
    Nombre de conflits co-canal : paires (i,j) avec W[i][j] > 0 et x[i] == x[j]
    """
    N = len(x)
    conflicts = 0
    for i in range(N):
        for j in range(i+1, N):
            if W[i][j] > 0 and x[i] == x[j]:
                conflicts += 1
    return conflicts

def compute_metrics_cochannel(x, W):
    """Retourne coût co-canal, conflits co-canal, canaux utilisés."""
    cost = compute_cochannel_cost(x, W)
    conflicts = count_cochannel_conflicts(x, W)
    used_channels = len(set(x))
    return {"cost": cost, "conflicts": conflicts, "used_channels": used_channels}

# ============================================================
# 2. Métriques pour les conflits avec interférences entre canaux adjacents (matrice M)
# ============================================================

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

def compute_adjacent_cost(x, W, M):
    """
    Coût total prenant en compte les interférences entre canaux adjacents (via M) :
    J = somme W[i][j] * M[x[i]][x[j]]
    """
    N = len(x)
    energy = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if W[i, j] > 0:
                energy += W[i, j] * M[x[i]][x[j]]
    return energy

def count_adjacent_conflicts(x, W, M, threshold=0.0):
    """
    Nombre de conflits avec interférences entre canaux adjacents :
    paires avec W[i][j] > 0 et M[x[i]][x[j]] > threshold.
    """
    N = len(x)
    conflicts = 0
    for i in range(N):
        for j in range(i+1, N):
            if W[i, j] > 0 and M[x[i]][x[j]] > threshold:
                conflicts += 1
    return conflicts

def compute_metrics_adjacent(x, W, M):
    """Retourne coût avec adjacents, conflits avec adjacents, canaux utilisés."""
    cost = compute_adjacent_cost(x, W, M)
    conflicts = count_adjacent_conflicts(x, W, M)
    used_channels = len(set(x))
    return {"cost": cost, "conflicts": conflicts, "used_channels": used_channels}

# ============================================================
# 3. Fonctions statistiques
# ============================================================

def compute_confidence_interval(data, confidence=0.95):
    """
    Calcule l'intervalle de confiance à 95% pour un tableau de données.
    Retourne (borne_inf, borne_sup)
    """
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)  # erreur standard
    # Pour 95% de confiance, on utilise z=1.96 (approximation normale)
    # Si n < 30, on pourrait utiliser la loi de Student, mais on garde 1.96 pour simplifier
    z = 1.96
    ci = z * std_err
    return mean - ci, mean + ci