# metrics.py
import numpy as np

def compute_cost(x, W):
    """
    Coût total J(x) = somme sur les paires i<j de W[i][j] si x[i] == x[j].
    (pénalité proportionnelle au niveau d'interférence)
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
    Nombre de paires en conflit (W[i][j] > 0 et x[i] == x[j]).
    (conflit co‑canal uniquement)
    """
    N = len(x)
    conflicts = 0
    for i in range(N):
        for j in range(i+1, N):
            if W[i][j] > 0 and x[i] == x[j]:
                conflicts += 1
    return conflicts

def count_spectrum_conflicts(x, W, M, threshold=0.0):
    """
    Nombre de paires en conflit en tenant compte de l'interférence entre canaux.
    Une paire (i,j) est en conflit si W[i][j] > 0 et M[x[i]][x[j]] > threshold.
    Par défaut, threshold=0.0 : toute interférence non nulle est comptée.
    """
    N = len(x)
    conflicts = 0
    for i in range(N):
        for j in range(i+1, N):
            if W[i][j] > 0 and M[x[i]][x[j]] > threshold:
                conflicts += 1
    return conflicts

def compute_metrics(x, W):
    """Retourne un dictionnaire avec coût, conflits, et taux d'utilisation des canaux."""
    cost = compute_cost(x, W)
    conflicts = count_conflicts(x, W)
    used_channels = len(set(x))
    return {
        "cost": cost,
        "conflicts": conflicts,
        "used_channels": used_channels
    }