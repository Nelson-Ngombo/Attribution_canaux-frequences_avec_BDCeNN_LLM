# metrics.py
"""
Independent validation metrics module for channel assignment optimization.
Computes co-channel costs, adjacent-channel interference penalties, conflict counts,
and formal statistical confidence intervals.
"""

import warnings
import numpy as np


# ============================================================================
# 1. MATRICE SPECTRALE M (interférence inter-canaux)
# ============================================================================

def create_channel_interference_matrix(K, decay=0.5, cutoff=2):
    """
    Construit la matrice spectrale M de taille K x K.

    M[k, l] = 1           si k = l
    M[k, l] = decay^|k-l| si 1 <= |k-l| <= cutoff
    M[k, l] = 0           si |k-l| > cutoff

    """
    M = np.zeros((K, K), dtype=float)
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


# ============================================================================
# 2. FONCTIONS DE COÛT GLOBAL J(x)
# ============================================================================

def compute_cost_cci(x, W):
    """
    Calcule le coût global sous le régime CCI-only (Eq. 2).

    J_CCI(x) = sum_{i<j} W_ij * 1_{x_i = x_j}

    Paramètres
    ----------
    x : array-like, shape (N,)
        Vecteur d'allocation (canaux attribués).
    W : ndarray, shape (N, N)
        Matrice de poids géographiques.

    Returns
    -------
    cost : float
        Coût global pondéré.
    """
    N = len(x)
    cost = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] > 0 and x[i] == x[j]:
                cost += W[i, j]
    return float(cost)


def compute_cost_cci_aci(x, W, M):
    """
    Calcule le coût global sous le régime CCI+ACI (Eq. 3).

    J_CCI+ACI(x) = sum_{i<j} W_ij * M_{x_i, x_j}

    Paramètres
    ----------
    x : array-like, shape (N,)
        Vecteur d'allocation.
    W : ndarray, shape (N, N)
        Matrice de poids géographiques.
    M : ndarray, shape (K, K)
        Matrice spectrale d'interférence inter-canaux.

    Returns
    -------
    cost : float
        Coût global pondéré incluant les fuites adjacentes.
    """
    N = len(x)
    cost = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] > 0:
                cost += W[i, j] * M[x[i], x[j]]
    return float(cost)


# ============================================================================
# 3. COMPTEURS DE CONFLITS (Eq. 4, 5, 6)
# ============================================================================

def count_conflicts_cci(x, W):
    """
    Compte les conflits co-canal stricts (Eq. 4).

    C_CCI(x) = #{(i,j) : i < j, W_ij > 0 et x_i = x_j}

    Paramètres
    ----------
    x : array-like, shape (N,)
        Vecteur d'allocation.
    W : ndarray, shape (N, N)
        Matrice de poids géographiques.

    Returns
    -------
    n_cci : int
        Nombre de paires en conflit co-canal.
    """
    N = len(x)
    n_cci = 0
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] > 0 and x[i] == x[j]:
                n_cci += 1
    return n_cci


def count_conflicts_aci(x, W, cutoff=2):
    """
    Compte les conflits de canal adjacent stricts (Eq. 5).

    C_ACI(x) = #{(i,j) : i < j, W_ij > 0 et 0 < |x_i - x_j| <= cutoff}

    Ce compteur est mutuellement exclusif avec C_CCI : une paire dont
    les canaux sont identiques (|x_i - x_j| = 0) n'est PAS comptée ici.

    Paramètres
    ----------
    x : array-like, shape (N,)
        Vecteur d'allocation.
    W : ndarray, shape (N, N)
        Matrice de poids géographiques.
    cutoff : int, optionnel
        Distance spectrale maximale de couplage (défaut 2).

    Returns
    -------
    n_aci : int
        Nombre de paires en conflit de canal adjacent.
    """
    N = len(x)
    n_aci = 0
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] > 0:
                diff = abs(int(x[i]) - int(x[j]))
                if 0 < diff <= cutoff:
                    n_aci += 1
    return n_aci


def count_conflicts_total(x, W, cutoff=2):
    """
    Compte le nombre total de conflits (Eq. 6).

    C_total(x) = C_CCI(x) + C_ACI(x)

    Paramètres
    ----------
    x : array-like, shape (N,)
        Vecteur d'allocation.
    W : ndarray, shape (N, N)
        Matrice de poids géographiques.
    cutoff : int, optionnel
        Distance spectrale maximale de couplage (défaut 2).

    Returns
    -------
    n_total : int
        Nombre total de paires en conflit (co-canal + adjacent).
    """
    return count_conflicts_cci(x, W) + count_conflicts_aci(x, W, cutoff)


# ============================================================================
# 4. FONCTIONS AGRÉGÉES (dictionnaires de métriques)
# ============================================================================

def compute_metrics_cci(x, W):
    """
    Calcule l'ensemble des métriques sous le régime CCI-only.

    Paramètres
    ----------
    x : array-like, shape (N,)
        Vecteur d'allocation.
    W : ndarray, shape (N, N)
        Matrice de poids géographiques.

    Returns
    -------
    metrics : dict
        Dictionnaire contenant :
        - "cost" : float, coût J_CCI(x)
        - "n_conflicts_cci" : int, nombre de conflits co-canal
        - "n_conflicts_aci" : int, toujours 0 en régime CCI-only
        - "n_conflicts_total" : int, égal à n_conflicts_cci
        - "used_channels" : int, nombre de canaux distincts utilisés
    """
    n_cci = count_conflicts_cci(x, W)
    return {
        "cost": compute_cost_cci(x, W),
        "n_conflicts_cci": n_cci,
        "n_conflicts_aci": 0,
        "n_conflicts_total": n_cci,
        "used_channels": len(set(x)),
    }


def compute_metrics_cci_aci(x, W, M, cutoff=2):
    """
    Calcule l'ensemble des métriques sous le régime CCI+ACI.

    Paramètres
    ----------
    x : array-like, shape (N,)
        Vecteur d'allocation.
    W : ndarray, shape (N, N)
        Matrice de poids géographiques.
    M : ndarray, shape (K, K)
        Matrice spectrale d'interférence inter-canaux.
    cutoff : int, optionnel
        Distance spectrale maximale de couplage (défaut 2).

    Returns
    -------
    metrics : dict
        Dictionnaire contenant :
        - "cost" : float, coût J_CCI+ACI(x)
        - "n_conflicts_cci" : int, nombre de conflits co-canal
        - "n_conflicts_aci" : int, nombre de conflits de canal adjacent
        - "n_conflicts_total" : int, somme des deux
        - "used_channels" : int, nombre de canaux distincts utilisés
    """
    n_cci = count_conflicts_cci(x, W)
    n_aci = count_conflicts_aci(x, W, cutoff)
    return {
        "cost": compute_cost_cci_aci(x, W, M),
        "n_conflicts_cci": n_cci,
        "n_conflicts_aci": n_aci,
        "n_conflicts_total": n_cci + n_aci,
        "used_channels": len(set(x)),
    }


# ============================================================================
# 5. FONCTIONS STATISTIQUES
# ============================================================================

def compute_confidence_interval(data, confidence=0.95):
    """
    Calcule l'intervalle de confiance paramétrique pour un échantillon.

    Paramètres
    ----------
    data : array-like
        Échantillon de mesures.
    confidence : float, optionnel
        Niveau de confiance (défaut 0.95).

    Returns
    -------
    lower : float
        Borne inférieure de l'intervalle.
    upper : float
        Borne supérieure de l'intervalle.
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    if n < 2:
        mean_val = float(np.mean(arr)) if n == 1 else 0.0
        return mean_val, mean_val

    mean_val = np.mean(arr)
    std_err = np.std(arr, ddof=1) / np.sqrt(n)
    z_score = 1.96 if confidence == 0.95 else 2.576
    margin = z_score * std_err
    return float(mean_val - margin), float(mean_val + margin)


# ============================================================================
# 6. ALIAS DE RÉTROCOMPATIBILITÉ (dépréciés, Chantier A)
# ============================================================================
# Ces wrappers permettent aux modules existants (experiments.py,
# bdcenn_solver.py, etc.) de continuer à fonctionner pendant la
# période de transition. Ils émettent un DeprecationWarning discret.

def compute_cochannel_cost(x, W):
    """
    Déprécié. Utiliser compute_cost_cci(x, W).
    """
    return compute_cost_cci(x, W)


def compute_adjacent_cost(x, W, M):
    """
    Déprécié. Utiliser compute_cost_cci_aci(x, W, M).
    """
    return compute_cost_cci_aci(x, W, M)


def count_cochannel_conflicts(x, W):
    """
    Déprécié. Utiliser count_conflicts_cci(x, W).
    """
    return count_conflicts_cci(x, W)


def count_adjacent_conflicts(x, W, M=None, threshold=0.0):
    """
    Déprécié. Utiliser count_conflicts_cci, count_conflicts_aci
    ou count_conflicts_total selon le besoin.

    Cet alias retourne C_total (C_CCI + C_ACI) pour préserver la
    sémantique de l'ancien comportement (toute paire avec M[x_i,x_j] > 0
    inclut le co-canal puisque M[k,k] = 1).

    Le paramètre M est ignoré dans le nouveau calcul mais conservé
    dans la signature pour la compatibilité des appels existants.
    """
    warnings.warn(
        "count_adjacent_conflicts est déprécié. Utilisez count_conflicts_cci, "
        "count_conflicts_aci ou count_conflicts_total selon le besoin.",
        DeprecationWarning,
        stacklevel=2,
    )
    return count_conflicts_total(x, W, cutoff=2)


def compute_metrics_cochannel(x, W):
    """
    Déprécié. Utiliser compute_metrics_cci(x, W).
    """
    warnings.warn(
        "compute_metrics_cochannel est déprécié. Utilisez compute_metrics_cci.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_metrics_cci(x, W)


def compute_metrics_adjacent(x, W, M):
    """
    Déprécié. Utiliser compute_metrics_cci_aci(x, W, M).
    """
    warnings.warn(
        "compute_metrics_adjacent est déprécié. Utilisez compute_metrics_cci_aci.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_metrics_cci_aci(x, W, M)