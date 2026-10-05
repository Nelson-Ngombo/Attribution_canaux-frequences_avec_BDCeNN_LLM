# bdcenn_solver.py
"""
Binary Discretized Cellular Neural Network (BD-CeNN) Optimizer.
Implements local asynchronous energy updates with multi-restart capability
for escape from local minima.
"""

import time
import warnings
import numpy as np

import config
from metrics import (
    compute_cost_cci,
    compute_cost_cci_aci,
    count_conflicts_cci,
    count_conflicts_aci,
)


# ============================================================================
# 1. EXÉCUTION UNITAIRE (un seul restart)
# ============================================================================

def _bdcenn_single_run(N, K, W, M=None, max_sweeps=50, random_order=True,
                       seed=None, verbose=False, **kwargs):
   
    # --- Rétrocompatibilité : max_iter → max_sweeps ---
    if "max_iter" in kwargs:
        warnings.warn(
            "Le paramètre 'max_iter' est déprécié. Utiliser 'max_sweeps'.",
            DeprecationWarning,
            stacklevel=2,
        )
        max_sweeps = kwargs.pop("max_iter")

    if seed is not None:
        np.random.seed(seed)

    # Initialisation aléatoire uniforme
    x = np.random.randint(0, K, size=N)
    best_x = x.copy()

    # Coût initial
    if M is not None:
        best_cost = compute_cost_cci_aci(x, W, M)
    else:
        best_cost = compute_cost_cci(x, W)

    best_sweep_index = 0
    history = [(0, best_cost, x.copy())]

    if verbose:
        print(f"  [BD-CeNN] Init : cout = {best_cost}")

    start_time = time.perf_counter()

    for sweep_idx in range(max_sweeps):
        # Ordre de parcours : aléatoire ou séquentiel
        order = np.random.permutation(N) if random_order else np.arange(N)

        # Balayage complet (sweep) : mise à jour in-place de chaque cellule
        for i in order:
            current_channel = x[i]
            best_local_cost = float('inf')
            best_local_channel = current_channel

            for c in range(K):
                local_cost = 0.0
                for j in range(N):
                    if W[i, j] > 0:
                        if M is not None:
                            local_cost += W[i, j] * M[c, x[j]]
                        else:
                            if x[j] == c:
                                local_cost += W[i, j]
                if local_cost < best_local_cost:
                    best_local_cost = local_cost
                    best_local_channel = c

            # Mise à jour in-place (règle Eq. 7)
            if best_local_channel != current_channel:
                x[i] = best_local_channel

        # Évaluation du coût global après le sweep complet
        if M is not None:
            current_cost = compute_cost_cci_aci(x, W, M)
        else:
            current_cost = compute_cost_cci(x, W)

        # Mise à jour du meilleur coût global
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.copy()
            best_sweep_index = sweep_idx + 1

        history.append((sweep_idx + 1, current_cost, x.copy()))

        if verbose and (sweep_idx % 10 == 0 or sweep_idx == max_sweeps - 1):
            print(f"  [BD-CeNN] Sweep {sweep_idx + 1}/{max_sweeps} : cout = {current_cost}")

        # Critère d'arrêt (i) : coût nul atteint
        if current_cost == 0:
            if verbose:
                print(f"  [BD-CeNN] Convergence (cout nul) au sweep {sweep_idx + 1}")
            break

        # Critère d'arrêt (ii) : stagnation sur patience_sweeps sweeps
        patience = config.PATIENCE_SWEEPS
        if len(history) > patience:
            recent_costs = [h[1] for h in history[-patience:]]
            if all(c == recent_costs[0] for c in recent_costs):
                if verbose:
                    print(f"  [BD-CeNN] Stagnation (patience={patience}) au sweep {sweep_idx + 1}")
                break

    elapsed = time.perf_counter() - start_time

    # Métriques de conflit détaillées (Chantier A)
    n_cci = count_conflicts_cci(best_x, W)
    n_aci = count_conflicts_aci(best_x, W, cutoff=2) if M is not None else 0

    return best_x, history, elapsed, n_cci, n_aci, best_sweep_index


# ============================================================================
# 2. SOLVEUR MULTISTART (API principale)
# ============================================================================

def bdcenn_allocation(N, K, W, M=None, num_restarts=10, max_sweeps=50,
                      random_order=True, seed=None, verbose=False, **kwargs):
    
    # --- Rétrocompatibilité : max_iter → max_sweeps ---
    if "max_iter" in kwargs:
        warnings.warn(
            "Le paramètre 'max_iter' est déprécié. Utiliser 'max_sweeps'.",
            DeprecationWarning,
            stacklevel=2,
        )
        max_sweeps = kwargs.pop("max_iter")

    if seed is None:
        seed = 42

    best_x = None
    best_cost = float('inf')
    best_history = None
    best_time = 0.0
    best_conflicts_cci = 0
    best_conflicts_aci = 0
    best_sweep_index = 0

    for r in range(num_restarts):
        seed_r = seed + r
        x, hist, elapsed, n_cci, n_aci, sweep_idx = _bdcenn_single_run(
            N, K, W, M=M,
            max_sweeps=max_sweeps,
            random_order=random_order,
            seed=seed_r,
            verbose=verbose if r == 0 else False,
        )

        # Évaluation du coût final pour comparaison entre restarts
        if M is not None:
            cost = compute_cost_cci_aci(x, W, M)
        else:
            cost = compute_cost_cci(x, W)

        if cost < best_cost:
            best_cost = cost
            best_x = x
            best_history = hist
            best_time = elapsed
            best_conflicts_cci = n_cci
            best_conflicts_aci = n_aci
            best_sweep_index = sweep_idx

    return (best_x, best_history, best_time,
            best_conflicts_cci, best_conflicts_aci, best_sweep_index)


# ============================================================================
# 3. WRAPPER DE RÉTROCOMPATIBILITÉ (ancien format 5-tuple)
# ============================================================================

def bdcenn_allocation_legacy(N, K, W, M=None, num_restarts=10, max_sweeps=50,
                             random_order=True, seed=None, verbose=False,
                             **kwargs):
    """
    Wrapper de rétrocompatibilité retournant l'ancien format 5-tuple.

    Déprécié. Utiliser bdcenn_allocation() qui retourne le 6-tuple complet
    avec les conflits CCI et ACI séparés.

    Returns
    -------
    best_x : ndarray
    best_history : list
    best_time : float
    best_conflicts_total : int
        Somme C_CCI + C_ACI (équivalent de l'ancien 'conflicts').
    best_sweep_index : int
    """
    warnings.warn(
        "bdcenn_allocation_legacy est déprécié. "
        "Utiliser bdcenn_allocation() et adapter l'unpacking.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Accepter l'ancien paramètre max_iter
    if "max_iter" in kwargs:
        max_sweeps = kwargs.pop("max_iter")

    result = bdcenn_allocation(
        N, K, W, M=M,
        num_restarts=num_restarts,
        max_sweeps=max_sweeps,
        random_order=random_order,
        seed=seed,
        verbose=verbose,
    )

    best_x, best_history, best_time, n_cci, n_aci, best_sweep = result
    return best_x, best_history, best_time, n_cci + n_aci, best_sweep