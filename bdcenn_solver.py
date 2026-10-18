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

# Tolérance du test d'égalité des énergies locales. Les produits W_ij * M_kl
# appartiennent à {0, 0.25, 0.5, 1, 2, 4} et sont exacts en binaire ; la
# tolérance absorbe uniquement d'éventuels poids réels injectés (E7, E10).
_ENERGY_TOL = 1e-9


# ============================================================================
# 1. EXÉCUTION UNITAIRE (un seul restart)
# ============================================================================

def _bdcenn_single_run(N, K, W, M=None, max_sweeps=config.MAX_SWEEPS_BD,
                       random_order=config.RANDOM_ORDER,
                       seed=None, verbose=False, initial_assignment=None,
                       **kwargs):
    """
    Exécute un restart BD-CeNN (spécification v1.0, §6 et §9).

    Règle locale de stabilité : pour chaque cellule i, l'énergie E_i(c) est
    évaluée sur tous les canaux. Le canal courant est conservé dès que
    E_i(x_i) égale le minimum ; sinon, le plus petit indice parmi les
    minimiseurs est retenu. Toute modification de x_i implique donc une baisse
    stricte de J, et un sweep sans changement est un point fixe.

    seed est la graine de restart (dérivée de la graine algorithmique) ; elle
    alimente un générateur local et ne touche pas l'état global de NumPy.
    """
   
    # --- Rétrocompatibilité : max_iter → max_sweeps ---
    if "max_iter" in kwargs:
        warnings.warn(
            "Le paramètre 'max_iter' est déprécié. Utiliser 'max_sweeps'.",
            DeprecationWarning,
            stacklevel=2,
        )
        max_sweeps = kwargs.pop("max_iter")

    # Générateur local alimenté par la graine de restart
    rng = np.random.RandomState(seed)

    if initial_assignment is None:
        # Initialisation aléatoire uniforme x^(0) ~ U(C^N)
        x = rng.randint(0, K, size=N)
    else:
        # Warm Start (E8) : départ exact depuis l'allocation fournie ; le
        # générateur ne sert plus qu'à l'ordre de visite des sweeps.
        x = np.array(initial_assignment, dtype=int, copy=True)
        if x.shape != (N,) or x.min() < 0 or x.max() >= K:
            raise ValueError("initial_assignment doit être un vecteur de N canaux dans [0, K).")
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
        order = rng.permutation(N) if random_order else np.arange(N)

        # Balayage complet (sweep) : mise à jour in-place de chaque cellule
        for i in order:
            current_channel = x[i]

            # Étape 1 : énergie locale E_i(c) sur le voisinage N(i), Eq. (10)
            local_energies = np.zeros(K)
            for c in range(K):
                local_cost = 0.0
                for j in range(N):
                    if W[i, j] > 0:
                        if M is not None:
                            local_cost += W[i, j] * M[c, x[j]]
                        elif x[j] == c:
                            local_cost += W[i, j]
                local_energies[c] = local_cost

            # Étape 2 : énergie minimale absolue
            best_local_cost = float(local_energies.min())

            # Étapes 3 et 4 : le canal courant est évalué en priorité et
            # conservé s'il atteint le minimum (aucune réallocation sans gain) ;
            # sinon, plus petit indice parmi les minimiseurs (spéc. v1.0, §6).
            if local_energies[current_channel] <= best_local_cost + _ENERGY_TOL:
                best_local_channel = current_channel
            else:
                best_local_channel = int(np.flatnonzero(
                    local_energies <= best_local_cost + _ENERGY_TOL
                )[0])

            # Mise à jour in-place, visible des cellules suivantes du sweep
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

        # Critère d'arrêt (ii) : J inchangé sur patience_sweeps sweeps
        # consécutifs, soit patience_sweeps + 1 valeurs égales dans l'historique.
        patience = config.PATIENCE_SWEEPS
        if len(history) > patience:
            recent_costs = [h[1] for h in history[-(patience + 1):]]
            if all(c == recent_costs[0] for c in recent_costs):
                if verbose:
                    print(f"  [BD-CeNN] Stagnation (patience={patience}) au sweep {sweep_idx + 1}")
                break

    elapsed = time.perf_counter() - start_time

    # Métriques de conflit détaillées (Chantier A)
    n_cci = count_conflicts_cci(best_x, W)
    n_aci = count_conflicts_aci(best_x, W, cutoff=config.CUTOFF) if M is not None else 0

    return best_x, history, elapsed, n_cci, n_aci, best_sweep_index


def restart_seed_sequence(seed, num_restarts):
    """
    Graines des restarts dérivées de la graine algorithmique par SeedSequence.
    La liste obtenue pour R est un préfixe de celle obtenue pour R' > R.
    """
    return [int(s) for s in
            np.random.SeedSequence(int(seed)).generate_state(num_restarts)]


# ============================================================================
# 2. SOLVEUR MULTISTART (API principale)
# ============================================================================

def bdcenn_allocation(N, K, W, M=None, num_restarts=config.NUM_RESTARTS,
                      max_sweeps=config.MAX_SWEEPS_BD,
                      random_order=config.RANDOM_ORDER, seed=None,
                      verbose=False, initial_assignment=None,
                      return_sweep_stats=False, **kwargs):
    """
    Solveur BD-CeNN multistart (spécification v1.0, §10).

    seed est la graine algorithmique (config.algorithm_seed). Les graines des
    R restarts en sont dérivées par SeedSequence : flux indépendants, et la
    liste obtenue pour R est un préfixe de celle obtenue pour R' > R
    (restarts imbriqués pour E9).
    """
    
    # --- Rétrocompatibilité : max_iter → max_sweeps ---
    if "max_iter" in kwargs:
        warnings.warn(
            "Le paramètre 'max_iter' est déprécié. Utiliser 'max_sweeps'.",
            DeprecationWarning,
            stacklevel=2,
        )
        max_sweeps = kwargs.pop("max_iter")

    if seed is None:
        seed = config.ALGORITHM_SEED_OFFSET

    best_x = None
    best_cost = float('inf')
    best_history = None
    best_time = 0.0
    best_conflicts_cci = 0
    best_conflicts_aci = 0
    best_sweep_index = 0
    # Travail global : sweeps exécutés par l'ensemble des restarts
    total_sweeps_all_restarts = 0

    # Graines de restart indépendantes dérivées de la graine algorithmique.
    # Remplace seed + r, qui faisait coïncider le restart r de l'instance s
    # avec le restart r - 1 de l'instance s + 1.
    for r, seed_r in enumerate(restart_seed_sequence(seed, num_restarts)):
        # L'allocation imposée (Warm Start) ne concerne que le premier restart
        x, hist, elapsed, n_cci, n_aci, sweep_idx = _bdcenn_single_run(
            N, K, W, M=M,
            max_sweeps=max_sweeps,
            random_order=random_order,
            seed=seed_r,
            verbose=verbose if r == 0 else False,
            initial_assignment=initial_assignment if r == 0 else None,
        )

        # Sweeps réellement exécutés par ce restart (history inclut l'état initial)
        total_sweeps_all_restarts += len(hist) - 1

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

    result = (best_x, best_history, best_time,
              best_conflicts_cci, best_conflicts_aci, best_sweep_index)
    # Septième élément uniquement sur demande : appels existants inchangés
    if return_sweep_stats:
        return result + (total_sweeps_all_restarts,)
    return result


def bdcenn_restart_trace(N, K, W, M=None, num_restarts=config.NUM_RESTARTS,
                         max_sweeps=config.MAX_SWEEPS_BD,
                         random_order=config.RANDOM_ORDER, seed=None):
    """
    Exécute et conserve chaque restart séparément (E9, essais imbriqués).

    Le restart r reçoit exactement la graine qu'il recevrait dans
    bdcenn_allocation ; le meilleur des R premiers éléments de la trace
    reproduit donc bdcenn_allocation(..., num_restarts=R) pour tout R.
    """
    if seed is None:
        seed = config.ALGORITHM_SEED_OFFSET

    trace = []
    for seed_r in restart_seed_sequence(seed, num_restarts):
        start = time.perf_counter()
        x, hist, _, n_cci, n_aci, sweep_idx = _bdcenn_single_run(
            N, K, W, M=M,
            max_sweeps=max_sweeps,
            random_order=random_order,
            seed=seed_r,
        )
        cost = compute_cost_cci_aci(x, W, M) if M is not None else compute_cost_cci(x, W)
        trace.append({
            "restart_seed": seed_r,
            "initial_assignment": hist[0][2],
            "assignment": x,
            "cost": cost,
            "conflicts_cci": n_cci,
            "conflicts_aci": n_aci,
            "best_sweep_index": sweep_idx,
            # Durée du restart, évaluation du coût final comprise
            "time": time.perf_counter() - start,
        })
    return trace


# ============================================================================
# 3. WRAPPER DE RÉTROCOMPATIBILITÉ (ancien format 5-tuple)
# ============================================================================

def bdcenn_allocation_legacy(N, K, W, M=None, num_restarts=config.NUM_RESTARTS,
                             max_sweeps=config.MAX_SWEEPS_BD,
                             random_order=config.RANDOM_ORDER, seed=None,
                             verbose=False, **kwargs):
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