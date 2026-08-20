# bdcenn_solver.py
import numpy as np
import time
from metrics import compute_cochannel_cost, count_cochannel_conflicts, compute_adjacent_cost
import config

def _bdcenn_single_run(N, K, W, M=None, max_iter=50, random_order=True, seed=None, verbose=False):
    """
    Une seule exécution du BD-CeNN (asynchrone, sans recuit).
    Retourne : (x, history, elapsed, conflicts, best_iteration)
    où best_iteration est l'itération à laquelle le meilleur coût a été atteint pour la première fois.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.randint(0, K, size=N)
    best_x = x.copy()
    
    if M is not None:
        best_cost = compute_adjacent_cost(x, W, M)
    else:
        best_cost = compute_cochannel_cost(x, W)
    
    # L'itération 0 est l'initialisation, c'est le point de départ
    best_iteration = 0  # Itération du meilleur coût trouvé
    
    history = [(0, best_cost, x.copy())]
    if verbose:
        print(f"  [BD-CeNN] Init : coût = {best_cost}")
    
    start_time = time.perf_counter()
    
    for iteration in range(max_iter):
        order = np.random.permutation(N) if random_order else np.arange(N)
        
        for i in order:
            current_channel = x[i]
            best_local_cost = float('inf')
            best_local_channel = current_channel
            
            for c in range(K):
                local_cost = 0
                for j in range(N):
                    if W[i][j] > 0:
                        if M is not None:
                            local_cost += W[i][j] * M[c][x[j]]
                        else:
                            if x[j] == c:
                                local_cost += W[i][j]
                if local_cost < best_local_cost:
                    best_local_cost = local_cost
                    best_local_channel = c
            
            if best_local_channel != current_channel:
                x[i] = best_local_channel
        
        if M is not None:
            current_cost = compute_adjacent_cost(x, W, M)
        else:
            current_cost = compute_cochannel_cost(x, W)
        
        # Si le coût courant est meilleur que le meilleur connu, on le met à jour
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.copy()
            best_iteration = iteration + 1  # L'itération courante (1-indexée)
        
        history.append((iteration + 1, current_cost, x.copy()))
        
        if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
            print(f"  [BD-CeNN] Itération {iteration+1}/{max_iter} : coût = {current_cost}")
        
        # Arrêt anticipé si coût nul
        if current_cost == 0:
            if verbose:
                print(f"  [BD-CeNN] Convergence atteinte (coût nul) à l'itération {iteration+1}")
            break
        
        # Vérification de la stabilisation avec la patience définie dans config
        patience = config.PATIENCE
        if len(history) > patience:
            recent_costs = [h[1] for h in history[-patience:]]
            if all(c == recent_costs[0] for c in recent_costs):
                if verbose:
                    print(f"  [BD-CeNN] Stabilisation détectée (patience={patience}) à l'itération {iteration+1}")
                break
    
    elapsed = time.perf_counter() - start_time
    conflicts = count_cochannel_conflicts(best_x, W)
    return best_x, history, elapsed, conflicts, best_iteration


def bdcenn_allocation(N, K, W, M=None, num_restarts=10, max_iter=50, random_order=True, seed=None, verbose=False):
    """
    Solveur BD-CeNN avec redémarrages multiples :
    exécute le solveur `num_restarts` fois avec des initialisations aléatoires différentes,
    et retourne la meilleure solution (coût minimal).
    Retourne : (best_x, best_history, best_time, best_conflicts, best_iteration)
    où best_iteration est l'itération du meilleur coût parmi les redémarrages.
    """
    if seed is None:
        seed = 42
    
    best_x = None
    best_cost = float('inf')
    best_history = None
    best_time = 0.0
    best_conflicts = 0
    best_iteration = 0  # Itération du meilleur coût global
    
    for i in range(num_restarts):
        seed_i = seed + i
        x, hist, elapsed, conf, iter_best = _bdcenn_single_run(
            N, K, W, M=M,
            max_iter=max_iter,
            random_order=random_order,
            seed=seed_i,
            verbose=verbose if i == 0 else False
        )
        # Calcul du coût final
        if M is not None:
            cost = compute_adjacent_cost(x, W, M)
        else:
            cost = compute_cochannel_cost(x, W)
        
        if cost < best_cost:
            best_cost = cost
            best_x = x
            best_history = hist
            best_time = elapsed
            best_conflicts = conf
            best_iteration = iter_best  # On garde l'itération de la meilleure solution
    
    return best_x, best_history, best_time, best_conflicts, best_iteration