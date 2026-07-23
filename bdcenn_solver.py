# bdcenn_solver.py
import numpy as np
import time
from metrics import compute_cost, count_conflicts, compute_spectrum_energy

def _bdcenn_single_run(N, K, W, M=None, max_iter=50, random_order=True, seed=None, verbose=False):
    """
    Une seule exécution du BD-CeNN (asynchrone, sans recuit).
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.randint(0, K, size=N)
    best_x = x.copy()
    
    if M is not None:
        best_cost = compute_spectrum_energy(x, W, M)
    else:
        best_cost = compute_cost(x, W)
    
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
            current_cost = compute_spectrum_energy(x, W, M)
        else:
            current_cost = compute_cost(x, W)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.copy()
        
        history.append((iteration + 1, current_cost, x.copy()))
        
        if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
            print(f"  [BD-CeNN] Itération {iteration+1}/{max_iter} : coût = {current_cost}")
        
        if current_cost == 0:
            if verbose:
                print(f"  [BD-CeNN] Convergence atteinte (coût nul) à l'itération {iteration+1}")
            break
        if len(history) > 10:
            recent_costs = [h[1] for h in history[-5:]]
            if all(c == recent_costs[0] for c in recent_costs):
                if verbose:
                    print(f"  [BD-CeNN] Stabilisation détectée à l'itération {iteration+1}")
                break
    
    elapsed = time.perf_counter() - start_time
    conflicts = count_conflicts(best_x, W)
    return best_x, history, elapsed, conflicts


def bdcenn_allocation(N, K, W, M=None, num_restarts=10, max_iter=50, random_order=True, seed=None, verbose=False):
    """
    Solveur BD-CeNN avec redémarrages multiples :
    exécute le solveur `num_restarts` fois avec des initialisations aléatoires différentes,
    et retourne la meilleure solution (coût minimal).
    """
    if seed is None:
        seed = 42
    
    best_x = None
    best_cost = float('inf')
    best_history = None
    best_time = 0.0
    best_conflicts = 0
    
    for i in range(num_restarts):
        seed_i = seed + i
        x, hist, elapsed, conf = _bdcenn_single_run(
            N, K, W, M=M,
            max_iter=max_iter,
            random_order=random_order,
            seed=seed_i,
            verbose=verbose if i == 0 else False  # verbose seulement pour le premier
        )
        # Calcul du coût final
        if M is not None:
            cost = compute_spectrum_energy(x, W, M)
        else:
            cost = compute_cost(x, W)
        if cost < best_cost:
            best_cost = cost
            best_x = x
            best_history = hist
            best_time = elapsed
            best_conflicts = conf
    
    return best_x, best_history, best_time, best_conflicts