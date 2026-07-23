# bdcenn_solver.py
import numpy as np
import time
from metrics import compute_cost, count_conflicts
from bdcenn_spectrum import compute_spectrum_energy

def bdcenn_allocation(N, K, W, M=None, max_iter=200, random_order=True, seed=None, verbose=False,
                      use_sa=False, T_init=100.0, T_min=0.1, cooling_rate=0.99):
    """
    Solveur BD-CeNN pour l'attribution de canaux (version ASYNCHRONE).
    - N : nombre de cellules
    - K : nombre de canaux (0 à K-1)
    - W : matrice d'interférence (N x N)
    - M : matrice d'interférence entre canaux (K x K),
    - max_iter : nombre maximal d'itérations
    - random_order : si True, on mélange l'ordre des cellules à chaque itération
    - seed : pour la reproductibilité de l'initialisation aléatoire
    - verbose : si True, affiche la progression dans le terminal
    - use_sa : si True, active le recuit simulé (acceptation des augmentations temporaires)
    - T_init : température initiale
    - T_min : température minimale (arrêt si T < T_min)
    - cooling_rate : facteur de refroidissement (ex: 0.99)
    Retourne : (x_final, history, time_elapsed, conflicts)
        - history : liste de tuples (itération, coût, allocation_complète)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialisation aléatoire
    x = np.random.randint(0, K, size=N)
    best_x = x.copy()
    
    # Calcul du coût initial (selon le mode)
    if M is not None:
        best_cost = compute_spectrum_energy(x, W, M)
    else:
        best_cost = compute_cost(x, W)
    
    history = [(0, best_cost, x.copy())]
    
    if verbose:
        print(f"  [BD-CeNN] Init : coût = {best_cost}")
    
    start_time = time.perf_counter()
    
    T = T_init  # température courante
    
    for iteration in range(max_iter):
        order = np.random.permutation(N) if random_order else np.arange(N)
        x_old = x.copy()  # pour le recuit (comparaison des coûts)
        
        for i in order:
            current_channel = x[i]
            best_local_cost = float('inf')
            best_local_channel = current_channel
            
            # Calcul du meilleur canal local
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
            
            # Décision d'acceptation
            if best_local_channel != current_channel:
                if use_sa:
                    # Calcul du coût local actuel pour le canal courant
                    current_local_cost = 0
                    for j in range(N):
                        if W[i][j] > 0:
                            if M is not None:
                                current_local_cost += W[i][j] * M[current_channel][x[j]]
                            else:
                                if x[j] == current_channel:
                                    current_local_cost += W[i][j]
                    delta = best_local_cost - current_local_cost  # peut être négatif ou positif
                    
                    if delta < 0:
                        # Amélioration : acceptée
                        x[i] = best_local_channel
                    else:
                        # Détérioration : acceptée avec probabilité exp(-delta/T)
                        proba = np.exp(-delta / T) if T > 0 else 0.0
                        if np.random.random() < proba:
                            x[i] = best_local_channel
                else:
                    # Comportement classique (acceptation uniquement si amélioration)
                    x[i] = best_local_channel
        
        # Calcul du coût global après cette itération
        if M is not None:
            current_cost = compute_spectrum_energy(x, W, M)
        else:
            current_cost = compute_cost(x, W)
        
        # Mise à jour de la meilleure solution
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.copy()
        
        history.append((iteration + 1, current_cost, x.copy()))
        
        if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
            if use_sa:
                print(f"  [BD-CeNN] Itération {iteration+1}/{max_iter} : coût = {current_cost}, T = {T:.4f}")
            else:
                print(f"  [BD-CeNN] Itération {iteration+1}/{max_iter} : coût = {current_cost}")
        
        # Refroidissement (si recuit activé)
        if use_sa:
            T *= cooling_rate
            if T < T_min:
                if verbose:
                    print(f"  [BD-CeNN] Température minimale atteinte, arrêt.")
                break
        
        # Conditions d'arrêt classiques
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