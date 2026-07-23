# bdcenn_spectrum.py
import numpy as np
import time
from metrics import count_conflicts

def create_channel_interference_matrix(K, decay=0.5, cutoff=2):
    """
    Matrice d'interférence entre canaux M (K x K).
    M[k,l] = 1.0 si même canal, decay^distance si distance <= cutoff, 0 sinon.
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

def activation_argmax(X):
    """Transforme les états internes X (N x K) en sorties binaires Y (N x K)."""
    N, K = X.shape
    Y = np.zeros((N, K), dtype=int)
    best = np.argmax(X, axis=1)
    for i in range(N):
        Y[i, best[i]] = 1
    return Y

def y_to_channels(Y):
    """Convertit une matrice one-hot Y en vecteur de canaux (0..K-1)."""
    return np.argmax(Y, axis=1)

def compute_spectrum_energy(x, W, M):
    """Coût global J = somme_{i<j} W[i,j] * M[x[i], x[j]]."""
    N = len(x)
    energy = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if W[i, j] > 0:
                energy += W[i, j] * M[x[i], x[j]]
    return energy

def bdcenn_spectrum_allocation(N, K, W, M=None, alpha=0.2, beta=1.0, gamma=0.0,
                               dt=0.1, max_iter=100, seed=None, verbose=False,
                               x_init=None):
    """
    Solveur BD-CeNN continu avec matrice d'interférence entre canaux.
    Si x_init est fourni, on force l'initialisation pour que la sortie Y
    corresponde exactement à x_init.
    Retourne : (best_x, history_spectrum, time_elapsed, conflicts, best_energy, used_channels)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if M is None:
        M = create_channel_interference_matrix(K)
    
    # Initialisation
    if x_init is not None:
        # 1. Construire Y directement à partir de x_init
        Y = np.zeros((N, K), dtype=int)
        for i in range(N):
            Y[i, x_init[i]] = 1
        
        # 2. Initialiser X pour que argmax donne x_init
        X = np.random.uniform(-0.1, 0.1, (N, K))
        for i in range(N):
            X[i, x_init[i]] = 5.0   # forte valeur pour le canal choisi
            # les autres restent faibles (positifs ou négatifs)
        
        # 3. Vérification (optionnelle)
        Y_test = activation_argmax(X)
        # Si Y_test != Y, on force X pour corriger
        if not np.array_equal(Y_test, Y):
            # on remet des valeurs extrêmes pour être sûr
            X = np.zeros((N, K))
            for i in range(N):
                X[i, x_init[i]] = 10.0
                for k in range(K):
                    if k != x_init[i]:
                        X[i, k] = -10.0
        
        # 4. x est exactement x_init
        x = x_init.copy()
    else:
        X = np.random.uniform(-0.5, 0.5, (N, K))
        Y = activation_argmax(X)
        x = y_to_channels(Y)
    
    history = [compute_spectrum_energy(x, W, M)]
    best_x = x.copy()
    best_energy = history[0]
    
    if verbose:
        print(f"  [Spectrum BD-CeNN] Init : coût global = {best_energy:.4f}")
    
    start_time = time.perf_counter()
    
    for it in range(max_iter):
        X_old = X.copy()
        Y_old = Y.copy()
        
        # Mise à jour synchrone
        for i in range(N):
            for k in range(K):
                local_energy = 0.0
                for j in range(N):
                    if W[i, j] > 0:
                        for l in range(K):
                            local_energy += W[i, j] * M[k, l] * Y_old[j, l]
                dx = -alpha * X_old[i, k] - beta * local_energy + gamma * 0.0
                X[i, k] = X_old[i, k] + dt * dx
        
        Y = activation_argmax(X)
        x_new = y_to_channels(Y)
        current_energy = compute_spectrum_energy(x_new, W, M)
        history.append(current_energy)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_x = x_new.copy()
        
        if verbose and (it % 10 == 0 or it == max_iter - 1):
            print(f"  [Spectrum BD-CeNN] Itération {it+1}/{max_iter} : coût global = {current_energy:.4f}")
        
        if np.array_equal(Y, Y_old):
            if verbose:
                print(f"  [Spectrum BD-CeNN] Stabilisation à l'itération {it+1}")
            break
        if current_energy == 0.0:
            if verbose:
                print(f"  [Spectrum BD-CeNN] Coût nul atteint à l'itération {it+1}")
            break
    
    elapsed = time.perf_counter() - start_time
    conflicts = count_conflicts(best_x, W)
    used_channels = len(set(best_x))
    return best_x, history, elapsed, conflicts, best_energy, used_channels