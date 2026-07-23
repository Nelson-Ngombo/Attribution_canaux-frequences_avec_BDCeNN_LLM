# baselines.py
import numpy as np

def create_channel_interference_matrix(K, decay=0.5, cutoff=2):
    """
    Crée une matrice d'interférence entre canaux M (K x K).
    (Copiée localement pour éviter une dépendance circulaire.)
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

def random_allocation(N, K):
    """Allocation aléatoire uniforme."""
    return np.random.randint(0, K, size=N)

def greedy_allocation(N, K, W, order=None, M=None):
    """
    Allocation gloutonne séquentielle, avec prise en compte de la matrice
    d'interférence entre canaux M (spectrum-aware).
    - Si M est None, on utilise le comportement classique (même canal uniquement).
    - Si M est fournie, le coût local devient :
        cost = sum_{j voisin déjà coloré} W[i][j] * M[c][x[j]]
    """
    if M is None:
        # Comportement classique : M est une matrice identité (ou on simule)
        # Pour éviter de créer une grosse matrice, on utilise une condition simple.
        use_spectrum = False
    else:
        use_spectrum = True

    if order is None:
        order = list(range(N))
    
    x = np.full(N, -1, dtype=int)  # -1 signifie non attribué
    
    for idx, i in enumerate(order):
        best_c = -1
        best_cost = float('inf')
        for c in range(K):
            local_cost = 0
            for j in order[:idx]:
                if W[i][j] > 0:
                    if use_spectrum:
                        # Spectrum-aware : on utilise M
                        local_cost += W[i][j] * M[c, x[j]]
                    else:
                        # Classique : même canal uniquement
                        if x[j] == c:
                            local_cost += W[i][j]
            if local_cost < best_cost:
                best_cost = local_cost
                best_c = c
        x[i] = best_c
    return x

def dsatur_allocation(N, K, W, M=None):
    """
    Allocation par DSATUR, avec prise en compte de la matrice
    d'interférence entre canaux M (spectrum-aware).
    """
    # Calcul du degré (nombre de voisins avec W > 0)
    degree = np.sum(W > 0, axis=1)
    
    colored = np.full(N, False, dtype=bool)
    x = np.full(N, -1, dtype=int)
    neighbor_colors = [set() for _ in range(N)]
    
    use_spectrum = (M is not None)
    
    def select_next():
        best_vertex = -1
        best_sat = -1
        best_deg = -1
        for v in range(N):
            if not colored[v]:
                sat = len(neighbor_colors[v])
                if sat > best_sat or (sat == best_sat and degree[v] > best_deg):
                    best_sat = sat
                    best_deg = degree[v]
                    best_vertex = v
        return best_vertex
    
    for _ in range(N):
        v = select_next()
        best_c = -1
        best_cost = float('inf')
        for c in range(K):
            local_cost = 0
            for j in range(N):
                if colored[j] and W[v][j] > 0:
                    if use_spectrum:
                        local_cost += W[v][j] * M[c, x[j]]
                    else:
                        if x[j] == c:
                            local_cost += W[v][j]
            if local_cost < best_cost:
                best_cost = local_cost
                best_c = c
        x[v] = best_c
        colored[v] = True
        for j in range(N):
            if not colored[j] and W[v][j] > 0:
                neighbor_colors[j].add(best_c)
    
    return x