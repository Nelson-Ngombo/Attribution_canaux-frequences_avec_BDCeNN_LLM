#baselines.py
"""
Baseline heuristics for the Channel Assignment Problem (CAP).
Implements Random, Order-based Greedy, and DSATUR assignment algorithms.
"""

import numpy as np

def create_channel_interference_matrix(K, decay=0.5, cutoff=2):
    """
    Generates a K x K adjacent channel interference (ACI) matrix M.
    Models the physical spectral leakage between contiguous frequency bands.
    
    Args:
        K (int): Total available channels
        decay (float): Power attenuation factor per channel distance unit
        cutoff (int): Maximum channel distance experiencing spectral leakage
        
    Returns:
        M (ndarray): K x K symmetric interference decay matrix
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
    """
    Assigns channels uniformly at random across all N cells.
    
    Args:
        N (int): Number of cellular antennas
        K (int): Number of available channels
        
    Returns:
        ndarray: Allocated channel indices of size N
    """
    return np.random.randint(0, K, size=N)

def greedy_allocation(N, K, W, order=None, M=None):
    """
    Sequential Greedy Channel Allocation. Iteratively assigns the best local
    channel to minimize co-channel and adjacent-channel interference.
    
    Args:
        N (int): Number of cells
        K (int): Number of channels
        W (ndarray): Spatial cell interference matrix
        order (list): Optional custom ordering of cells to process
        M (ndarray): Optional channel adjacency penalty matrix (ACI awareness)
        
    Returns:
        ndarray: Vector of length N containing assigned channels
    """
    use_adjacent = (M is not None)

    if order is None:
        order = list(range(N))
    
    x = np.full(N, -1, dtype=int)
    
    for idx, i in enumerate(order):
        best_c = -1
        best_cost = float('inf')
        for c in range(K):
            local_cost = 0.0
            for j in order[:idx]:
                if W[i][j] > 0:
                    if use_adjacent:
                        local_cost += W[i][j] * M[c, x[j]]
                    else:
                        if x[j] == c:
                            local_cost += W[i][j]
            if local_cost < best_cost:
                best_cost = local_cost
                best_c = c
        x[i] = best_c
    return x

def dsatur_allocation(N, K, W, M=None):
    """
    Degree of Saturation (DSATUR) allocation tailored for weighted network topologies.
    Prioritizes coloring of nodes with higher numbers of distinct neighbor colors.
    
    Args:
        N (int): Number of cells
        K (int): Number of channels
        W (ndarray): Spatial cell interference matrix
        M (ndarray): Optional channel adjacency penalty matrix
        
    Returns:
        ndarray: Vector of length N containing assigned channels
    """
    degree = np.sum(W > 0, axis=1)
    colored = np.full(N, False, dtype=bool)
    x = np.full(N, -1, dtype=int)
    neighbor_colors = [set() for _ in range(N)]
    use_adjacent = (M is not None)
    
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
            local_cost = 0.0
            for j in range(N):
                if colored[j] and W[v][j] > 0:
                    if use_adjacent:
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