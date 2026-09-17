# metrics.py
"""
Independent validation metrics module for channel assignment optimization.
Computes co-channel costs, adjacent-channel interference penalties, conflict counts,
and formal statistical confidence intervals.
"""

import numpy as np


# ============================================================================
# 1. CO-CHANNEL INTERFERENCE METRICS (CCI-ONLY)
# ============================================================================

def compute_cochannel_cost(x, W):
    """
    Computes total penalty violation under the Co-Channel Interference (CCI) model:
    J_CCI(x) = sum_{i < j, x_i == x_j} W[i, j]

    Args:
        x (ndarray): Vector of channel assignments (length N).
        W (ndarray): Symmetric interference weight matrix (N x N).

    Returns:
        float: Scalar objective cost value.
    """
    N = len(x)
    cost = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if x[i] == x[j]:
                cost += W[i, j]
    return float(cost)


def count_cochannel_conflicts(x, W):
    """
    Counts the number of active interfering links sharing identical channels.

    Args:
        x (ndarray): Vector of channel assignments (length N).
        W (ndarray): Symmetric interference weight matrix (N x N).

    Returns:
        int: Number of interfering pairs where W[i, j] > 0 and x[i] == x[j].
    """
    N = len(x)
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] > 0 and x[i] == x[j]:
                conflicts += 1
    return conflicts


def compute_metrics_cochannel(x, W):
    """
    Aggregates all scalar indicators under the Co-Channel model.

    Args:
        x (ndarray): Vector of channel assignments.
        W (ndarray): Interference weight matrix.

    Returns:
        dict: Summary containing cost, conflicts count, and distinct channels used.
    """
    cost = compute_cochannel_cost(x, W)
    conflicts = count_cochannel_conflicts(x, W)
    used_channels = len(set(x))
    return {
        "cost": cost,
        "conflicts": conflicts,
        "used_channels": used_channels,
    }


# ============================================================================
# 2. ADJACENT CHANNEL INTERFERENCE METRICS (CCI+ACI)
# ============================================================================

def create_channel_interference_matrix(K, decay=0.5, cutoff=2):
    """
    Constructs a K x K inter-channel spectral leakage penalty matrix M.
    M[k, l] = 1.0 if k == l,
              decay^(|k - l|) if |k - l| <= cutoff,
              0.0 otherwise.

    Args:
        K (int): Total number of available frequency channels.
        decay (float): Power attenuation factor per channel separation unit.
        cutoff (int): Maximum channel distance experiencing adjacent leakage.

    Returns:
        ndarray: K x K symmetric channel penalty matrix.
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


def compute_adjacent_cost(x, W, M):
    """
    Computes total penalty violation under the extended CCI+ACI model:
    J_ACI(x) = sum_{i < j, W[i, j] > 0} W[i, j] * M[x_i, x_j]

    Args:
        x (ndarray): Vector of channel assignments (length N).
        W (ndarray): Spatial cell weight matrix (N x N).
        M (ndarray): Channel spectral penalty matrix (K x K).

    Returns:
        float: Scalar objective cost accounting for co-channel and adjacent leakage.
    """
    N = len(x)
    energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] > 0:
                energy += W[i, j] * M[x[i], x[j]]
    return float(energy)


def count_adjacent_conflicts(x, W, M, threshold=0.0):
    """
    Counts pairs with active physical interference under adjacent channel leakage.

    Args:
        x (ndarray): Vector of channel assignments.
        W (ndarray): Spatial cell weight matrix.
        M (ndarray): Channel penalty matrix.
        threshold (float): Minimum penalty threshold to consider as active conflict.

    Returns:
        int: Number of interfering pairs with W[i, j] > 0 and M[x_i, x_j] > threshold.
    """
    N = len(x)
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] > 0 and M[x[i], x[j]] > threshold:
                conflicts += 1
    return conflicts


def compute_metrics_adjacent(x, W, M):
    """
    Aggregates all scalar indicators under the combined CCI+ACI model.

    Args:
        x (ndarray): Vector of channel assignments.
        W (ndarray): Spatial interference weight matrix.
        M (ndarray): Channel penalty matrix.

    Returns:
        dict: Summary containing adjacent cost, conflict count, and channels used.
    """
    cost = compute_adjacent_cost(x, W, M)
    conflicts = count_adjacent_conflicts(x, W, M)
    used_channels = len(set(x))
    return {
        "cost": cost,
        "conflicts": conflicts,
        "used_channels": used_channels,
    }


# ============================================================================
# 3. STATISTICAL FUNCTIONS
# ============================================================================

def compute_confidence_interval(data, confidence=0.95):
    """
    Computes parametric confidence interval for an array of experimental observations.

    Args:
        data (array-like): Sample measurement values.
        confidence (float): Desired confidence level (default: 0.95).

    Returns:
        tuple: (lower_bound, upper_bound)
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