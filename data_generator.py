# data_generator.py
"""
Network topology generator. Builds random geometric cellular layouts
and constructs interference matrices based on physical distances.
"""

import numpy as np
import json
import networkx as nx
from config import SCENARIOS, SCENARIOS_FILE, TOPOLOGY_SEEDS

def build_weight_matrix(positions, threshold):
    """
    Matrice W à quatre paliers {0, 1, 2, 4} (spécification v1.0, §2), source
    unique partagée par le générateur, les campagnes et le validateur.
    """
    positions = np.asarray(positions, dtype=float)
    N = len(positions)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < threshold * 0.4:
                w = 4
            elif dist < threshold * 0.65:
                w = 2
            elif dist < threshold:
                w = 1
            else:
                w = 0
            W[i, j] = w
            W[j, i] = w
    return W


def generate_network(N, K, area, threshold, seed):
    """
    Generates a localized network topology with positions, a weight interference matrix W,
    and a networkx graph object using a deterministic random seed.
    
    Args:
        N (int): Number of cellular nodes
        K (int): Number of available frequency channels
        area (float): Size of the 2D simulation plane
        threshold (float): Spatial distance limit for inter-cell interference
        seed (int): Reproducibility seed
        
    Returns:
        W (ndarray): N x N weight matrix containing interference levels
        positions (ndarray): N x 2 matrix representing coordinate pairs
        G (Graph): NetworkX Graph representation
    """
    # Générateur local réservé à la topologie : positions identiques bit à bit
    # à l'ancien np.random.seed(seed), sans modifier l'état global des solveurs.
    rng = np.random.RandomState(seed)
    positions = rng.rand(N, 2) * area
    W = build_weight_matrix(positions, threshold)
            
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i+1, N):
            if W[i, j] > 0:
                G.add_edge(i, j, weight=W[i, j])
                
    return W, positions, G

def apply_measurement_noise(W_ref, noise_level, seed):
    """
    Bruit de mesure sur W (E7, E10 S6) : W_bruit = max(0, W_ref + b.W_ref.eps),
    eps ~ U(-1, 1) tiré une fois par arête existante. Matrice continue (aucun
    arrondi), symétrique, diagonale nulle ; aucune arête n'est créée.
    """
    W_ref = np.asarray(W_ref, dtype=float)
    rng = np.random.RandomState(seed)
    upper = np.triu_indices(W_ref.shape[0], k=1)
    weights = W_ref[upper]
    edge_mask = weights > 0
    epsilon = rng.uniform(-1.0, 1.0, size=int(edge_mask.sum()))
    noisy = weights.copy()
    noisy[edge_mask] = np.maximum(
        0.0, weights[edge_mask] + noise_level * weights[edge_mask] * epsilon
    )
    W_noisy = np.zeros_like(W_ref)
    W_noisy[upper] = noisy
    return W_noisy + W_noisy.T


# Paliers admissibles d'une arête active (spécification v1.0)
_WEIGHT_LEVELS = (1.0, 2.0, 4.0)


def apply_dynamic_modification(W_ref, modification_level, seed):
    """
    Évolution structurelle de W (E8, E10 S7) : une fraction des arêtes est
    supprimée (probabilité 1/2) ou passe au palier voisin dans {1, 2, 4}.
    Chaque arête tirée change réellement d'état ; W reste dans {0, 1, 2, 4}.
    """
    W_dyn = np.array(W_ref, dtype=float, copy=True)
    rng = np.random.RandomState(seed)
    rows, cols = np.nonzero(np.triu(W_dyn, k=1))
    n_modified = int(len(rows) * modification_level)
    if n_modified == 0:
        return W_dyn

    for idx in rng.choice(len(rows), n_modified, replace=False):
        i, j = rows[idx], cols[idx]
        if rng.random_sample() < 0.5:
            new_weight = 0.0
        else:
            level = int(np.argmin(np.abs(np.asarray(_WEIGHT_LEVELS) - W_dyn[i, j])))
            step = 1 if rng.random_sample() < 0.5 else -1
            if not 0 <= level + step < len(_WEIGHT_LEVELS):
                step = -step
            new_weight = _WEIGHT_LEVELS[level + step]
        W_dyn[i, j] = new_weight
        W_dyn[j, i] = new_weight
    return W_dyn


def generate_all_scenarios():
    """
    Generates and saves 30 distinct spatial seeds for all configured scenarios (S1-S7).
    Saves results inside the designated central JSON file defined in config.
    """
    all_data = {}
    print("[INFO] Generating 30 distinct topological seeds for each scenario...")
    
    for name, params in SCENARIOS.items():
        N = params["N"]
        K = params["K"]
        area = params["area"]
        threshold = params["threshold"]
        
        scenario_instances = {}
        for seed in TOPOLOGY_SEEDS:
            W, positions, _ = generate_network(N, K, area, threshold, seed)
            scenario_instances[str(seed)] = {
                "seed": seed,
                "N": N,
                "K": K,
                "threshold": threshold,
                "positions": positions.tolist(),
                "W": W.tolist()
            }
        all_data[name] = scenario_instances
        print(f"  [INFO] Scenario {name} : {len(scenario_instances)} seeds successfully generated.")

    # Serialize topologies to target database file
    data_to_save = {}
    for name, instances in all_data.items():
        data_to_save[name] = {str(seed): inst for seed, inst in instances.items()}

    with open(SCENARIOS_FILE, "w") as f:
        json.dump(data_to_save, f, indent=4)

    print(f"[SUCCESS] Topologies saved in '{SCENARIOS_FILE}' containing seeds 1 to {len(TOPOLOGY_SEEDS)}.")

if __name__ == "__main__":
    generate_all_scenarios()