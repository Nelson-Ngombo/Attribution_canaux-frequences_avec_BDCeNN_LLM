# data_generator.py
"""
Network topology generator. Builds random geometric cellular layouts
and constructs interference matrices based on physical distances.
"""

import numpy as np
import json
import networkx as nx
from config import SCENARIOS, SCENARIOS_FILE, TOPOLOGY_SEEDS

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
    np.random.seed(seed)
    positions = np.random.rand(N, 2) * area
    W = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i+1, N):
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
            
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i+1, N):
            if W[i, j] > 0:
                G.add_edge(i, j, weight=W[i, j])
                
    return W, positions, G

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