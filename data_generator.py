# data_generator.py
import numpy as np
import json
import networkx as nx
from config import SCENARIOS, SCENARIOS_FILE, TOPOLOGY_SEEDS

def generate_network(N, K, area, threshold, seed):
    """
    Génère une matrice d'interférence W, les positions et le graphe G
    à partir d'une seed donnée.
    Retourne : (W, positions, G)
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

# --- Génération des 30 instances pour chaque scénario ---
all_data = {}

print("🔄 Génération des 30 instances pour chaque scénario...")
for name, params in SCENARIOS.items():
    N = params["N"]
    K = params["K"]
    area = params["area"]
    threshold = params["threshold"]
    
    scenario_instances = {}
    for seed in TOPOLOGY_SEEDS:
        W, positions, G = generate_network(N, K, area, threshold, seed)
        scenario_instances[str(seed)] = {
            "seed": seed,
            "N": N,
            "K": K,
            "threshold": threshold,
            "positions": positions.tolist(),
            "W": W.tolist()
            # Le graphe G n'est pas sauvegardé dans le JSON car il contient des objets NetworkX non sérialisables
        }
    all_data[name] = scenario_instances
    print(f"   - Scénario {name} : {len(scenario_instances)} instances générées.")

# Sauvegarde dans le JSON
data_to_save = {}
for name, instances in all_data.items():
    # On convertit les clés en chaînes de caractères pour le JSON
    data_to_save[name] = {str(seed): inst for seed, inst in instances.items()}

with open(SCENARIOS_FILE, "w") as f:
    json.dump(data_to_save, f, indent=4)

print(f"💾 Données sauvegardées dans '{SCENARIOS_FILE}' avec les seeds 1 à {len(TOPOLOGY_SEEDS)} pour chaque scénario.")