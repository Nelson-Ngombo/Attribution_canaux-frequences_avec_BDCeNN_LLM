# data_generator.py
import numpy as np
import json
import networkx as nx
from config import SEED_BASE, SCENARIOS, SCENARIOS_FILE  # <-- utilisation de SCENARIOS_FILE

def generate_scenario(name, params, seed_offset=0):
    # ... (inchangé)
    seed = SEED_BASE + seed_offset
    np.random.seed(seed)
    N = params["N"]
    K = params["K"]
    area = params["area"]
    threshold = params["threshold"]
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
    return {
        "name": name,
        "seed": seed,
        "N": N,
        "K": K,
        "threshold": threshold,
        "positions": positions.tolist(),
        "W": W.tolist(),
        "graph": G
    }

all_data = {}
for idx, (name, params) in enumerate(SCENARIOS.items()):
    all_data[name] = generate_scenario(name, params, seed_offset=idx)
    print(f"✅ Scénario {name} généré avec la seed {all_data[name]['seed']} : N={params['N']}, K={params['K']}, threshold={params['threshold']}")

data_to_save = {}
for name, data in all_data.items():
    data_to_save[name] = {
        "seed": data["seed"],
        "N": data["N"],
        "K": data["K"],
        "threshold": data["threshold"],
        "positions": data["positions"],
        "W": data["W"]
    }

# Sauvegarde dans le dossier data/ via config
with open(SCENARIOS_FILE, "w") as f:
    json.dump(data_to_save, f, indent=4)

print(f"💾 Données sauvegardées dans '{SCENARIOS_FILE}' avec les seeds suivantes :")
for name, data in data_to_save.items():
    print(f"   - {name} : seed {data['seed']}, threshold = {data['threshold']}")