# visualize_scenarios.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from data_generator import all_data
from bdcenn_spectrum import create_channel_interference_matrix
import config
from matplotlib.patches import Patch
import os

# On crée le dossier figures s'il n'existe pas
fig_dir = config.FIGURES_DIR
os.makedirs(fig_dir, exist_ok=True)

# --- 1. Générer la matrice d'interférence entre canaux M (une seule fois pour K=8) ---
# On prend K=8 comme cas représentatif
K_M = 8
M = create_channel_interference_matrix(K_M, decay=0.5, cutoff=2)
# On arrondit pour avoir des valeurs visibles (0, 0.5, 1) mais on garde les labels des axes en entiers
plt.figure(figsize=(6, 5))
im = plt.imshow(M, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)

# Ajouter les valeurs dans les cases (avec 1 décimale pour 0.5)
for i in range(K_M):
    for j in range(K_M):
        val = M[i, j]
        label = f"{val:.1f}" if val % 1 != 0 else f"{int(val)}"
        plt.text(j, i, label, ha='center', va='center',
                 color='black' if val <= 0.5 else 'white',
                 fontsize=9, fontweight='bold')

plt.colorbar(im, label="Interférence entre canaux", shrink=0.8)
plt.title(f"Matrice d'interférence entre canaux M (K={K_M})", fontsize=14, fontweight='bold')
plt.xlabel("Canal j", fontsize=12)
plt.ylabel("Canal i", fontsize=12)

# --- FORCER LES ÉTIQUETTES DES AXES EN ENTIERS ---
plt.xticks(np.arange(K_M), labels=[str(i) for i in range(K_M)])
plt.yticks(np.arange(K_M), labels=[str(i) for i in range(K_M)])

plt.tight_layout()
plt.savefig(fig_dir / "matrix_M.png", dpi=300)
plt.close()
print(f"✅ Matrice M sauvegardée dans {fig_dir / 'matrix_M.png'}")

# --- 2. Générer les graphes et matrices W pour chaque scénario ---
for name, data in all_data.items():
    N = data["N"]
    K = data["K"]
    seed = data["seed"]
    G = data["graph"]
    positions = data["positions"]
    pos_dict = {i: tuple(positions[i]) for i in range(N)}
    
    # ---------- A. Graphe d'interférence ----------
    plt.figure(figsize=(10, 8))
    node_size = 300 if N <= 15 else 100 if N <= 30 else 60
    nx.draw_networkx_nodes(G, pos_dict, node_size=node_size,
                           node_color='lightblue', edgecolors='black', linewidths=1)
    for u, v, w in G.edges(data='weight'):
        if w >= 4:
            color, width = 'red', 3.5
        elif w >= 2:
            color, width = 'orange', 2.5
        else:
            color, width = 'gray', 1.5
        nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
                               width=width, edge_color=color)
    if N <= 20:
        nx.draw_networkx_labels(G, pos_dict, font_size=9 if N <= 15 else 7)
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Critique (poids ≥ 4)'),
        Patch(facecolor='orange', edgecolor='orange', label='Moyenne (poids ≥ 2)'),
        Patch(facecolor='gray', edgecolor='gray', label='Faible (poids ≥ 1)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.title(f"Scénario {name} - Graphe d'interférence (N={N}, K={K}, seed={seed})", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fig_dir / f"graph_{name}.png", dpi=300)
    plt.close()
    print(f"   - Graphe {name} sauvegardé.")

    # ---------- B. Matrice d'interférence W ----------
    plt.figure(figsize=(8, 6))
    W = np.array(data["W"])
    im = plt.imshow(W, cmap='Reds', interpolation='nearest', vmin=0, vmax=4)
    if N <= 15:
        for i in range(N):
            for j in range(N):
                if W[i, j] > 0:
                    plt.text(j, i, int(W[i, j]), ha='center', va='center',
                             color='black', fontsize=9, fontweight='bold')
    plt.colorbar(im, label="Niveau d'interférence", shrink=0.8)
    plt.title(f"Scénario {name} - Matrice d'interférence W (N={N}, seed={seed})", fontsize=14, fontweight='bold')
    plt.xlabel("Cellule j")
    plt.ylabel("Cellule i")
    plt.tight_layout()
    plt.savefig(fig_dir / f"matrix_W_{name}.png", dpi=300)
    plt.close()
    print(f"   - Matrice W {name} sauvegardée.")

print(f"\n✅ Visualisation terminée. Toutes les figures sont dans {fig_dir}")