# visualize_scenarios.py

"""
Static visualization generator for standard test scenarios (S1-S7).
Generates standalone network graph topologies, spatial weight matrices W,
and channel interference matrices M for reporting and verification.
"""

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import numpy as np

import config
from metrics import create_channel_interference_matrix


def generate_scenario_visualizations():
    """
    Génère et exporte l'ensemble des fichiers graphiques pour les scénarios
    de référence définis dans la base de données centrale JSON.
    """
    fig_dir = config.FIGURES_DIR
    os.makedirs(fig_dir, exist_ok=True)

    if not config.SCENARIOS_FILE.exists():
        print(f"[ERROR] Le fichier de scénarios '{config.SCENARIOS_FILE}' est introuvable.")
        return

    with open(config.SCENARIOS_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Représentation thermique de la matrice spectrale M (K=8)
    # -------------------------------------------------------------------------
    K_M = 8
    M = create_channel_interference_matrix(K_M, decay=0.5, cutoff=2)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)

    for i in range(K_M):
        for j in range(K_M):
            val = M[i, j]
            label = f"{val:.1f}" if val % 1 != 0 else f"{int(val)}"
            plt.text(
                j, i, label,
                ha="center", va="center",
                color="black" if val <= 0.5 else "white",
                fontsize=9, fontweight="bold",
            )

    plt.colorbar(im, label="Intensité d'atténuation d'interférence", shrink=0.8)
    plt.title(f"Matrice d'atténuation inter-canaux M (K={K_M})", fontsize=11, fontweight="bold")
    plt.xlabel("Canal l", fontsize=10)
    plt.ylabel("Canal k", fontsize=10)
    plt.xticks(np.arange(K_M), labels=[str(i) for i in range(K_M)])
    plt.yticks(np.arange(K_M), labels=[str(i) for i in range(K_M)])
    plt.tight_layout()
    output_m_path = str(fig_dir / "matrix_M.png")
    plt.savefig(output_m_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Matrice spectrale sauvegardée : {output_m_path}")

    # -------------------------------------------------------------------------
    # 2. Topologies cellulaires et matrices W associées
    # -------------------------------------------------------------------------
    for name, instances in all_data.items():
        instance = instances.get("1")
        if instance is None:
            first_seed = list(instances.keys())[0]
            instance = instances[first_seed]

        N = instance["N"]
        K = instance["K"]
        seed = instance["seed"]
        positions = np.array(instance["positions"])
        W = np.array(instance["W"])

        # Reconstruction du graphe d'interférence
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if W[i, j] > 0:
                    G.add_edge(i, j, weight=W[i, j])

        pos_dict = {i: tuple(positions[i]) for i in range(N)}

        # A. Rendu du Graphe de topologie
        plt.figure(figsize=(10, 8))
        node_size = 300 if N <= 15 else (100 if N <= 30 else 60)
        nx.draw_networkx_nodes(
            G, pos_dict,
            node_size=node_size,
            node_color="lightblue",
            edgecolors="black",
            linewidths=1,
        )

        for u, v, w in G.edges(data="weight"):
            if w >= 4:
                color, width = "red", 3.5
            elif w >= 2:
                color, width = "orange", 2.5
            else:
                color, width = "gray", 1.5
            nx.draw_networkx_edges(
                G, pos_dict,
                edgelist=[(u, v)],
                width=width,
                edge_color=color,
            )

        if N <= 20:
            nx.draw_networkx_labels(G, pos_dict, font_size=9 if N <= 15 else 7)

        legend_elements = [
            Patch(facecolor="red", edgecolor="red", label="Couplage severe (poids >= 4)"),
            Patch(facecolor="orange", edgecolor="orange", label="Couplage modere (poids >= 2)"),
            Patch(facecolor="gray", edgecolor="gray", label="Couplage faible (poids >= 1)"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize=10)
        plt.title(f"Scenario {name} - Graphe d'interférence (N={N}, K={K}, seed={seed})", fontsize=12, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        output_g_path = str(fig_dir / f"graph_{name}.png")
        plt.savefig(output_g_path, dpi=300)
        plt.close()
        print(f"  [INFO] Graphe sauvegardé : graph_{name}.png")

        # B. Rendu de la matrice W 
        plt.figure(figsize=(8, 6))
        im = plt.imshow(W, cmap="Reds", interpolation="nearest", vmin=0, vmax=4)
        
        # Graduations entieres visibles pour les petits reseaux
        if N <= 30:
            plt.xticks(np.arange(N), labels=[str(i) for i in range(N)], fontsize=8 if N > 15 else 10)
            plt.yticks(np.arange(N), labels=[str(i) for i in range(N)], fontsize=8 if N > 15 else 10)
            
        if N <= 15:
            for i in range(N):
                for j in range(N):
                    if W[i, j] > 0:
                        text_color = "white" if W[i, j] >= 2 else "black"
                        plt.text(
                            j, i, int(W[i, j]),
                            ha="center", va="center",
                            color=text_color, fontsize=9, fontweight="bold",
                        )
        plt.colorbar(im, label="Intensite geographique d'interference W_ij", shrink=0.8)
        plt.title(f"Scenario {name} - Matrice W (N={N}, seed={seed})", fontsize=11, fontweight="bold")
        plt.xlabel("Cellule j", fontsize=10)
        plt.ylabel("Cellule i", fontsize=10)
        plt.tight_layout()
        output_w_path = str(fig_dir / f"matrix_W_{name}.png")
        plt.savefig(output_w_path, dpi=300)
        plt.close()
        plt.close('all')
        print(f"  [INFO] Matrice sauvegardee : matrix_W_{name}.png")

    print(f"[SUCCESS] Visualisations statiques exportées dans {fig_dir}")


if __name__ == "__main__":
    generate_scenario_visualizations()