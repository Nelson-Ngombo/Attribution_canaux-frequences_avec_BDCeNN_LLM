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
    Renders and exports graphical artifacts for scenarios S1 through S7
    based on the central database JSON file.
    """
    fig_dir = config.FIGURES_DIR
    os.makedirs(fig_dir, exist_ok=True)

    if not config.SCENARIOS_FILE.exists():
        print(f"[ERROR] Central scenarios file '{config.SCENARIOS_FILE}' not found.")
        return

    with open(config.SCENARIOS_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Global Inter-Channel Interference Matrix M (K=8)
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

    plt.colorbar(im, label="Inter-channel Interference Weight", shrink=0.8)
    plt.title(f"Channel Adjacency Matrix M (K={K_M})", fontsize=12, fontweight="bold")
    plt.xlabel("Channel l", fontsize=10)
    plt.ylabel("Channel k", fontsize=10)
    plt.xticks(np.arange(K_M), labels=[str(i) for i in range(K_M)])
    plt.yticks(np.arange(K_M), labels=[str(i) for i in range(K_M)])
    plt.tight_layout()
    plt.savefig(fig_dir / "matrix_M.png", dpi=300)
    plt.close()
    print(f"[SUCCESS] Channel interference matrix saved: {fig_dir / 'matrix_M.png'}")

    # -------------------------------------------------------------------------
    # 2. Topology Graphs and Weight Matrices W for Scenarios S1-S7
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

        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if W[i, j] > 0:
                    G.add_edge(i, j, weight=W[i, j])

        pos_dict = {i: tuple(positions[i]) for i in range(N)}

        # Render Topology Graph
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
            Patch(facecolor="red", edgecolor="red", label="Severe (weight >= 4)"),
            Patch(facecolor="orange", edgecolor="orange", label="Moderate (weight >= 2)"),
            Patch(facecolor="gray", edgecolor="gray", label="Mild (weight >= 1)"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize=10)
        plt.title(f"Scenario {name} - Interference Graph (N={N}, K={K}, seed={seed})", fontsize=12, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fig_dir / f"graph_{name}.png", dpi=300)
        plt.close()
        print(f"  [INFO] Graph saved: graph_{name}.png")

        # Render Spatial Matrix W
        plt.figure(figsize=(8, 6))
        im = plt.imshow(W, cmap="Reds", interpolation="nearest", vmin=0, vmax=4)
        if N <= 15:
            for i in range(N):
                for j in range(N):
                    if W[i, j] > 0:
                        plt.text(
                            j, i, int(W[i, j]),
                            ha="center", va="center",
                            color="black", fontsize=9, fontweight="bold",
                        )
        plt.colorbar(im, label="Spatial Interference Level", shrink=0.8)
        plt.title(f"Scenario {name} - Weight Matrix W (N={N}, seed={seed})", fontsize=12, fontweight="bold")
        plt.xlabel("Cell j", fontsize=10)
        plt.ylabel("Cell i", fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_dir / f"matrix_W_{name}.png", dpi=300)
        plt.close()
        print(f"  [INFO] Matrix saved: matrix_W_{name}.png")

    print(f"[SUCCESS] All static scenario visualizations exported to {fig_dir}")


if __name__ == "__main__":
    generate_scenario_visualizations()