# experiments.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import json
from baselines import greedy_allocation, dsatur_allocation, random_allocation
from bdcenn_solver import bdcenn_allocation
from metrics import (
    create_channel_interference_matrix,
    compute_cochannel_cost,
    count_cochannel_conflicts,
    compute_adjacent_cost,
    count_adjacent_conflicts
)
import config
"""
Fichier regroupant toutes les expériences spécifiques du mémoire :
E1 - Vérification visuelle (petit graphe)
E3 - Impact du nombre de canaux (K)
E4 - Impact de la densité
E6 - Scalabilité (temps vs N)
E7 - Robustesse au bruit
E8 - Réseau dynamique
E9 - Minima locaux (effet des redémarrages)
Toutes les expériences utilisent les métriques co-canal et adjacent.
Elles utilisent les topologies stockées dans scenarios_data.json (seed=1 par défaut).
"""
# --- Charger les données depuis le JSON ---
try:
    with open(config.SCENARIOS_FILE, "r") as f:
        all_instances = json.load(f)
    # On garde pour chaque scénario l'instance avec seed=1 (ou la première si seed=1 absente)
    all_data = {}
    for name, instances in all_instances.items():
        if "1" in instances:
            inst = instances["1"]
        else:
            first_key = list(instances.keys())[0]
            inst = instances[first_key]
        # Reconstruire le graphe
        N = inst["N"]
        W = np.array(inst["W"])
        positions = np.array(inst["positions"])
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i+1, N):
                if W[i, j] > 0:
                    G.add_edge(i, j, weight=W[i, j])
        all_data[name] = {
            "seed": inst["seed"],
            "N": N,
            "K": inst["K"],
            "threshold": inst["threshold"],
            "positions": positions.tolist(),
            "W": W.tolist(),
            "graph": G
        }
    # Pas d'affichage dans le terminal
except FileNotFoundError:
    print("❌ Fichier scenarios_data.json introuvable. Veuillez d'abord exécuter data_generator.py")
    import sys
    sys.exit(1)

# --- Création des dossiers de sortie ---
os.makedirs(config.CSV_DIR, exist_ok=True)
os.makedirs(config.FIGURES_DIR, exist_ok=True)


# ============================================================================
# E1 – VÉRIFICATION VISUELLE (Petit graphe)
# ============================================================================
def run_experiment_E1(verbose=False):
    """
    E1 - Vérification visuelle
    Produit les sorties obligatoires :
        - Graphe avant (Random) en une image
        - Graphe après (BD-CeNN) en une image séparée
        - Table cellule-canal avec coût co-canal et adjacent
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    scenario_name = "S1"
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W = np.array(data["W"])
    positions = data["positions"]
    G = data["graph"]
    M = create_channel_interference_matrix(K)
    seed = data["seed"]  # seed=1

    # 1. Random
    np.random.seed(seed)
    x_rand = random_allocation(N, K)
    co_cost_rand = compute_cochannel_cost(x_rand, W)
    co_conf_rand = count_cochannel_conflicts(x_rand, W)
    adj_cost_rand = compute_adjacent_cost(x_rand, W, M)
    adj_conf_rand = count_adjacent_conflicts(x_rand, W, M)

    # 2. Greedy
    x_greedy = greedy_allocation(N, K, W, M=M)
    co_cost_greedy = compute_cochannel_cost(x_greedy, W)
    co_conf_greedy = count_cochannel_conflicts(x_greedy, W)
    adj_cost_greedy = compute_adjacent_cost(x_greedy, W, M)
    adj_conf_greedy = count_adjacent_conflicts(x_greedy, W, M)

    # 3. DSATUR
    x_dsatur = dsatur_allocation(N, K, W, M=M)
    co_cost_dsatur = compute_cochannel_cost(x_dsatur, W)
    co_conf_dsatur = count_cochannel_conflicts(x_dsatur, W)
    adj_cost_dsatur = compute_adjacent_cost(x_dsatur, W, M)
    adj_conf_dsatur = count_adjacent_conflicts(x_dsatur, W, M)

    # 4. BD-CeNN
    x_bd, _, _, _, _ = bdcenn_allocation(N, K, W, M=M, num_restarts=10, max_iter=50, seed=seed)
    co_cost_bd = compute_cochannel_cost(x_bd, W)
    co_conf_bd = count_cochannel_conflicts(x_bd, W)
    adj_cost_bd = compute_adjacent_cost(x_bd, W, M)
    adj_conf_bd = count_adjacent_conflicts(x_bd, W, M)

    # Table
    df = pd.DataFrame({
        "Méthode": ["Random", "Greedy", "DSATUR", "BD-CeNN"],
        "Allocation": [x_rand.tolist(), x_greedy.tolist(), x_dsatur.tolist(), x_bd.tolist()],
        "Co-canal coût": [co_cost_rand, co_cost_greedy, co_cost_dsatur, co_cost_bd],
        "Co-canal conflits": [co_conf_rand, co_conf_greedy, co_conf_dsatur, co_conf_bd],
        "Adjacent coût": [adj_cost_rand, adj_cost_greedy, adj_cost_dsatur, adj_cost_bd],
        "Adjacent conflits": [adj_conf_rand, adj_conf_greedy, adj_conf_dsatur, adj_conf_bd]
    })
    df.to_csv(config.CSV_DIR / "E1_visual_check.csv", index=False)

    # Palette de couleurs
    channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = {c: channel_colors[c % len(channel_colors)] for c in range(K)}
    pos_dict = {i: tuple(positions[i]) for i in range(N)}

    def draw_graph(ax, x, title):
        node_colors = [colors[c] for c in x]
        nx.draw_networkx_nodes(G, pos_dict, ax=ax, node_color=node_colors, node_size=500, edgecolors='black', linewidths=1)
        nx.draw_networkx_edges(G, pos_dict, ax=ax, edge_color='gray', width=2)
        nx.draw_networkx_labels(G, pos_dict, ax=ax, font_size=10, font_weight='bold')
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # Figure avant (Random)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    draw_graph(ax1, x_rand, f"Random (adjacent coût = {adj_cost_rand:.1f}, conflits = {adj_conf_rand})")
    legend_elements = [Patch(facecolor=colors[c], edgecolor='black', label=f'Canal {c}') for c in range(K)]
    fig1.legend(handles=legend_elements, loc='lower center', ncol=K, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle(f"E1 - Graphe avant (Random) - S1 (N={N}, K={K})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E1_graph_before_Random.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Figure après (BD-CeNN)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    draw_graph(ax2, x_bd, f"BD-CeNN (adjacent coût = {adj_cost_bd:.1f}, conflits = {adj_conf_bd})")
    legend_elements = [Patch(facecolor=colors[c], edgecolor='black', label=f'Canal {c}') for c in range(K)]
    fig2.legend(handles=legend_elements, loc='lower center', ncol=K, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle(f"E1 - Graphe après (BD-CeNN) - S1 (N={N}, K={K})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E1_graph_after_BDCeNN.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print("✅ E1 terminée.")


# ============================================================================
# E3 – IMPACT DU NOMBRE DE CANAUX (K)
# ============================================================================
def run_experiment_E3(verbose=False):
    """
    E3 - Impact du nombre de canaux
    Mesure l'évolution du coût adjacent et des conflits adjacents en fonction de K.
    """
    scenario_name = "S4"
    data = all_data[scenario_name]
    N = data["N"]
    W = np.array(data["W"])
    seed = data["seed"]  # seed=1
    K_values = [2, 3, 4, 5, 6, 8]

    results = []
    for K in K_values:
        M = create_channel_interference_matrix(K)
        x_bd, _, _, _, _ = bdcenn_allocation(N, K, W, M=M, num_restarts=10, max_iter=50, seed=seed)
        adj_cost = compute_adjacent_cost(x_bd, W, M)
        adj_conf = count_adjacent_conflicts(x_bd, W, M)
        co_cost = compute_cochannel_cost(x_bd, W)
        co_conf = count_cochannel_conflicts(x_bd, W)
        results.append({
            "K": K,
            "Adjacent coût": adj_cost,
            "Adjacent conflits": adj_conf,
            "Co-canal coût": co_cost,
            "Co-canal conflits": co_conf
        })

    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E3_impact_K.csv", index=False)

    # Figure avec double axe pour adjacent
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(df["K"], df["Adjacent coût"], marker='o', color='red', label='Coût adjacent')
    ax2.plot(df["K"], df["Adjacent conflits"], marker='s', color='blue', label='Conflits adjacents')
    ax1.set_xlabel("Nombre de canaux K")
    ax1.set_ylabel("Coût global J(x) (adjacent)", color='red')
    ax2.set_ylabel("Conflits adjacents", color='blue')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"E3 - Impact de K sur le coût adjacent et les conflits ({scenario_name}, N={N})")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E3_impact_K.png", dpi=300)
    plt.close()
    print("✅ E3 terminée.")


# ============================================================================
# E4 – IMPACT DE LA DENSITÉ
# ============================================================================
def run_experiment_E4(verbose=False):
    """
    E4 - Impact de la densité sur le coût adjacent.
    """
    scenario_name = "S4"
    base_data = all_data[scenario_name]
    N = base_data["N"]
    K = base_data["K"]
    base_seed = base_data["seed"]  # seed=1

    density_configs = [
        {"label": "Faible", "threshold": 30},
        {"label": "Moyenne", "threshold": 50},
        {"label": "Forte", "threshold": 70}
    ]

    results = []
    for cfg in density_configs:
        th = cfg["threshold"]
        np.random.seed(base_seed)
        area = 200
        positions = np.random.rand(N, 2) * area
        W = np.zeros((N, N))
        edge_count = 0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < th * 0.4:
                    w = 4
                elif dist < th * 0.65:
                    w = 2
                elif dist < th:
                    w = 1
                else:
                    w = 0
                W[i, j] = w
                W[j, i] = w
                if w > 0:
                    edge_count += 1
        density = 2 * edge_count / (N * (N - 1))

        M = create_channel_interference_matrix(K)
        x_bd, _, _, _, _ = bdcenn_allocation(N, K, W, M=M, num_restarts=10, max_iter=50, seed=base_seed)
        adj_cost = compute_adjacent_cost(x_bd, W, M)
        adj_conf = count_adjacent_conflicts(x_bd, W, M)
        co_cost = compute_cochannel_cost(x_bd, W)
        co_conf = count_cochannel_conflicts(x_bd, W)

        results.append({
            "Densité (%)": density * 100,
            "Seuil": th,
            "Arêtes": edge_count,
            "Adjacent coût": adj_cost,
            "Adjacent conflits": adj_conf,
            "Co-canal coût": co_cost,
            "Co-canal conflits": co_conf
        })

    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E4_impact_density.csv", index=False)

    # Figure : coût adjacent vs densité
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Densité (%)"], df["Adjacent coût"], marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_xlabel("Densité du graphe (%)", fontsize=12)
    ax.set_ylabel("Coût global J(x) (adjacent)", fontsize=12)
    ax.set_title(f"E4 - Impact de la densité sur le coût adjacent (N={N}, K={K})", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    for i, row in df.iterrows():
        ax.annotate(f"{row['Densité (%)']:.1f}%", (row['Densité (%)'], row['Adjacent coût']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E4_impact_density.png", dpi=300)
    plt.close()
    print("✅ E4 terminée.")


# ============================================================================
# E6 – SCALABILITÉ
# ============================================================================
def run_experiment_E6(verbose=False):
    """
    E6 - Scalabilité : temps, coût adjacent et itérations vs N.
    """
    K = 4
    area = 300
    threshold = 50
    num_restarts = 10
    max_iter = 50
    N_values = [20, 30, 50, 100, 150, 200]

    results = []
    for N in N_values:
        np.random.seed(42 + N)
        positions = np.random.rand(N, 2) * area
        W = np.zeros((N, N))
        edge_count = 0
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
                if w > 0:
                    edge_count += 1
        M = create_channel_interference_matrix(K)
        start = time.perf_counter()
        x_bd, history, elapsed, _, _ = bdcenn_allocation(
            N, K, W, M=M,
            num_restarts=num_restarts,
            max_iter=max_iter,
            seed=42 + N,
            verbose=False
        )
        adj_cost = compute_adjacent_cost(x_bd, W, M)
        iterations = len(history) - 1
        density = 2 * edge_count / (N * (N - 1))
        results.append({
            "N": N,
            "Temps (s)": elapsed,
            "Adjacent coût": adj_cost,
            "Itérations": iterations,
            "Densité": density
        })

    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E6_scalability.csv", index=False)

    # Figure avec trois sous-graphiques
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot(df["N"], df["Temps (s)"], marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel("Nombre de cellules N")
    ax1.set_ylabel("Temps d'exécution (s)")
    ax1.set_title("Temps vs N")
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.plot(df["N"], df["Adjacent coût"], marker='s', color='red', linestyle='-', linewidth=2, markersize=8)
    ax2.set_xlabel("Nombre de cellules N")
    ax2.set_ylabel("Coût global J(x) (adjacent)")
    ax2.set_title("Coût adjacent vs N")
    ax2.grid(True, linestyle='--', alpha=0.3)

    ax3.plot(df["N"], df["Itérations"], marker='^', color='green', linestyle='-', linewidth=2, markersize=8)
    ax3.set_xlabel("Nombre de cellules N")
    ax3.set_ylabel("Nombre d'itérations")
    ax3.set_title("Itérations vs N")
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.axhline(y=10, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.suptitle(f"E6 - Scalabilité (K={K}, densité ≈ 7.5%)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E6_scalability.png", dpi=300)
    plt.close()
    print("✅ E6 terminée.")


# ============================================================================
# E7 – ROBUSTESSE AU BRUIT
# ============================================================================
def run_experiment_E7(verbose=False):
    """
    E7 - Robustesse au bruit sur la matrice W.
    Compare le coût adjacent par rapport à une référence (0% bruit).
    Utilise la topologie de S6 seed=1.
    """
    scenario_name = "S6"
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W_clean = np.array(data["W"])
    M = create_channel_interference_matrix(K)
    base_seed = data["seed"]  # seed=1
    num_runs = 30
    num_restarts = 10
    max_iter = 50

    # Référence (0% bruit)
    x_ref, _, _, _, _ = bdcenn_allocation(N, K, W_clean, M=M,
                                       num_restarts=num_restarts,
                                       max_iter=max_iter,
                                       seed=base_seed)
    adj_cost_ref = compute_adjacent_cost(x_ref, W_clean, M)

    results = [{
        "Bruit (%)": 0.0,
        "Adjacent coût moyen": adj_cost_ref,
        "Écart-type coût": 0.0,
        "Cellules modifiées": 0,
        "Taux changement (%)": 0.0,
        "Coût relatif (%)": 0.0
    }]

    noise_levels = [0.05, 0.10, 0.20]
    for noise in noise_levels:
        costs = []
        changes = []
        for run_seed in range(num_runs):
            np.random.seed(base_seed + run_seed + int(noise * 1000))
            noise_std = noise * np.max(W_clean)
            W_noisy = W_clean + np.random.normal(0, noise_std, W_clean.shape)
            W_noisy = np.clip(W_noisy, 0, None)
            W_noisy = np.round(W_noisy)
            for i in range(N):
                for j in range(i + 1, N):
                    W_noisy[j, i] = W_noisy[i, j]
            W_noisy[W_noisy < 0] = 0
            np.fill_diagonal(W_noisy, 0)

            x_bd, _, _, _, _ = bdcenn_allocation(N, K, W_noisy, M=M,
                                              num_restarts=num_restarts,
                                              max_iter=max_iter,
                                              seed=base_seed + run_seed + 100)
            cost = compute_adjacent_cost(x_bd, W_noisy, M)
            change = np.sum(x_bd != x_ref)
            costs.append(cost)
            changes.append(change)

        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        mean_changes = int(round(np.mean(changes)))
        change_rate = (mean_changes / N) * 100
        degradation = ((mean_cost - adj_cost_ref) / adj_cost_ref) * 100

        results.append({
            "Bruit (%)": noise * 100,
            "Adjacent coût moyen": mean_cost,
            "Écart-type coût": std_cost,
            "Cellules modifiées": mean_changes,
            "Taux changement (%)": change_rate,
            "Coût relatif (%)": degradation
        })

    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E7_noise_robustness.csv", index=False)

    # Figure : coût adjacent vs bruit
    plt.figure(figsize=(10, 6))
    plt.errorbar(df["Bruit (%)"], df["Adjacent coût moyen"],
                 yerr=df["Écart-type coût"],
                 marker='o', capsize=5, color='red',
                 ecolor='gray', elinewidth=2, capthick=2)
    plt.xlabel("Niveau de bruit (%)")
    plt.ylabel("Coût global J(x) (adjacent)")
    plt.title(f"E7 - Robustesse au bruit (N={N}, K={K})")
    plt.grid(True, linestyle='--', alpha=0.3)
    for i, row in df.iterrows():
        if row["Bruit (%)"] > 0:
            plt.annotate(f"{row['Coût relatif (%)']:.1f}%",
                         (row["Bruit (%)"], row["Adjacent coût moyen"]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E7_noise_robustness.png", dpi=300)
    plt.close()
    print("✅ E7 terminée.")


# ============================================================================
# E8 – RÉSEAU DYNAMIQUE
# ============================================================================
def run_experiment_E8(verbose=False):
    """
    E8 - Réseau dynamique : modification de 5%, 10%, 20% des arêtes/poids.
    Compare adaptation (depuis ancienne solution) vs réinitialisation.
    Utilise la topologie de S7 seed=1.
    """
    scenario_name = "S7"
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W_original = np.array(data["W"])
    M = create_channel_interference_matrix(K)
    base_seed = data["seed"]  # seed=1
    num_runs = 30
    num_restarts = 10
    max_iter = 50

    # Référence
    x_ref, _, _, _, _ = bdcenn_allocation(N, K, W_original, M=M,
                                       num_restarts=num_restarts,
                                       max_iter=max_iter,
                                       seed=base_seed)
    adj_cost_ref = compute_adjacent_cost(x_ref, W_original, M)

    modifications = [0.05, 0.10, 0.20]
    results = []

    for mod in modifications:
        costs_adapt = []
        costs_scratch = []
        changes = []
        times_adapt = []

        for run_seed in range(num_runs):
            np.random.seed(base_seed + run_seed + int(mod * 1000))
            W_modified = W_original.copy()
            edges = [(i, j) for i in range(N) for j in range(i + 1, N) if W_original[i, j] > 0]
            num_mod = int(len(edges) * mod)
            indices = np.random.choice(len(edges), num_mod, replace=False)
            for idx in indices:
                i, j = edges[idx]
                if np.random.random() > 0.5:
                    W_modified[i, j] = 0
                    W_modified[j, i] = 0
                else:
                    current = W_modified[i, j]
                    if current > 0:
                        new_w = current + np.random.choice([-1, 1]) * min(current, 1)
                        new_w = max(0, new_w)
                        W_modified[i, j] = new_w
                        W_modified[j, i] = new_w

            # Adaptation (départ de x_ref)
            start = time.perf_counter()
            x_adapt, _, _, _, _ = bdcenn_allocation(
                N, K, W_modified, M=M,
                num_restarts=1,
                max_iter=30,
                seed=base_seed + run_seed,
                verbose=False
            )
            time_adapt = time.perf_counter() - start
            cost_adapt = compute_adjacent_cost(x_adapt, W_modified, M)

            # Depuis zéro (redémarrages multiples)
            x_scratch, _, _, _, _ = bdcenn_allocation(
                N, K, W_modified, M=M,
                num_restarts=num_restarts,
                max_iter=max_iter,
                seed=base_seed + run_seed + 1000,
                verbose=False
            )
            cost_scratch = compute_adjacent_cost(x_scratch, W_modified, M)

            change = np.sum(x_adapt != x_ref)

            costs_adapt.append(cost_adapt)
            costs_scratch.append(cost_scratch)
            changes.append(change)
            times_adapt.append(time_adapt)

        mean_adapt = np.mean(costs_adapt)
        std_adapt = np.std(costs_adapt)
        mean_scratch = np.mean(costs_scratch)
        std_scratch = np.std(costs_scratch)
        mean_changes = int(round(np.mean(changes)))
        mean_time = np.mean(times_adapt)
        gain = mean_scratch - mean_adapt

        results.append({
            "Modification (%)": mod * 100,
            "Coût adapté": mean_adapt,
            "Écart-type adapté": std_adapt,
            "Coût depuis zéro": mean_scratch,
            "Écart-type depuis zéro": std_scratch,
            "Réaffectations": mean_changes,
            "Temps adaptation (s)": mean_time,
            "Gain (zéro - adapté)": gain
        })

    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E8_dynamic_network.csv", index=False)

    # Figure : coût adapté vs depuis zéro
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(df["Modification (%)"], df["Coût adapté"],
                yerr=df["Écart-type adapté"],
                marker='o', capsize=5, color='red', label='Adapté', linewidth=2)
    ax.errorbar(df["Modification (%)"], df["Coût depuis zéro"],
                yerr=df["Écart-type depuis zéro"],
                marker='s', capsize=5, color='blue', label='Depuis zéro', linewidth=2)
    ax.set_xlabel("Taux de modification du réseau (%)")
    ax.set_ylabel("Coût global J(x) (adjacent)")
    ax.set_title(f"E8 - Coût adapté vs depuis zéro (N={N}, K={K})")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    for i, row in df.iterrows():
        ax.annotate(f"Gain={row['Gain (zéro - adapté)']:+.1f}",
                    (row["Modification (%)"], row["Coût adapté"]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E8_cost_comparison.png", dpi=300)
    plt.close()
    print("✅ E8 terminée.")


# ============================================================================
# E9 – MINIMA LOCAUX (effet des redémarrages)
# ============================================================================
def run_experiment_E9(verbose=False):
    """
    E9 - Minima locaux : effet du nombre de redémarrages sur le coût adjacent et le temps.
    """
    scenario_name = "S4"
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W = np.array(data["W"])
    M = create_channel_interference_matrix(K)
    seed = data["seed"]  # seed=1

    restart_values = [1, 5, 10, 20]
    results = []

    for num_restarts in restart_values:
        start_time = time.perf_counter()
        x_bd, _, _, _, _ = bdcenn_allocation(
            N, K, W, M=M,
            num_restarts=num_restarts,
            max_iter=50,
            seed=seed,
            verbose=False
        )
        elapsed = time.perf_counter() - start_time
        adj_cost = compute_adjacent_cost(x_bd, W, M)
        adj_conf = count_adjacent_conflicts(x_bd, W, M)
        co_cost = compute_cochannel_cost(x_bd, W)
        co_conf = count_cochannel_conflicts(x_bd, W)
        results.append({
            "Redémarrages": num_restarts,
            "Adjacent coût": adj_cost,
            "Adjacent conflits": adj_conf,
            "Co-canal coût": co_cost,
            "Co-canal conflits": co_conf,
            "Temps total (s)": elapsed
        })

    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E9_restart_effect.csv", index=False)

    # Figure : coût et temps vs redémarrages
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["Redémarrages"], df["Adjacent coût"], marker='o', color='red', linestyle='-', linewidth=2, markersize=8, label='Coût adjacent')
    ax1.set_xlabel("Nombre de redémarrages")
    ax1.set_ylabel("Meilleur coût adjacent", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["Redémarrages"], df["Temps total (s)"], marker='s', color='blue', linestyle='--', linewidth=2, markersize=8, label='Temps total')
    ax2.set_ylabel("Temps total d'exécution (s)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    for i, row in df.iterrows():
        ax1.annotate(f"{row['Adjacent coût']:.1f}", (row['Redémarrages'], row['Adjacent coût']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        ax2.annotate(f"{row['Temps total (s)']:.3f}s", (row['Redémarrages'], row['Temps total (s)']),
                     textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title(f"E9 - Effet des redémarrages sur le coût adjacent et le temps (N={N}, K={K})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E9_restart_effect.png", dpi=300)
    plt.close()
    print("✅ E9 terminée.")


# ============================================================================
# ORCHESTRATEUR PRINCIPAL
# ============================================================================
def run_all_experiments(verbose=False):
    """
    Lance toutes les expériences (E1, E3, E4, E6, E7, E8, E9).
    """
    print("\n" + "🚀 LANCEMENT DES EXPÉRIENCES SPÉCIFIQUES")
    print("=" * 80)

    run_experiment_E1(verbose)
    run_experiment_E3(verbose)
    run_experiment_E4(verbose)
    run_experiment_E6(verbose)
    run_experiment_E7(verbose)
    run_experiment_E8(verbose)
    run_experiment_E9(verbose)

    print("\n" + "=" * 80)
    print("✅ TOUTES LES EXPÉRIENCES SONT TERMINÉES.")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()