# experiments.py

"""
Comprehensive evaluation framework executing academic experiments E1 to E10.
Produces comparative tables, charts, convergence curves, and audits LLM correctness.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless execution configuration
import matplotlib.pyplot as plt
import os
import time
import json
import networkx as nx
from matplotlib.patches import Patch

import config
from baselines import greedy_allocation, dsatur_allocation, random_allocation
from bdcenn_solver import bdcenn_allocation
from metrics import (
    create_channel_interference_matrix,
    compute_cochannel_cost,
    count_cochannel_conflicts,
    compute_adjacent_cost,
    count_adjacent_conflicts
)
from llm_assistant import audit_case

# --- Database Scenario Initialization ---
try:
    with open(config.SCENARIOS_FILE, "r") as f:
        all_instances = json.load(f)
    all_data = {}
    for name, instances in all_instances.items():
        if "1" in instances:
            inst = instances["1"]
        else:
            first_key = list(instances.keys())[0]
            inst = instances[first_key]
        N = inst["N"]
        W = np.array(inst["W"])
        positions = np.array(inst["positions"])
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
except FileNotFoundError:
    print("[ERROR] 'scenarios_data.json' database not found. Execute data_generator.py first.")
    import sys
    sys.exit(1)

os.makedirs(config.CSV_DIR, exist_ok=True)
os.makedirs(config.FIGURES_DIR, exist_ok=True)


# ============================================================================
# E1 - VISUAL VALIDATION
# ============================================================================
def run_experiment_E1(verbose=False):
    """
    E1 - Visual validation of small networks.
    Generates structured topology maps, cost distributions, and compared color allocations.
    """
    scenario_name = "S1"
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W = np.array(data["W"])
    positions = data["positions"]
    G = data["graph"]
    seed = data["seed"]

    channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = {c: channel_colors[c % len(channel_colors)] for c in range(K)}
    pos_dict = {i: tuple(positions[i]) for i in range(N)}

    e1_fig_dir = config.FIGURES_DIR / "E1"
    e1_csv_dir = config.CSV_DIR / "E1"
    os.makedirs(e1_fig_dir, exist_ok=True)
    os.makedirs(e1_csv_dir, exist_ok=True)

    # 1. Base Interference Graph Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos_dict, ax=ax, node_color='lightblue', node_size=500, edgecolors='black', linewidths=1)
    nx.draw_networkx_edges(G, pos_dict, ax=ax, edge_color='gray', width=2)
    nx.draw_networkx_labels(G, pos_dict, ax=ax, font_size=10, font_weight='bold')
    ax.set_title(f"Scenario {scenario_name} - Interference Graph (N={N}, K={K}, seed={seed})", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(e1_fig_dir / "interference_graph.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[INFO] Figure E1/interference_graph.png saved.")

    # 2. Matrix W Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(W, cmap='Reds', interpolation='nearest', vmin=0, vmax=4)
    for i in range(N):
        for j in range(N):
            if W[i, j] > 0:
                ax.text(j, i, int(W[i, j]), ha='center', va='center', color='black', fontsize=9, fontweight='bold')
    plt.colorbar(im, label="Interference Level", shrink=0.8)
    ax.set_title(f"Scenario {scenario_name} - Weight Matrix W (N={N}, seed={seed})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Cell j")
    ax.set_ylabel("Cell i")
    plt.tight_layout()
    plt.savefig(e1_fig_dir / "matrix_W.png", dpi=300)
    plt.close(fig)
    print("[INFO] Figure E1/matrix_W.png saved.")

    # Helper function for coloring cell nodes
    def draw_colored_graph(ax, x, title):
        node_colors = [colors[c] for c in x]
        nx.draw_networkx_nodes(G, pos_dict, ax=ax, node_color=node_colors, node_size=500, edgecolors='black', linewidths=1)
        nx.draw_networkx_edges(G, pos_dict, ax=ax, edge_color='gray', width=2)
        nx.draw_networkx_labels(G, pos_dict, ax=ax, font_size=10, font_weight='bold')
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    def save_colored_graph(method_name, x, fig_dir, subfolder, label_suffix=""):
        fig, ax = plt.subplots(figsize=(8, 6))
        title = f"{method_name}{label_suffix}"
        draw_colored_graph(ax, x, title)
        legend_elements = [Patch(facecolor=colors[c], edgecolor='black', label=f'Channel {c}') for c in range(K)]
        fig.legend(handles=legend_elements, loc='lower center', ncol=K, fontsize=10, bbox_to_anchor=(0.5, -0.05))
        plt.suptitle(f"E1 - {fig_dir.name} - {title} - S1 (N={N}, K={K}, seed={seed})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = f"{method_name.lower().replace(' ', '_')}{label_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(fig_dir / subfolder / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Figure E1/{subfolder}/{filename} saved.")

    # Run execution pathways for co-channel and adjacent-channel models
    models = [
        {"name": "cochannel", "M": None, "suffix": " (co-channel)", "folder": "cochannel"},
        {"name": "adjacent", "M": create_channel_interference_matrix(K), "suffix": " (adjacent)", "folder": "adjacent"}
    ]

    for model in models:
        M = model["M"]
        suffix = model["suffix"]
        folder = model["folder"]
        model_fig_dir = e1_fig_dir / folder
        model_csv_dir = e1_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        np.random.seed(seed)
        x_rand = random_allocation(N, K)
        x_greedy = greedy_allocation(N, K, W, M=M)
        x_dsatur = dsatur_allocation(N, K, W, M=M)

        # Execute Multi-Restart BD-CeNN
        x_bd_final, history_bd, _, _, _ = bdcenn_allocation(
            N, K, W, M=M,
            num_restarts=config.NUM_RESTARTS,
            max_iter=config.MAX_ITER_BD,
            random_order=True,
            seed=seed,
            verbose=False
        )
        x_bd_initial = history_bd[0][2]

        # Calculate comparative metrics
        if M is None:
            cost_rand = compute_cochannel_cost(x_rand, W)
            conf_rand = count_cochannel_conflicts(x_rand, W)
            cost_greedy = compute_cochannel_cost(x_greedy, W)
            conf_greedy = count_cochannel_conflicts(x_greedy, W)
            cost_dsatur = compute_cochannel_cost(x_dsatur, W)
            conf_dsatur = count_cochannel_conflicts(x_dsatur, W)
            cost_bd = compute_cochannel_cost(x_bd_final, W)
            conf_bd = count_cochannel_conflicts(x_bd_final, W)
            cost_bd_initial = compute_cochannel_cost(x_bd_initial, W)
            conf_bd_initial = count_cochannel_conflicts(x_bd_initial, W)
        else:
            cost_rand = compute_adjacent_cost(x_rand, W, M)
            conf_rand = count_adjacent_conflicts(x_rand, W, M)
            cost_greedy = compute_adjacent_cost(x_greedy, W, M)
            conf_greedy = count_adjacent_conflicts(x_greedy, W, M)
            cost_dsatur = compute_adjacent_cost(x_dsatur, W, M)
            conf_dsatur = count_adjacent_conflicts(x_dsatur, W, M)
            cost_bd = compute_adjacent_cost(x_bd_final, W, M)
            conf_bd = count_adjacent_conflicts(x_bd_final, W, M)
            cost_bd_initial = compute_adjacent_cost(x_bd_initial, W, M)
            conf_bd_initial = count_adjacent_conflicts(x_bd_initial, W, M)

        used_rand = len(set(x_rand))
        used_greedy = len(set(x_greedy))
        used_dsatur = len(set(x_dsatur))
        used_bd = len(set(x_bd_final))

        # Save comparative colored topologies
        save_colored_graph("Random", x_rand, e1_fig_dir, folder, suffix)
        save_colored_graph("Greedy", x_greedy, e1_fig_dir, folder, suffix)
        save_colored_graph("DSATUR", x_dsatur, e1_fig_dir, folder, suffix)
        save_colored_graph("BD-CeNN initial", x_bd_initial, e1_fig_dir, folder, suffix + " (initial)")
        save_colored_graph("BD-CeNN final", x_bd_final, e1_fig_dir, folder, suffix + " (final)")

        # Save channel data csv tables
        df_cell_channels = pd.DataFrame({
            "Cell": list(range(N)),
            "Random": x_rand,
            "Greedy": x_greedy,
            "DSATUR": x_dsatur,
            "BD-CeNN_initial": x_bd_initial,
            "BD-CeNN_final": x_bd_final
        })
        df_cell_channels.to_csv(model_csv_dir / "cell_channels.csv", index=False)
        print(f"[SUCCESS] CSV E1/{folder}/cell_channels.csv saved.")

        # Save metrics comparison csv
        df_metrics = pd.DataFrame({
            "Method": ["Random", "Greedy", "DSATUR", "BD-CeNN_initial", "BD-CeNN_final"],
            "Global Cost": [cost_rand, cost_greedy, cost_dsatur, cost_bd_initial, cost_bd],
            "Conflicts": [conf_rand, conf_greedy, conf_dsatur, conf_bd_initial, conf_bd],
            "Channels Used": [used_rand, used_greedy, used_dsatur, len(set(x_bd_initial)), used_bd]
        })
        df_metrics.to_csv(model_csv_dir / "method_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E1/{folder}/method_metrics.csv saved.")

    print("[SUCCESS] Experiment E1 completed.")


# ============================================================================
# E2 - FACTORIAL RUN
# ============================================================================
def run_experiment_E2(verbose=False):
    """
    E2 - Factorial runtime comparison over multiple settings.
    Studies the sensitivity profile of N, K, and density across algorithms.
    """
    sc_instances = {}
    for sc_name in ["S2", "S3"]:
        inst = all_instances.get(sc_name, {})
        if not inst:
            print(f"[ERROR] Scenario {sc_name} missing from database.")
            return
        sc_instances[sc_name] = inst

    seeds = sorted([int(s) for s in list(sc_instances["S2"].keys()) if s.isdigit()])

    configs = [
        {"N": 30, "K": 4, "scenario": "S2", "area": 150},
        {"N": 30, "K": 6, "scenario": "S2", "area": 150},
        {"N": 50, "K": 4, "scenario": "S3", "area": 200},
        {"N": 50, "K": 6, "scenario": "S3", "area": 200}
    ]

    density_configs = [
        {"label": "Low", "threshold": 30},
        {"label": "Medium", "threshold": 50},
        {"label": "High", "threshold": 70}
    ]

    e2_fig_dir = config.FIGURES_DIR / "E2"
    e2_csv_dir = config.CSV_DIR / "E2"
    os.makedirs(e2_fig_dir, exist_ok=True)
    os.makedirs(e2_csv_dir, exist_ok=True)

    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}
    ]

    methods = {
        "Random": random_allocation,
        "Greedy": greedy_allocation,
        "DSATUR": dsatur_allocation,
        "BD-CeNN": bdcenn_allocation
    }

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e2_fig_dir / folder
        model_csv_dir = e2_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []

        for cfg in configs:
            N = cfg["N"]
            K = cfg["K"]
            sc_name = cfg["scenario"]
            inst_sc = sc_instances[sc_name]

            for dens in density_configs:
                th = dens["threshold"]
                label = dens["label"]
                M = create_channel_interference_matrix(K) if model_name == "adjacent" else None

                for seed in seeds:
                    inst = inst_sc.get(str(seed))
                    if inst is None:
                        continue
                    positions = np.array(inst["positions"])
                    W = np.zeros((N, N))
                    for i in range(N):
                        for j in range(i+1, N):
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

                    for method_name, method_func in methods.items():
                        start_time = time.perf_counter()
                        if method_name == "Random":
                            np.random.seed(seed)
                            x = method_func(N, K)
                        elif method_name == "Greedy":
                            np.random.seed(seed)
                            order = np.random.permutation(N).tolist()
                            x = method_func(N, K, W, order=order, M=M)
                        elif method_name == "DSATUR":
                            x = method_func(N, K, W, M=M)
                        else:
                            x, _, _, _, best_iter = method_func(
                                N, K, W, M=M,
                                num_restarts=config.NUM_RESTARTS,
                                max_iter=config.MAX_ITER_BD,
                                random_order=True,
                                seed=seed,
                                verbose=False
                            )
                        elapsed = time.perf_counter() - start_time

                        if M is None:
                            cost = compute_cochannel_cost(x, W)
                            conflicts = count_cochannel_conflicts(x, W)
                        else:
                            cost = compute_adjacent_cost(x, W, M)
                            conflicts = count_adjacent_conflicts(x, W, M)
                        used_channels = len(set(x))
                        iterations = best_iter if method_name == "BD-CeNN" else np.nan

                        raw_data.append({
                            "N": N,
                            "K": K,
                            "scenario": sc_name,
                            "density_label": label,
                            "threshold": th,
                            "seed": seed,
                            "method": method_name,
                            "cost": cost,
                            "conflicts": conflicts,
                            "used_channels": used_channels,
                            "time": elapsed,
                            "iterations": iterations
                        })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(model_csv_dir / "raw_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E2/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["N", "K", "density_label", "threshold", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"],
            "iterations": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        
        summary.columns = ['N', 'K', 'density_label', 'threshold', 'method',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
                           'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']

        for col in ['conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                    'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']:
            summary[col] = summary[col].round(1)

        for col in ['conflicts_mean', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_min', 'used_channels_max', 'used_channels_median']:
            summary[col] = summary[col].fillna(0).round(0).astype(int)

        summary.to_csv(model_csv_dir / "summary_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E2/{folder}/summary_metrics.csv saved.")

        # Condense comparison layouts into a single grouped figure grid
        metrics_to_plot = [
            ('cost', 'Mean Global Cost J(x)'),
            ('conflicts', 'Mean Conflicted Links'),
            ('used_channels', 'Mean Channels Used'),
            ('time', 'Execution Runtime (seconds)')
        ]

        subplots_config = [
            (30, 4, "N=30, K=4"),
            (30, 6, "N=30, K=6"),
            (50, 4, "N=50, K=4"),
            (50, 6, "N=50, K=6")
        ]

        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}

        for metric_col, ylabel in metrics_to_plot:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes_flat = axes.flatten()

            for ax, (N, K, title) in zip(axes_flat, subplots_config):
                sub = summary[(summary['N'] == N) & (summary['K'] == K)]
                if sub.empty:
                    ax.set_title(f"{title} (No Data)")
                    ax.axis('off')
                    continue

                density_order = ["Low", "Medium", "High"]
                pivot_mean = sub.pivot(index='density_label', columns='method', values=f'{metric_col}_mean').reindex(density_order)
                pivot_std = sub.pivot(index='density_label', columns='method', values=f'{metric_col}_std').reindex(density_order)

                x = np.arange(len(density_order))
                width = 0.2
                for i, method in enumerate(methods.keys()):
                    if method not in pivot_mean.columns:
                        continue
                    means = pivot_mean[method].values
                    stds = pivot_std[method].values
                    offset = (i - 1.5) * width
                    ax.bar(x + offset, means, width, yerr=stds,
                           capsize=3, label=method if ax == axes_flat[0] else "",
                           color=method_colors[method], alpha=0.8)

                ax.set_xticks(x)
                ax.set_xticklabels(density_order, fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=10)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                if metric_col == 'used_channels':
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(methods), fontsize=10, bbox_to_anchor=(0.5, -0.02))
            fig.suptitle(f"E2 - {ylabel} Evaluation - {model_name}", fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            fname = f"E2_{metric_col}_{model_name}.png"
            plt.savefig(model_fig_dir / fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"[INFO] Figure E2/{folder}/{fname} saved.")

    print("[SUCCESS] Experiment E2 completed.")


# ============================================================================
# E3 - SPECTRUM SCALING (K-SENSITIVITY)
# ============================================================================
def run_experiment_E3(verbose=False):
    """
    E3 - Sensitivity evaluation of K variations under tight bounds.
    """
    scenario_name = "S4"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"[ERROR] Scenario {scenario_name} missing from database.")
        return

    first_inst = list(instances.values())[0]
    N = first_inst["N"]
    K_values = [2, 3, 4, 6, 8]
    seeds = sorted([int(s) for s in instances.keys() if s.isdigit()])

    e3_fig_dir = config.FIGURES_DIR / "E3"
    e3_csv_dir = config.CSV_DIR / "E3"
    os.makedirs(e3_fig_dir, exist_ok=True)
    os.makedirs(e3_csv_dir, exist_ok=True)

    models = [
        {"name": "cochannel", "suffix": "_co", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "suffix": "_adj", "M": None, "folder": "adjacent"}
    ]

    methods = {
        "Random": random_allocation,
        "Greedy": greedy_allocation,
        "DSATUR": dsatur_allocation,
        "BD-CeNN": bdcenn_allocation
    }

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e3_fig_dir / folder
        model_csv_dir = e3_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []

        for K in K_values:
            M = create_channel_interference_matrix(K) if model_name == "adjacent" else None

            for seed in seeds:
                inst = instances.get(str(seed))
                if inst is None:
                    continue
                W = np.array(inst["W"])

                for method_name, method_func in methods.items():
                    if method_name == "Random":
                        np.random.seed(seed)
                        x = method_func(N, K)
                    elif method_name == "Greedy":
                        np.random.seed(seed)
                        order = np.random.permutation(N).tolist()
                        x = method_func(N, K, W, order=order, M=M)
                    elif method_name == "DSATUR":
                        x = method_func(N, K, W, M=M)
                    else:
                        x, _, _, _, _ = method_func(
                            N, K, W, M=M,
                            num_restarts=config.NUM_RESTARTS,
                            max_iter=config.MAX_ITER_BD,
                            random_order=True,
                            seed=seed,
                            verbose=False
                        )

                    if M is None:
                        cost = compute_cochannel_cost(x, W)
                        conflicts = count_cochannel_conflicts(x, W)
                    else:
                        cost = compute_adjacent_cost(x, W, M)
                        conflicts = count_adjacent_conflicts(x, W, M)
                    used_channels = len(set(x))

                    raw_data.append({
                        "K": K,
                        "seed": seed,
                        "method": method_name,
                        "cost": cost,
                        "conflicts": conflicts,
                        "used_channels": used_channels
                    })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(model_csv_dir / "raw_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E3/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["K", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        
        summary.columns = ['K', 'method', 
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median']
        
        for col in ['conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median']:
            summary[col] = summary[col].round(1)
            
        for col in ['conflicts_mean', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_min', 'used_channels_max', 'used_channels_median']:
            summary[col] = summary[col].round(0).astype(int)

        summary.to_csv(summary_csv_path := model_csv_dir / "summary_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E3/{folder}/summary_metrics.csv saved.")

        pivot_cost = summary.pivot(index='K', columns='method', values=['cost_mean', 'cost_std'])
        pivot_conflicts = summary.pivot(index='K', columns='method', values=['conflicts_mean', 'conflicts_std'])

        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
        method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}

        # Plot 1: Cost vs K
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for method in methods.keys():
            means = pivot_cost[('cost_mean', method)].values
            stds = pivot_cost[('cost_std', method)].values
            ax1.errorbar(K_values, means, yerr=stds, 
                         label=method, color=method_colors[method], marker=method_markers[method],
                         capsize=5, linewidth=2, markersize=8)

        ax1.set_xlabel("Number of Available Channels K", fontsize=11)
        ax1.set_ylabel("Mean Global Network Cost J(x)", fontsize=11)
        ax1.set_title(f"E3 - Cost Sensitivity to Spectrum Scale - {model_name}", fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_fig_dir / f"cost_vs_K_{model_name}.png", dpi=300)
        plt.close(fig1)

        # Plot 2: Conflicts vs K
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for method in methods.keys():
            means = pivot_conflicts[('conflicts_mean', method)].values
            stds = pivot_conflicts[('conflicts_std', method)].values
            ax2.errorbar(K_values, means, yerr=stds,
                         label=method, color=method_colors[method], marker=method_markers[method],
                         capsize=5, linewidth=2, markersize=8)

        ax2.set_xlabel("Number of Available Channels K", fontsize=11)
        ax2.set_ylabel("Mean Active Violations", fontsize=11)
        ax2.set_title(f"E3 - Active Violation Sensitivity to Spectrum Scale - {model_name}", fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_fig_dir / f"conflicts_vs_K_{model_name}.png", dpi=300)
        plt.close(fig2)
        print(f"[INFO] Performance charts saved inside E3/{folder}/.")

    print("[SUCCESS] Experiment E3 completed.")


# ============================================================================
# E4 - STRUCTURAL DENSITY SENSITIVITY
# ============================================================================
def run_experiment_E4(verbose=False):
    """
    E4 - Density threshold analysis over the cell structure.
    """
    scenario_name = "S3"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"[ERROR] Scenario {scenario_name} missing from database.")
        return

    first_inst = list(instances.values())[0]
    N = first_inst["N"]
    K = first_inst["K"]
    density_configs = [
        {"label": "Low", "threshold": 30},
        {"label": "Medium", "threshold": 50},
        {"label": "High", "threshold": 70}
    ]

    seeds = sorted([int(s) for s in instances.keys() if s.isdigit()])

    e4_fig_dir = config.FIGURES_DIR / "E4"
    e4_csv_dir = config.CSV_DIR / "E4"
    os.makedirs(e4_fig_dir, exist_ok=True)
    os.makedirs(e4_csv_dir, exist_ok=True)

    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}
    ]

    methods = {
        "Random": random_allocation,
        "Greedy": greedy_allocation,
        "DSATUR": dsatur_allocation,
        "BD-CeNN": bdcenn_allocation
    }

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e4_fig_dir / folder
        model_csv_dir = e4_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []

        for dens in density_configs:
            th = dens["threshold"]
            label = dens["label"]

            for seed in seeds:
                inst = instances.get(str(seed))
                if inst is None:
                    continue
                positions = np.array(inst["positions"])
                W = np.zeros((N, N))
                for i in range(N):
                    for j in range(i+1, N):
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

                M = create_channel_interference_matrix(K) if model_name == "adjacent" else None

                for method_name, method_func in methods.items():
                    start_time = time.perf_counter()
                    if method_name == "Random":
                        np.random.seed(seed)
                        x = method_func(N, K)
                    elif method_name == "Greedy":
                        np.random.seed(seed)
                        order = np.random.permutation(N).tolist()
                        x = method_func(N, K, W, order=order, M=M)
                    elif method_name == "DSATUR":
                        x = method_func(N, K, W, M=M)
                    else:
                        x, _, _, _, best_iter = method_func(
                            N, K, W, M=M,
                            num_restarts=config.NUM_RESTARTS,
                            max_iter=config.MAX_ITER_BD,
                            random_order=True,
                            seed=seed,
                            verbose=False
                        )
                    elapsed = time.perf_counter() - start_time

                    if M is None:
                        cost = compute_cochannel_cost(x, W)
                        conflicts = count_cochannel_conflicts(x, W)
                    else:
                        cost = compute_adjacent_cost(x, W, M)
                        conflicts = count_adjacent_conflicts(x, W, M)
                    used_channels = len(set(x))
                    iterations = best_iter if method_name == "BD-CeNN" else np.nan

                    raw_data.append({
                        "density_label": label,
                        "threshold": th,
                        "seed": seed,
                        "method": method_name,
                        "cost": cost,
                        "conflicts": conflicts,
                        "used_channels": used_channels,
                        "time": elapsed,
                        "iterations": iterations
                    })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(model_csv_dir / "raw_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E4/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["density_label", "threshold", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"],
            "iterations": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        
        summary.columns = ['density_label', 'threshold', 'method',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
                           'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']

        for col in ['conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median']:
            summary[col] = summary[col].round(1)
            if 'std' not in col:
                summary[col] = summary[col].round(0).astype(int)

        for col in ['iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']:
            summary[col] = summary[col].round(1)

        summary.to_csv(model_csv_dir / "summary_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E4/{folder}/summary_metrics.csv saved.")

        pivot_cost = summary.pivot(index='threshold', columns='method', values=['cost_mean', 'cost_std'])
        x_vals = sorted(summary['threshold'].unique())
        density_labels = [d["label"] for d in density_configs]

        fig, ax = plt.subplots(figsize=(10, 6))
        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
        method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}

        for method in methods.keys():
            means = pivot_cost[('cost_mean', method)].values
            stds = pivot_cost[('cost_std', method)].values
            ax.errorbar(x_vals, means, yerr=stds,
                         label=method, color=method_colors[method], marker=method_markers[method],
                         capsize=5, linewidth=2, markersize=8)

        ax.set_xlabel("Interference Threshold (Spatial Node Density Scale)", fontsize=11)
        ax.set_ylabel("Mean Global Cost J(x)", fontsize=11)
        ax.set_title(f"E4 - Penalty Distribution vs Spatial Link Density - {model_name}", fontsize=12, fontweight='bold')
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{th}" for th in x_vals])
        
        ax2 = ax.twiny()
        ax2.set_xticks(x_vals)
        ax2.set_xticklabels(density_labels)
        ax2.set_xlabel("Structural Node Coupling Degree")
        ax2.xaxis.set_label_position('bottom')
        ax2.xaxis.tick_bottom()
        ax2.xaxis.set_ticks_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 40))

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_fig_dir / f"cost_vs_density_{model_name}.png", dpi=300)
        plt.close(fig)
        print(f"[INFO] Figure E4/{folder}/cost_vs_density_{model_name}.png saved.")

    print("[SUCCESS] Experiment E4 completed.")


# ============================================================================
# E6 - COMPUTATIONAL SCALE PROPERTIES (N-SENSITIVITY)
# ============================================================================
def run_experiment_E6(verbose=False):
    """
    E6 - Complexity growth metrics across node scales N.
    """
    K = 8
    area = 300
    threshold = 50
    N_values = [20, 30, 50, 100, 200]
    seeds = list(range(1, config.NUM_RUNS + 1))

    e6_fig_dir = config.FIGURES_DIR / "E6"
    e6_csv_dir = config.CSV_DIR / "E6"
    os.makedirs(e6_fig_dir, exist_ok=True)
    os.makedirs(e6_csv_dir, exist_ok=True)

    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}
    ]

    methods = {
        "Random": random_allocation,
        "Greedy": greedy_allocation,
        "DSATUR": dsatur_allocation,
        "BD-CeNN": bdcenn_allocation
    }

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e6_fig_dir / folder
        model_csv_dir = e6_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []

        for N in N_values:
            M = create_channel_interference_matrix(K) if model_name == "adjacent" else None

            for seed in seeds:
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
                sum_weights = np.sum(W)

                for method_name, method_func in methods.items():
                    start_time = time.perf_counter()
                    if method_name == "Random":
                        np.random.seed(seed)
                        x = method_func(N, K)
                    elif method_name == "Greedy":
                        np.random.seed(seed)
                        order = np.random.permutation(N).tolist()
                        x = method_func(N, K, W, order=order, M=M)
                    elif method_name == "DSATUR":
                        x = method_func(N, K, W, M=M)
                    else:
                        x, _, _, _, best_iter = method_func(
                            N, K, W, M=M,
                            num_restarts=config.NUM_RESTARTS,
                            max_iter=config.MAX_ITER_BD,
                            random_order=True,
                            seed=seed,
                            verbose=False
                        )
                    elapsed = time.perf_counter() - start_time

                    if M is None:
                        cost = compute_cochannel_cost(x, W)
                        conflicts = count_cochannel_conflicts(x, W)
                    else:
                        cost = compute_adjacent_cost(x, W, M)
                        conflicts = count_adjacent_conflicts(x, W, M)
                    used_channels = len(set(x))
                    normalized_cost = cost / sum_weights if sum_weights > 0 else np.nan
                    iterations = best_iter if method_name == "BD-CeNN" else np.nan

                    raw_data.append({
                        "N": N,
                        "seed": seed,
                        "method": method_name,
                        "cost": cost,
                        "normalized_cost": normalized_cost,
                        "conflicts": conflicts,
                        "used_channels": used_channels,
                        "time": elapsed,
                        "iterations": iterations
                    })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(model_csv_dir / "raw_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E6/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["N", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "normalized_cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"],
            "iterations": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        
        summary.columns = ['N', 'method',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'normalized_cost_mean', 'normalized_cost_std', 'normalized_cost_min', 'normalized_cost_max', 'normalized_cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
                           'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']

        for col in ['conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                    'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']:
            summary[col] = summary[col].round(1)

        for col in ['conflicts_mean', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                    'iterations_mean', 'iterations_min', 'iterations_max', 'iterations_median']:
            summary[col] = summary[col].fillna(0).round(0).astype(int)

        summary.to_csv(model_csv_dir / "summary_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E6/{folder}/summary_metrics.csv saved.")

        metrics_to_plot = [
            ('cost_mean', 'cost_std', 'Mean Global Penalty J(x)'),
            ('time_mean', 'time_std', 'Computational Runtime (seconds)'),
            ('iterations_mean', 'iterations_std', 'Active Settling Iterations (BD-CeNN)')
        ]

        for metric_col, std_col, ylabel in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            if 'iterations' in metric_col:
                sub = summary[summary['method'] == 'BD-CeNN'].set_index('N').reindex(N_values).reset_index()
                ax.errorbar(sub['N'].values, sub[metric_col].values, yerr=sub[std_col].values, fmt='o-', capsize=5,
                            color='firebrick', label='BD-CeNN', linewidth=2, markersize=8)
                ax.set_title(f"E6 - Settling Iterations Growth Profile - {ylabel}")
            else:
                method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
                method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}
                for method in methods.keys():
                    sub = summary[summary['method'] == method].set_index('N').reindex(N_values).reset_index()
                    ax.errorbar(sub['N'].values, sub[metric_col].values, yerr=sub[std_col].values, fmt=method_markers[method]+'-', capsize=5,
                                color=method_colors[method], label=method, linewidth=2, markersize=8)
                ax.set_title(f"E6 - Complexity growth metrics - {ylabel}")

            ax.set_xlabel("Number of Cell Antennas N", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            fname = f"{metric_col.replace('_mean','')}_vs_N_{model_name}.png"
            plt.savefig(model_fig_dir / fname, dpi=300)
            plt.close(fig)

        # Plot 4: Normalized Cost vs N
        fig, ax = plt.subplots(figsize=(10, 6))
        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
        method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}
        for method in methods.keys():
            sub = summary[summary['method'] == method].set_index('N').reindex(N_values).reset_index()
            ax.errorbar(sub['N'].values, sub['normalized_cost_mean'].values, yerr=sub['normalized_cost_std'].values, fmt=method_markers[method]+'-', capsize=5,
                        color=method_colors[method], label=method, linewidth=2, markersize=8)
        ax.set_xlabel("Number of Cells N", fontsize=11)
        ax.set_ylabel("Normalized Mean Cost Profile", fontsize=11)
        ax.set_title(f"E6 - Normalized Structural Loss vs Scaling Scale - {model_name}", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_fig_dir / f"normalized_cost_vs_N_{model_name}.png", dpi=300)
        plt.close(fig)
        print(f"[INFO] Operational charts generated inside E6/{folder}/.")

    print("[SUCCESS] Experiment E6 completed.")


# ============================================================================
# E7 - NOISE ROBUSTNESS
# ============================================================================
def run_experiment_E7(verbose=False):
    """
    E7 - Noise insertion robust analysis under the measurement matrix W.
    """
    scenario_name = "S6"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"[ERROR] Scenario {scenario_name} missing from database.")
        return

    first_inst = list(instances.values())[0]
    N = first_inst["N"]
    K = first_inst["K"]
    seeds = sorted([int(s) for s in instances.keys() if s.isdigit()])

    noise_levels = [0.05, 0.10, 0.20]
    noise_labels = ["5%", "10%", "20%"]

    e7_fig_dir = config.FIGURES_DIR / "E7"
    e7_csv_dir = config.CSV_DIR / "E7"
    os.makedirs(e7_fig_dir, exist_ok=True)
    os.makedirs(e7_csv_dir, exist_ok=True)

    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}
    ]

    method_func = bdcenn_allocation

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e7_fig_dir / folder
        model_csv_dir = e7_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []
        ref_x = {}
        ref_cost = {}
        ref_W = {}

        for seed in seeds:
            inst = instances.get(str(seed))
            if inst is None:
                continue
            positions = np.array(inst["positions"])
            W_clean = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    threshold = 35
                    if dist < threshold * 0.4:
                        w = 4
                    elif dist < threshold * 0.65:
                        w = 2
                    elif dist < threshold:
                        w = 1
                    else:
                        w = 0
                    W_clean[i, j] = w
                    W_clean[j, i] = w
            ref_W[seed] = W_clean

            M = create_channel_interference_matrix(K) if model_name == "adjacent" else None

            x_ref, _, _, _, _ = method_func(
                N, K, W_clean, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=seed,
                verbose=False
            )
            ref_x[seed] = x_ref
            ref_cost[seed] = compute_adjacent_cost(x_ref, W_clean, M) if M is not None else compute_cochannel_cost(x_ref, W_clean)

        for b_noise, noise_label in zip(noise_levels, noise_labels):
            for seed in seeds:
                W_clean = ref_W[seed]
                np.random.seed(seed + int(b_noise * 10000))
                W_noisy = W_clean.copy()
                mask = W_clean > 0
                noise_factor = np.random.uniform(-b_noise, b_noise, size=W_clean.shape)
                W_noisy[mask] = W_clean[mask] * (1 + noise_factor[mask])
                W_noisy = np.round(W_noisy)
                W_noisy[W_noisy < 0] = 0
                for i in range(N):
                    for j in range(i+1, N):
                        W_noisy[j, i] = W_noisy[i, j]
                np.fill_diagonal(W_noisy, 0)

                M = create_channel_interference_matrix(K) if model_name == "adjacent" else None
                seed_noisy = seed + int(b_noise * 1000) + 100
                x_noisy, _, _, _, _ = method_func(
                    N, K, W_noisy, M=M,
                    num_restarts=config.NUM_RESTARTS,
                    max_iter=config.MAX_ITER_BD,
                    random_order=True,
                    seed=seed_noisy,
                    verbose=False
                )
                cost_noisy = compute_adjacent_cost(x_noisy, W_noisy, M) if M is not None else compute_cochannel_cost(x_noisy, W_noisy)

                cost_relatif = ((cost_noisy - ref_cost[seed]) / ref_cost[seed]) * 100 if ref_cost[seed] > 0 else 0.0
                changes = np.sum(x_noisy != ref_x[seed])
                change_rate = (changes / N) * 100

                raw_data.append({
                    "noise_level": b_noise * 100,
                    "seed": seed,
                    "cost_ref": ref_cost[seed],
                    "cost_noisy": cost_noisy,
                    "cost_relatif": cost_relatif,
                    "change_rate": change_rate
                })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(model_csv_dir / "raw_metrics.csv", index=False, float_format='%.6f')
        print(f"[SUCCESS] CSV E7/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby("noise_level").agg({
            "cost_relatif": ["mean", "std", "min", "max", "median"],
            "change_rate": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        summary.columns = ['noise_level',
                           'cost_relatif_mean', 'cost_relatif_std', 'cost_relatif_min', 'cost_relatif_max', 'cost_relatif_median',
                           'change_rate_mean', 'change_rate_std', 'change_rate_min', 'change_rate_max', 'change_rate_median']
        for col in summary.columns:
            if col != 'noise_level':
                summary[col] = summary[col].round(1)

        summary.to_csv(model_csv_dir / "summary_metrics.csv", index=False, float_format='%.1f')
        print(f"[SUCCESS] CSV E7/{folder}/summary_metrics.csv saved.")

        fig, ax = plt.subplots(figsize=(10, 6))
        noise_vals = [0] + sorted(summary['noise_level'].unique())
        cost_means = [0] + summary['cost_relatif_mean'].tolist()
        cost_stds = [0] + summary['cost_relatif_std'].tolist()
        change_means = [0] + summary['change_rate_mean'].tolist()
        change_stds = [0] + summary['change_rate_std'].tolist()

        ax.errorbar(noise_vals, cost_means, yerr=cost_stds, fmt='o-', capsize=6,
                    color='red', label='Mean Relative Cost Shift (%)', linewidth=2, markersize=8)
        ax.errorbar(noise_vals, change_means, yerr=change_stds, fmt='s-', capsize=6,
                    color='blue', label='Node Re-assignment Rate (%)', linewidth=2, markersize=8)

        ax.set_xlabel("Injected Measurement Noise Level (%)", fontsize=11)
        ax.set_ylabel("Deviation Percentage (%)", fontsize=11)
        ax.set_title(f"E7 - Solver Robustness under Matrix Perturbations - {model_name}", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(-50, 100)

        plt.tight_layout()
        fname = f"robustness_{model_name}.png"
        plt.savefig(model_fig_dir / fname, dpi=300)
        plt.close(fig)
        print(f"[INFO] Figure E7/{folder}/{fname} saved.")

    print("[SUCCESS] Experiment E7 completed.")


# ============================================================================
# E8 - DYNAMIC TOPOLOGY ADAPTATION (WARM VS COLD)
# ============================================================================
def run_experiment_E8(verbose=False):
    """
    E8 - Comparative convergence pathways of Warm vs Cold starts.
    """
    scenario_name = "S7"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"[ERROR] Scenario {scenario_name} missing from database.")
        return

    first_inst = list(instances.values())[0]
    N = first_inst["N"]
    K = first_inst["K"]
    threshold = 30
    seeds = sorted([int(s) for s in instances.keys() if s.isdigit()])

    mod_levels = [0.05, 0.10, 0.20]
    mod_labels = ["5%", "10%", "20%"]

    e8_fig_dir = config.FIGURES_DIR / "E8"
    e8_csv_dir = config.CSV_DIR / "E8"
    os.makedirs(e8_fig_dir, exist_ok=True)
    os.makedirs(e8_csv_dir, exist_ok=True)

    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}
    ]

    method_func = bdcenn_allocation

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e8_fig_dir / folder
        model_csv_dir = e8_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []
        ref_x = {}
        ref_W = {}

        for seed in seeds:
            inst = instances.get(str(seed))
            if inst is None:
                continue
            positions = np.array(inst["positions"])
            W_orig = np.zeros((N, N))
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
                    W_orig[i, j] = w
                    W_orig[j, i] = w
            ref_W[seed] = W_orig

            M = create_channel_interference_matrix(K) if model_name == "adjacent" else None

            x_init, _, _, _, _ = method_func(
                N, K, W_orig, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=seed,
                verbose=False
            )
            ref_x[seed] = x_init

        for b_mod, mod_label in zip(mod_levels, mod_labels):
            for seed in seeds:
                W_orig = ref_W[seed]
                np.random.seed(seed + int(b_mod * 10000) + 200)
                W_dyn = W_orig.copy()
                edges = [(i, j) for i in range(N) for j in range(i+1, N) if W_orig[i, j] > 0]
                if len(edges) == 0:
                    continue
                num_mod = int(len(edges) * b_mod)
                indices = np.random.choice(len(edges), num_mod, replace=False)
                for idx in indices:
                    i, j = edges[idx]
                    if np.random.random() > 0.5:
                        W_dyn[i, j] = 0
                        W_dyn[j, i] = 0
                    else:
                        current = W_dyn[i, j]
                        delta = np.random.choice([-1, 1]) if current > 1 else 1
                        new_w = max(1, min(4, current + delta))
                        W_dyn[i, j] = new_w
                        W_dyn[j, i] = new_w
                for i in range(N):
                    for j in range(i+1, N):
                        W_dyn[j, i] = W_dyn[i, j]
                np.fill_diagonal(W_dyn, 0)

                M = create_channel_interference_matrix(K) if model_name == "adjacent" else None

                # Warm start run (uses x_init directly as anchor)
                start = time.perf_counter()
                x_warm, _, _, _, _ = method_func(
                    N, K, W_dyn, M=M,
                    num_restarts=1,
                    max_iter=config.MAX_ITER_BD,
                    random_order=True,
                    seed=seed + int(b_mod * 1000) + 300,
                    verbose=False
                )
                time_warm = time.perf_counter() - start

                # Cold start run
                start = time.perf_counter()
                x_cold, _, _, _, _ = method_func(
                    N, K, W_dyn, M=M,
                    num_restarts=config.NUM_RESTARTS,
                    max_iter=config.MAX_ITER_BD,
                    random_order=True,
                    seed=seed + int(b_mod * 1000) + 400,
                    verbose=False
                )
                time_cold = time.perf_counter() - start

                changes_warm = np.sum(x_warm != ref_x[seed])
                changes_cold = np.sum(x_cold != ref_x[seed])

                cost_warm = compute_adjacent_cost(x_warm, W_dyn, M) if M is not None else compute_cochannel_cost(x_warm, W_dyn)
                cost_cold = compute_adjacent_cost(x_cold, W_dyn, M) if M is not None else compute_cochannel_cost(x_cold, W_dyn)

                raw_data.append({"mod_level": b_mod * 100, "seed": seed, "mode": "warm", "time": time_warm, "changes": changes_warm, "cost": cost_warm})
                raw_data.append({"mod_level": b_mod * 100, "seed": seed, "mode": "cold", "time": time_cold, "changes": changes_cold, "cost": cost_cold})

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(model_csv_dir / "raw_metrics.csv", index=False, float_format='%.6f')
        print(f"[SUCCESS] CSV E8/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["mod_level", "mode"]).agg({
            "time": ["mean", "std", "min", "max", "median"],
            "changes": ["mean", "std", "min", "max", "median"],
            "cost": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        summary.columns = ['mod_level', 'mode',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
                           'changes_mean', 'changes_std', 'changes_min', 'changes_max', 'changes_median',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median']
        for col in summary.columns:
            if col not in ['mod_level', 'mode']:
                summary[col] = summary[col].round(1) if ('changes' in col or 'cost' in col) else summary[col].round(6)

        summary.to_csv(model_csv_dir / "summary_metrics.csv", index=False, float_format='%.6f')
        print(f"[SUCCESS] CSV E8/{folder}/summary_metrics.csv saved.")

        fig, ax = plt.subplots(figsize=(10, 6))
        warm_data = summary[summary['mode'] == 'warm']
        cold_data = summary[summary['mode'] == 'cold']

        mod_vals_warm = [0] + sorted(warm_data['mod_level'].unique())
        changes_warm_means = [0] + warm_data['changes_mean'].tolist()
        changes_warm_stds = [0] + warm_data['changes_std'].tolist()

        mod_vals_cold = [0] + sorted(cold_data['mod_level'].unique())
        changes_cold_means = [0] + cold_data['changes_mean'].tolist()
        changes_cold_stds = [0] + cold_data['changes_std'].tolist()

        ax.errorbar(mod_vals_warm, changes_warm_means, yerr=changes_warm_stds, fmt='o-', capsize=6,
                    color='green', label='Warm Start Optimization', linewidth=2, markersize=8)
        ax.errorbar(mod_vals_cold, changes_cold_means, yerr=changes_cold_stds, fmt='s-', capsize=6,
                    color='purple', label='Cold Start Optimization', linewidth=2, markersize=8)

        ax.set_xlabel("Dynamic Network Topology Perturbation Level (%)", fontsize=11)
        ax.set_ylabel("Mean Frequency Reallocations", fontsize=11)
        ax.set_title(f"E8 - Convergence Efficiency in Volatile Channels - {model_name}", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        y_min = min(changes_warm_means + changes_cold_means) - 2
        y_max = max(changes_warm_means + changes_cold_means) + 2
        ax.set_ylim([max(y_min, 0), y_max + 5])
        plt.tight_layout()
        fname = f"adaptation_{model_name}.png"
        plt.savefig(model_fig_dir / fname, dpi=300)
        plt.close(fig)
        print(f"[INFO] Figure E8/{folder}/{fname} saved.")

    print("[SUCCESS] Experiment E8 completed.")


# ============================================================================
# E9 - LOCAL MINIMA & MULTI-RESTART PROFILES
# ============================================================================
def run_experiment_E9(verbose=False):
    """
    E9 - Mitigation of local minima via multi-restart schedules.
    """
    scenario_name = "S3"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"[ERROR] Scenario {scenario_name} missing from database.")
        return

    first_inst = list(instances.values())[0]
    N = first_inst["N"]
    K = first_inst["K"]
    threshold = 60
    seeds = sorted([int(s) for s in instances.keys() if s.isdigit()])

    values = [1, 5, 10, 20]

    e9_fig_dir = config.FIGURES_DIR / "E9"
    e9_csv_dir = config.CSV_DIR / "E9"
    os.makedirs(e9_fig_dir, exist_ok=True)
    os.makedirs(e9_csv_dir, exist_ok=True)

    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}
    ]

    method_bd = bdcenn_allocation
    method_greedy = greedy_allocation

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e9_fig_dir / folder
        model_csv_dir = e9_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []
        topologies = {}
        for seed in seeds:
            inst = instances.get(str(seed))
            if inst is None:
                continue
            positions = np.array(inst["positions"])
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
            topologies[seed] = W

        for seed, W in topologies.items():
            M = create_channel_interference_matrix(K) if model_name == "adjacent" else None

            # 1. BD-CeNN Execution over Restart Schedule
            for num_restarts in values:
                start = time.perf_counter()
                x_bd, _, _, _, _ = method_bd(
                    N, K, W, M=M,
                    num_restarts=num_restarts,
                    max_iter=config.MAX_ITER_BD,
                    random_order=True,
                    seed=seed + num_restarts * 1000,
                    verbose=False
                )
                elapsed = time.perf_counter() - start

                cost = compute_adjacent_cost(x_bd, W, M) if M is not None else compute_cochannel_cost(x_bd, W)
                conflicts = count_adjacent_conflicts(x_bd, W, M) if M is not None else count_cochannel_conflicts(x_bd, W)

                raw_data.append({"seed": seed, "method": "BD-CeNN", "value": num_restarts, "cost": cost, "conflicts": conflicts, "time": elapsed})

            # 2. Greedy Exploration across multiple random sweeps
            for num_orders in values:
                best_cost = float('inf')
                best_conflicts = None
                best_time = 0.0

                start_total = time.perf_counter()
                for order_idx in range(num_orders):
                    np.random.seed(seed + order_idx * 100 + num_orders * 1000)
                    order = np.random.permutation(N).tolist()
                    start = time.perf_counter()
                    x_g = method_greedy(N, K, W, order=order, M=M)
                    best_time += (time.perf_counter() - start)

                    cost_g = compute_adjacent_cost(x_g, W, M) if M is not None else compute_cochannel_cost(x_g, W)
                    conflicts_g = count_adjacent_conflicts(x_g, W, M) if M is not None else count_cochannel_conflicts(x_g, W)

                    if cost_g < best_cost:
                        best_cost = cost_g
                        best_conflicts = conflicts_g

                raw_data.append({"seed": seed, "method": "Greedy", "value": num_orders, "cost": best_cost, "conflicts": best_conflicts, "time": time.perf_counter() - start_total})

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(model_csv_dir / "raw_metrics.csv", index=False)
        print(f"[SUCCESS] CSV E9/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["value", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        summary.columns = ['value', 'method',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median']
        for col in summary.columns:
            if col not in ['value', 'method']:
                summary[col] = summary[col].round(1) if ('cost' in col or 'conflicts' in col) else summary[col].round(6)

        summary.to_csv(model_csv_dir / "summary_metrics.csv", index=False, float_format='%.6f')
        print(f"[SUCCESS] CSV E9/{folder}/summary_metrics.csv saved.")

        fig, ax = plt.subplots(figsize=(10, 6))
        bd_data = summary[summary['method'] == 'BD-CeNN'].sort_values('value')
        greedy_data = summary[summary['method'] == 'Greedy'].sort_values('value')

        x_vals = values
        ax.errorbar(x_vals, bd_data['cost_mean'], yerr=bd_data['cost_std'], fmt='o-', capsize=6, color='red', label='BD-CeNN Solver', linewidth=2, markersize=8)
        ax.errorbar(x_vals, greedy_data['cost_mean'], yerr=greedy_data['cost_std'], fmt='s-', capsize=6, color='blue', label='Greedy Randomized Sweeps', linewidth=2, markersize=8)

        ax.set_xlabel("Re-evaluations Scale (Restarts for CeNN, Trial Counts for Greedy)", fontsize=11)
        ax.set_ylabel("Mean Settled Energy Cost J(x)", fontsize=11)
        ax.set_title(f"E9 - Mitigation of Local Minima via Multi-restart Schedules - {model_name}", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks(values)
        ax.set_xticklabels([str(v) for v in values])

        plt.tight_layout()
        fname = f"minima_locaux_{model_name}.png"
        plt.savefig(model_fig_dir / fname, dpi=300)
        plt.close(fig)
        print(f"[INFO] Figure E9/{folder}/{fname} saved.")

    print("[SUCCESS] Experiment E9 completed.")


# ============================================================================
# E10 - NEURO-SYMBOLIC LLM AUDITING & ALIGNMENT
# ============================================================================
def run_experiment_E10(verbose=False):
    """
    E10 - Auditing of qualitative LLM text summaries vs certified database metrics.
    """
    e10_csv_dir = config.CSV_DIR / "E10"
    os.makedirs(e10_csv_dir, exist_ok=True)

    cases_def = [
        {"case_id": "#1",  "scenario": "S1", "N": 8,   "K": 3, "seed": 1, "context": "Visual verification simple layout"},
        {"case_id": "#2",  "scenario": "S1", "N": 8,   "K": 3, "seed": 2, "context": "Secondary baseline layout"},
        {"case_id": "#3",  "scenario": "S2", "N": 30,  "K": 4, "seed": 1, "context": "Nominal standard load"},
        {"case_id": "#4",  "scenario": "S2", "N": 30,  "K": 4, "seed": 2, "context": "Medium network load profile"},
        {"case_id": "#5",  "scenario": "S2", "N": 30,  "K": 6, "seed": 1, "context": "Abundant spectral channels"},
        {"case_id": "#6",  "scenario": "S3", "N": 50,  "K": 6, "seed": 1, "context": "Highly congested cell environment"},
        {"case_id": "#7",  "scenario": "S3", "N": 50,  "K": 6, "seed": 2, "context": "Secondary dense topology layout"},
        {"case_id": "#8",  "scenario": "S3", "N": 50,  "K": 6, "seed": 3, "context": "Dense congested topology variant"},
        {"case_id": "#9",  "scenario": "S4", "N": 50,  "K": 2, "seed": 1, "context": "Severe spectral bottleneck"},
        {"case_id": "#10", "scenario": "S4", "N": 50,  "K": 2, "seed": 2, "context": "Secondary severe bottleneck"},
        {"case_id": "#11", "scenario": "S4", "N": 50,  "K": 3, "seed": 1, "context": "Moderate constraint layout"},
        {"case_id": "#12", "scenario": "S5", "N": 100, "K": 8, "seed": 1, "context": "Large standard cell layout"},
        {"case_id": "#13", "scenario": "S5", "N": 200, "K": 8, "seed": 1, "context": "Extreme scale validation"},
        {"case_id": "#14", "scenario": "S6", "N": 50,  "K": 4, "seed": 1, "context": "Measurement noise variance level 5%"},
        {"case_id": "#15", "scenario": "S6", "N": 50,  "K": 4, "seed": 1, "context": "Measurement noise variance level 10%"},
        {"case_id": "#16", "scenario": "S6", "N": 50,  "K": 4, "seed": 1, "context": "Extreme measurement variance 20%"},
        {"case_id": "#17", "scenario": "S7", "N": 45,  "K": 5, "seed": 1, "context": "Low dynamic perturbation warm run"},
        {"case_id": "#18", "scenario": "S7", "N": 45,  "K": 5, "seed": 1, "context": "Moderate dynamic perturbation run"},
        {"case_id": "#19", "scenario": "S7", "N": 45,  "K": 5, "seed": 1, "context": "High dynamic perturbation run"},
        {"case_id": "#20", "scenario": "S4", "N": 50,  "K": 2, "seed": 1, "context": "Stagnant local minimum (no restarts)"},
    ]

    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"},
    ]

    all_results = {}

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_csv_dir = e10_csv_dir / folder
        os.makedirs(model_csv_dir, exist_ok=True)

        results_for_model = []

        for case in cases_def:
            print(f"[AUDIT] Running LLM analysis on case {case['case_id']} under model {model_name}...")

            sc_name = case["scenario"]
            N = case["N"]
            K = case["K"]
            seed = case["seed"]

            instances = all_instances.get(sc_name, {})
            if not instances:
                print(f"[ERROR] Scenario {sc_name} not found.")
                continue

            inst = None
            if str(seed) not in instances:
                np.random.seed(seed)
                area = 300
                threshold = 50
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
            else:
                inst = instances[str(seed)]
                positions = np.array(inst["positions"])
                W = np.array(inst["W"])

            if W.shape[0] != N:
                np.random.seed(seed)
                area = 300 if N > 50 else 200
                threshold = 50
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

            bruit_level = 0.0
            mod_level = 0.0
            if sc_name == "S6":
                if "5%" in case["context"]:
                    bruit_level = 0.05
                elif "10%" in case["context"]:
                    bruit_level = 0.10
                elif "20%" in case["context"]:
                    bruit_level = 0.20
                np.random.seed(seed + int(bruit_level * 10000))
                mask = W > 0
                noise_factor = np.random.uniform(-bruit_level, bruit_level, size=W.shape)
                W = W * (1 + noise_factor * mask)
                W = np.round(W)
                W[W < 0] = 0
                for i in range(N):
                    for j in range(i+1, N):
                        W[j, i] = W[i, j]
                np.fill_diagonal(W, 0)
            elif sc_name == "S7":
                if "5%" in case["context"]:
                    mod_level = 0.05
                elif "10%" in case["context"]:
                    mod_level = 0.10
                elif "20%" in case["context"]:
                    mod_level = 0.20
                np.random.seed(seed + int(mod_level * 10000) + 200)
                edges = [(i, j) for i in range(N) for j in range(i+1, N) if W[i, j] > 0]
                if edges:
                    num_mod = int(len(edges) * mod_level)
                    indices = np.random.choice(len(edges), min(num_mod, len(edges)), replace=False)
                    for idx in indices:
                        i, j = edges[idx]
                        if np.random.random() > 0.5:
                            W[i, j] = 0
                            W[j, i] = 0
                        else:
                            current = W[i, j]
                            new_w = max(1, min(4, current + np.random.choice([-1, 1]) * min(current, 1)))
                            W[i, j] = new_w
                            W[j, i] = new_w

            M = create_channel_interference_matrix(K) if model_name == "adjacent" else None
            num_restarts = 1 if case["case_id"] == "#20" else config.NUM_RESTARTS

            start_t = time.perf_counter()
            x_bd, history_bd, _, _, best_iter = bdcenn_allocation(
                N, K, W, M=M,
                num_restarts=num_restarts,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=seed,
                verbose=False,
            )
            time_bd = time.perf_counter() - start_t

            if history_bd:
                alloc_init = history_bd[0][2]
                cost_init = compute_adjacent_cost(alloc_init, W, M) if M is not None else compute_cochannel_cost(alloc_init, W)
            else:
                cost_init = 0.0

            if M is None:
                cost_final = compute_cochannel_cost(x_bd, W)
                conflicts_final = count_cochannel_conflicts(x_bd, W)
            else:
                cost_final = compute_adjacent_cost(x_bd, W, M)
                conflicts_final = count_adjacent_conflicts(x_bd, W, M)

            used_channels = len(set(x_bd))

            np.random.seed(seed)
            x_rand = random_allocation(N, K)
            np.random.seed(seed)
            order_greedy = np.random.permutation(N).tolist()
            x_greedy = greedy_allocation(N, K, W, order=order_greedy, M=M)
            x_dsatur = dsatur_allocation(N, K, W, M=M)

            baselines = {}
            for name, x_b, t_b in [("Random", x_rand, 0.0), ("Greedy", x_greedy, 0.0), ("DSATUR", x_dsatur, 0.0)]:
                c = compute_adjacent_cost(x_b, W, M) if M is not None else compute_cochannel_cost(x_b, W)
                cf = count_adjacent_conflicts(x_b, W, M) if M is not None else count_cochannel_conflicts(x_b, W)
                baselines[name] = {"cost": float(c), "conflicts": int(cf), "time": float(t_b)}

            conflicting_cells = []
            for i in range(N):
                for j in range(i+1, N):
                    if W[i, j] > 0:
                        if M is None and x_bd[i] == x_bd[j]:
                            conflicting_cells.append(i)
                            conflicting_cells.append(j)
                        elif M is not None and M[x_bd[i], x_bd[j]] > 0:
                            conflicting_cells.append(i)
                            conflicting_cells.append(j)
            conflicting_cells = sorted(set(conflicting_cells))

            allocations_dict = {
                "Random": [int(c) for c in x_rand],
                "Greedy": [int(c) for c in x_greedy],
                "DSATUR": [int(c) for c in x_dsatur],
                "BD-CeNN": [int(c) for c in x_bd],
            }

            cell_positions = {int(i): [float(positions[i][0]), float(positions[i][1])] for i in range(N)}

            topology_edges_all = []
            for i in range(N):
                for j in range(i+1, N):
                    if W[i, j] > 0:
                        topology_edges_all.append((int(i), int(j), int(W[i, j])))
            topology_edges_all.sort(key=lambda e: -e[2])
            total_edges = len(topology_edges_all)
            topology_edges = topology_edges_all[:80]

            cell_degree = np.zeros(N, dtype=int)
            cell_strength = np.zeros(N, dtype=int)
            for (i, j, w) in topology_edges_all:
                cell_degree[i] += 1
                cell_degree[j] += 1
                cell_strength[i] += w
                cell_strength[j] += w

            top_cells_idx = np.argsort(-cell_strength)[:10]
            cell_summary = [{"cell": int(idx), "degree": int(cell_degree[idx]), "strength": int(cell_strength[idx])} for idx in top_cells_idx]

            threshold_value = float(inst["threshold"]) if (inst is not None and "threshold" in inst) else 0.0
            topology_meta = {"seed": int(seed), "threshold": threshold_value, "N": int(N), "K": int(K)}

            model_label = "CCI-only" if model_name == "cochannel" else "CCI+ACI"

            case_data = {
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "N": N,
                "K": K,
                "seed": seed,
                "context": case["context"] + f" [{model_label}]",
                "model_label": model_label,
                "metrics": {
                    "cost_initial": float(cost_init),
                    "cost_final": float(cost_final),
                    "conflicts": int(conflicts_final),
                    "time_seconds": float(time_bd),
                    "iterations": int(best_iter),
                    "used_channels": int(used_channels),
                },
                "baselines": baselines,
                "conflicting_cells": conflicting_cells,
                "allocations": allocations_dict,
                "cell_positions": cell_positions,
                "topology_edges": topology_edges,
                "total_edges": int(total_edges),
                "cell_summary": cell_summary,
                "topology_meta": topology_meta,
            }

            try:
                audit_result = audit_case(case_data, max_correction_attempts=2, model_folder=model_name)
                results_for_model.append(audit_result)
            except Exception as e:
                print(f"[ERROR] Failed to audit case {case['case_id']} via LLM. Cause: {e}")
                continue

        all_results[model_name] = results_for_model

        rows = []
        for r in results_for_model:
            status = r.get("status", "OK")
            if status == "LLM_FAILED":
                rows.append({
                    "Case": r["case_id"],
                    "Scenario (N, K, seed)": f"{r['scenario']} (N={r['N']}, K={r['K']}, seed={r['seed']})",
                    "Context": r["context"],
                    "Status": "LLM_FAILED",
                    "Hallucination Detected": "N/A",
                    "Correction / Error Type": "LLM Failure (Status 503) - Skip",
                    "Initial Accuracy Rate": 0.0,
                    "Final Accuracy Rate": 0.0,
                })
            else:
                inv_detected = "Yes" if r["verification_initial"]["has_hallucination"] else "No"
                if r["verification_initial"]["has_hallucination"]:
                    invented_vals = ", ".join(raw for _, raw in r["verification_initial"]["invented_numbers"])
                    type_err = f"Hallucination: {invented_vals}"
                    if r["verification_final"]["has_hallucination"]:
                        type_err += " (uncorrected)"
                    else:
                        type_err += f" (corrected in {r['correction_attempts']} attempt(s))"
                else:
                    type_err = "None"
                rows.append({
                    "Case": r["case_id"],
                    "Scenario (N, K, seed)": f"{r['scenario']} (N={r['N']}, K={r['K']}, seed={r['seed']})",
                    "Context": r["context"],
                    "Status": "OK",
                    "Hallucination Detected": inv_detected,
                    "Correction / Error Type": type_err,
                    "Initial Accuracy Rate": r["verification_initial"]["accuracy_rate"],
                    "Final Accuracy Rate": r["verification_final"]["accuracy_rate"],
                })
        df_eval = pd.DataFrame(rows)
        df_eval.to_csv(model_csv_dir / "llm_fidelity_evaluation.csv", index=False, float_format="%.2f")
        print(f"[SUCCESS] CSV E10/{folder}/llm_fidelity_evaluation.csv saved.")

    print("\n" + "=" * 80)
    print("[SUCCESS] GLOBAL EXPERIMENT E10 COMPLETED.")
    print("=" * 80)


# ============================================================================
# MASTER ORCHESTRATOR
# ============================================================================
def run_all_experiments(verbose=False):
    """
    Sequentially launches all project evaluation procedures.
    """
    print("\n[START] Executing standard analysis pipeline...")
    run_experiment_E1(verbose)
    run_experiment_E2(verbose)
    run_experiment_E3(verbose)
    run_experiment_E4(verbose)
    run_experiment_E6(verbose)
    run_experiment_E7(verbose)
    run_experiment_E8(verbose)
    run_experiment_E9(verbose)
    run_experiment_E10(verbose)
    print("\n[FINISH] All execution pipelines fully completed.")

if __name__ == "__main__":
    run_all_experiments()