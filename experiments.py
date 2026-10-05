# experiments.py

"""
Comprehensive evaluation framework executing academic experiments E1 to E10.
Produces comparative tables, charts, convergence curves, and audits LLM correctness.

Note de robustesse : toutes les figures utilisent le pattern [FIX-LAYOUT]
base sur Figure() + FigureCanvasAgg() + set_layout_engine('none') au lieu
de plt.figure(). Cela court-circuite totalement le moteur de layout pyplot
(constrained/tight/autolayout) et immunise les figures contre toute
pollution d'etat pyplot entre deux runs.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration matplotlib robuste (defense en profondeur) ---
plt.rcParams['figure.constrained_layout.use'] = False
plt.rcParams['figure.autolayout'] = False

# [FIX-LAYOUT] Imports pour creer des figures independantes de pyplot.
# Ces deux classes permettent de sortir du registre global pyplot (Gcf),
# ce qui rend les figures insensibles aux moteurs de layout automatiques.
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator

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
    compute_cost_cci,
    compute_cost_cci_aci,
    count_conflicts_cci,
    count_conflicts_aci,
    count_conflicts_total
)
from llm_assistant import audit_case

# --- Database Scenario Initialization ---
try:
    with open(config.SCENARIOS_FILE, "r", encoding="utf-8") as f:
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
# HELPER FUNCTIONS FOR INDEPENDENT METRICS & VALIDATION
# ============================================================================

def _new_figure(figsize):
    """
    [FIX-LAYOUT] Helper : cree une figure matplotlib totalement independante
    de pyplot et de tout moteur de layout automatique.

    Retourne (fig, canvas) : le canvas est necessaire pour que fig.savefig()
    fonctionne hors du registre pyplot (Gcf).
    """
    fig = Figure(figsize=figsize)
    canvas = FigureCanvasAgg(fig)
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass
    return fig, canvas


def _compute_all_conflicts(x, W, M=None):
    n_cci = count_conflicts_cci(x, W)
    n_aci = count_conflicts_aci(x, W, cutoff=2) if M is not None else 0
    return n_cci, n_aci, n_cci + n_aci


def _compute_cost(x, W, M=None):
    if M is not None:
        return compute_cost_cci_aci(x, W, M)
    return compute_cost_cci(x, W)


def _validate_if_available(experiment_id: str, modes: list = None):
    try:
        from validate_results import validate_experiment
        if modes is None:
            modes = ["cochannel", "adjacent"]
        validate_experiment(experiment_id=experiment_id, modes=modes)
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARNING] Validation pipeline failed for {experiment_id}: {e}")


# ============================================================================
# E1 - VISUAL VALIDATION
# ============================================================================
def run_experiment_E1(verbose=False):
    plt.close('all')
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
    # [FIX-LAYOUT] Figure() au lieu de plt.figure() : figure hors registre pyplot.
    fig, canvas = _new_figure((8, 6))
    ax = fig.add_subplot(111)
    nx.draw_networkx_nodes(G, pos_dict, ax=ax, node_color='lightblue',
                           node_size=500, edgecolors='black', linewidths=1)
    nx.draw_networkx_edges(G, pos_dict, ax=ax, edge_color='gray', width=2)
    nx.draw_networkx_labels(G, pos_dict, ax=ax, font_size=10, font_weight='bold')

    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    if x_range <= 0:
        x_range = 1.0
    if y_range <= 0:
        y_range = 1.0
    margin_x = x_range * 0.15
    margin_y = y_range * 0.15

    ax.set_xlim(min(x_coords) - margin_x, max(x_coords) + margin_x)
    ax.set_ylim(min(y_coords) - margin_y, max(y_coords) + margin_y)

    ax.set_title(f"Scenario {scenario_name} - Interference Graph "
                 f"(N={N}, K={K}, seed={seed})",
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # [FIX-LAYOUT] Marges explicites + fig.savefig (le canvas est deja attache).
    fig.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95)
    fig.savefig(str(e1_fig_dir / "interference_graph.png"), dpi=300)
    print("[INFO] Figure E1/interference_graph.png saved.")

    # 2. Matrix W Visualization
    # [FIX-LAYOUT] Figure() + positions en coordonnees figure via add_axes.
    # La colorbar utilise une cax dediee, ce qui evite tout conflit avec
    # un eventuel moteur de layout.
    fig, canvas = _new_figure((8, 6))

    # Axes principal : position explicite [left, bottom, w, h]
    ax = fig.add_axes([0.12, 0.12, 0.70, 0.75])
    im = ax.imshow(W, cmap='Reds', interpolation='nearest', vmin=0, vmax=4)

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels([str(i) for i in range(N)], fontsize=10)
    ax.set_yticklabels([str(i) for i in range(N)], fontsize=10)

    for i in range(N):
        for j in range(N):
            if W[i, j] > 0:
                text_color = "white" if W[i, j] >= 2 else "black"
                ax.text(j, i, int(W[i, j]), ha='center', va='center',
                        color=text_color, fontsize=10, fontweight='bold')

    # Axes dedie exclusivement a la colorbar (position fixe sur la droite)
    cax = fig.add_axes([0.84, 0.12, 0.03, 0.75])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Intensite de couplage W_ij", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(f"Scenario {scenario_name} - Weight Matrix W (N={N}, seed={seed})",
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Cellule j", fontsize=11)
    ax.set_ylabel("Cellule i", fontsize=11)

    fig.savefig(str(e1_fig_dir / "matrix_W.png"), dpi=300)
    print("[INFO] Figure E1/matrix_W.png saved.")

    # Helper functions for colored graphs
    def draw_colored_graph(ax, x, title):
        node_colors = [colors[c] for c in x]
        nx.draw_networkx_nodes(G, pos_dict, ax=ax, node_color=node_colors,
                               node_size=500, edgecolors='black', linewidths=1)
        nx.draw_networkx_edges(G, pos_dict, ax=ax, edge_color='gray', width=2)
        nx.draw_networkx_labels(G, pos_dict, ax=ax, font_size=10, font_weight='bold')
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    def save_colored_graph(method_name, x, fig_dir, subfolder, label_suffix=""):
        # [FIX-LAYOUT] Figure() au lieu de plt.figure().
        # Combine fig.legend global + fig.suptitle + subplots_adjust : c'etait
        # le pattern le plus sensible au moteur de layout (bug E2 d'origine).
        fig, canvas = _new_figure((8, 6))
        ax = fig.add_subplot(111)
        title = f"{method_name}{label_suffix}"
        draw_colored_graph(ax, x, title)

        x_coords_col = [p[0] for p in positions]
        y_coords_col = [p[1] for p in positions]
        margin_x_col = (max(x_coords_col) - min(x_coords_col)) * 0.15 if len(x_coords_col) > 1 else 10.0
        margin_y_col = (max(y_coords_col) - min(y_coords_col)) * 0.15 if len(y_coords_col) > 1 else 10.0
        ax.set_xlim(min(x_coords_col) - margin_x_col, max(x_coords_col) + margin_x_col)
        ax.set_ylim(min(y_coords_col) - margin_y_col, max(y_coords_col) + margin_y_col)

        legend_elements = [Patch(facecolor=colors[c], edgecolor='black',
                                 label=f'Channel {c}') for c in range(K)]
        fig.legend(handles=legend_elements, loc='lower center', ncol=K,
                   fontsize=10, bbox_to_anchor=(0.5, 0.02))
        fig.suptitle(f"E1 - {fig_dir.name} - {title} - S1 (N={N}, K={K}, seed={seed})",
                     fontsize=13, fontweight='bold', y=0.97)

        # Marges explicites : bottom eleve pour accueillir la legende
        fig.subplots_adjust(top=0.90, bottom=0.14, left=0.05, right=0.95)

        filename = f"{method_name.lower().replace(' ', '_')}{label_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        fig.savefig(str(fig_dir / subfolder / filename), dpi=300)
        print(f"[INFO] Figure E1/{subfolder}/{filename} saved.")

    models = [
        {"name": "cochannel", "M": None, "suffix": " (co-channel)", "folder": "cochannel"},
        {"name": "adjacent", "M": create_channel_interference_matrix(K),
         "suffix": " (adjacent)", "folder": "adjacent"}
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

        x_bd_final, history_bd, _, n_cci, n_aci, best_sweep = bdcenn_allocation(
            N, K, W, M=M,
            num_restarts=config.NUM_RESTARTS,
            max_sweeps=config.MAX_SWEEPS_BD,
            random_order=True,
            seed=seed,
            verbose=False
        )
        x_bd_initial = history_bd[0][2]

        results = {}
        for name, x_arr in [("Random", x_rand), ("Greedy", x_greedy),
                            ("DSATUR", x_dsatur), ("BD-CeNN_initial", x_bd_initial),
                            ("BD-CeNN_final", x_bd_final)]:
            cost = _compute_cost(x_arr, W, M)
            c_cci, c_aci, c_tot = _compute_all_conflicts(x_arr, W, M)
            results[name] = {
                "cost": cost,
                "conflicts_cci": c_cci,
                "conflicts_aci": c_aci,
                "conflicts_total": c_tot,
                "used_channels": len(set(x_arr))
            }

        save_colored_graph("Random", x_rand, e1_fig_dir, folder, suffix)
        save_colored_graph("Greedy", x_greedy, e1_fig_dir, folder, suffix)
        save_colored_graph("DSATUR", x_dsatur, e1_fig_dir, folder, suffix)
        save_colored_graph("BD-CeNN initial", x_bd_initial, e1_fig_dir, folder,
                           suffix + " (initial)")
        save_colored_graph("BD-CeNN final", x_bd_final, e1_fig_dir, folder,
                           suffix + " (final)")

        df_cell_channels = pd.DataFrame({
            "Cell": list(range(N)),
            "Random": x_rand,
            "Greedy": x_greedy,
            "DSATUR": x_dsatur,
            "BD-CeNN_initial": x_bd_initial,
            "BD-CeNN_final": x_bd_final,
            "seed": seed
        })
        df_cell_channels.to_csv(str(model_csv_dir / "cell_channels.csv"), index=False)
        print(f"[SUCCESS] CSV E1/{folder}/cell_channels.csv saved.")

        df_metrics = pd.DataFrame([
            {
                "Method": name,
                "Cout": v["cost"],
                "conflicts_cci": v["conflicts_cci"],
                "conflicts_aci": v["conflicts_aci"],
                "conflicts_total": v["conflicts_total"],
                "used_channels": v["used_channels"],
                "seed": seed
            }
            for name, v in results.items()
        ])
        df_metrics.to_csv(str(model_csv_dir / "method_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E1/{folder}/method_metrics.csv saved.")

    _validate_if_available("E1")
    print("[SUCCESS] Experiment E1 completed.")


# ============================================================================
# E2 - FACTORIAL RUN  [FIX-LAYOUT] deja migre
# ============================================================================
def run_experiment_E2(verbose=False):
    """
    E2 - Factorial runtime comparison over multiple settings.
    Studies the sensitivity profile of N, K, and density across algorithms.

    Note de mise en page : les figures 2x2 avec fig.legend + fig.suptitle
    utilisent le pattern Figure() + FigureCanvasAgg() + set_layout_engine('none')
    avec positions explicites via add_axes(), ce qui garantit l'absence de
    chevauchement et de page blanche.
    """
    plt.close('all')
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
                            n_sweeps = np.nan
                        elif method_name == "Greedy":
                            np.random.seed(seed)
                            order = np.random.permutation(N).tolist()
                            x = method_func(N, K, W, order=order, M=M)
                            n_sweeps = np.nan
                        elif method_name == "DSATUR":
                            x = method_func(N, K, W, M=M)
                            n_sweeps = np.nan
                        else:
                            x, _, _, _, _, best_sweep = method_func(
                                N, K, W, M=M,
                                num_restarts=config.NUM_RESTARTS,
                                max_sweeps=config.MAX_SWEEPS_BD,
                                random_order=True,
                                seed=seed,
                                verbose=False
                            )
                            n_sweeps = best_sweep
                        elapsed = time.perf_counter() - start_time

                        cost = _compute_cost(x, W, M)
                        c_cci, c_aci, c_tot = _compute_all_conflicts(x, W, M)

                        raw_data.append({
                            "N": N,
                            "K": K,
                            "scenario": sc_name,
                            "density_label": label,
                            "threshold": th,
                            "seed": seed,
                            "method": method_name,
                            "cost": cost,
                            "conflicts_cci": c_cci,
                            "conflicts_aci": c_aci,
                            "conflicts_total": c_tot,
                            "used_channels": len(set(x)),
                            "time": elapsed,
                            "n_sweeps": n_sweeps
                        })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(str(model_csv_dir / "raw_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E2/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["N", "K", "density_label", "threshold", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts_cci": ["mean", "std", "min", "max", "median"],
            "conflicts_aci": ["mean", "std", "min", "max", "median"],
            "conflicts_total": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"],
            "n_sweeps": ["mean", "std", "min", "max", "median"]
        }).reset_index()

        summary.columns = [
            'N', 'K', 'density_label', 'threshold', 'method',
            'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
            'conflicts_cci_mean', 'conflicts_cci_std', 'conflicts_cci_min', 'conflicts_cci_max', 'conflicts_cci_median',
            'conflicts_aci_mean', 'conflicts_aci_std', 'conflicts_aci_min', 'conflicts_aci_max', 'conflicts_aci_median',
            'conflicts_total_mean', 'conflicts_total_std', 'conflicts_total_min', 'conflicts_total_max', 'conflicts_total_median',
            'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
            'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
            'n_sweeps_mean', 'n_sweeps_std', 'n_sweeps_min', 'n_sweeps_max', 'n_sweeps_median'
        ]

        for col in summary.columns[5:]:
            summary[col] = summary[col].round(1)

        # n_sweeps est exclu pour preserver NaN sur les baselines
        for col in summary.columns:
            if any(k in col for k in ['conflicts', 'used_channels']) and 'std' not in col:
                summary[col] = summary[col].fillna(0).round(0).astype(int)

        summary.to_csv(str(model_csv_dir / "summary_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E2/{folder}/summary_metrics.csv saved.")

        # ---- Figures ----
        metrics_to_plot = [
            ('cost', 'Mean Global Cost J(x)'),
            ('conflicts_total', 'Mean Conflicted Links'),
            ('used_channels', 'Mean Channels Used'),
            ('time', 'Execution Runtime (seconds)')
        ]

        subplots_config = [
            (30, 4, "N=30, K=4"),
            (30, 6, "N=30, K=6"),
            (50, 4, "N=50, K=4"),
            (50, 6, "N=50, K=6")
        ]

        method_colors = {
            'Random': 'royalblue',
            'Greedy': 'forestgreen',
            'DSATUR': 'darkorange',
            'BD-CeNN': 'firebrick'
        }

        for metric_col, ylabel in metrics_to_plot:
            # [FIX-LAYOUT] Figure() + FigureCanvasAgg() : figure hors pyplot.
            fig, canvas = _new_figure((14, 10))

            # Positions en coordonnees figure [left, bottom, width, height]
            positions = [
                (0.07, 0.54, 0.41, 0.34),  # haut-gauche   : N=30, K=4
                (0.55, 0.54, 0.41, 0.34),  # haut-droite   : N=30, K=6
                (0.07, 0.13, 0.41, 0.34),  # bas-gauche    : N=50, K=4
                (0.55, 0.13, 0.41, 0.34),  # bas-droite    : N=50, K=6
            ]
            axes_flat = [fig.add_axes(pos) for pos in positions]

            for ax, (N, K, title) in zip(axes_flat, subplots_config):
                sub = summary[(summary['N'] == N) & (summary['K'] == K)]
                if sub.empty:
                    ax.set_title(f"{title} (No Data)")
                    ax.axis('off')
                    continue

                density_order = ["Low", "Medium", "High"]
                pivot_mean = sub.pivot(
                    index='density_label', columns='method',
                    values=f'{metric_col}_mean'
                ).reindex(density_order)
                pivot_std = sub.pivot(
                    index='density_label', columns='method',
                    values=f'{metric_col}_std'
                ).reindex(density_order)

                x = np.arange(len(density_order))
                width = 0.2
                for i, method in enumerate(methods.keys()):
                    if method not in pivot_mean.columns:
                        continue
                    means = pivot_mean[method].fillna(0).values
                    stds = pivot_std[method].fillna(0).values
                    offset = (i - 1.5) * width
                    ax.bar(x + offset, means, width, yerr=stds,
                           capsize=3, label=method,
                           color=method_colors[method], alpha=0.8)

                ax.set_xticks(x)
                ax.set_xticklabels(density_order, fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=10)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                if metric_col == 'used_channels':
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(
                handles, labels,
                loc='lower center',
                ncol=len(methods),
                fontsize=10,
                bbox_to_anchor=(0.5, 0.005),
                frameon=False
            )

            fig.suptitle(
                f"E2 - {ylabel} Evaluation - {model_name}",
                fontsize=14, fontweight='bold', y=0.96
            )

            fname = f"E2_{metric_col}_{model_name}.png"
            fig.savefig(str(model_fig_dir / fname), dpi=300)
            print(f"[INFO] Figure E2/{folder}/{fname} saved.")

    _validate_if_available("E2")
    print("[SUCCESS] Experiment E2 completed.")


# ============================================================================
# E3 - SPECTRUM SCALING (K-SENSITIVITY)
# ============================================================================
def run_experiment_E3(verbose=False):
    plt.close('all')
    scenario_name = "S4"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"[ERROR] Scenario {scenario_name} missing from database."); return

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
                        x, _, _, _, _, _ = method_func(
                            N, K, W, M=M,
                            num_restarts=config.NUM_RESTARTS,
                            max_sweeps=config.MAX_SWEEPS_BD,
                            random_order=True,
                            seed=seed,
                            verbose=False
                        )

                    cost = _compute_cost(x, W, M)
                    c_cci, c_aci, c_tot = _compute_all_conflicts(x, W, M)

                    raw_data.append({
                        "K": K,
                        "seed": seed,
                        "method": method_name,
                        "cost": cost,
                        "conflicts_cci": c_cci,
                        "conflicts_aci": c_aci,
                        "conflicts_total": c_tot,
                        "used_channels": len(set(x))
                    })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(str(model_csv_dir / "raw_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E3/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["K", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts_cci": ["mean", "std", "min", "max", "median"],
            "conflicts_aci": ["mean", "std", "min", "max", "median"],
            "conflicts_total": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"]
        })

        # Aplatissement des colonnes multi-niveaux vers la nomenclature du projet
        summary.columns = [
            f"{col}_{stat}" for col, stat in summary.columns
        ]
        summary = summary.reset_index()

        for col in summary.columns[2:]:
            summary[col] = summary[col].round(1)

        for col in summary.columns:
            if any(k in col for k in ['conflicts', 'used_channels']) and 'std' not in col:
                summary[col] = summary[col].round(0).astype(int)

        summary.to_csv(str(model_csv_dir / "summary_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E3/{folder}/summary_metrics.csv saved.")

        pivot_cost = summary.pivot(index='K', columns='method', values=['cost_mean', 'cost_std'])
        pivot_conflicts = summary.pivot(index='K', columns='method',
                                        values=['conflicts_total_mean', 'conflicts_total_std'])

        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen',
                         'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
        method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}

        # ---- Figure 1 : Cost vs K ----
        # [FIX-LAYOUT] Figure() au lieu de plt.figure().
        fig1, canvas1 = _new_figure((10, 6))
        ax1 = fig1.add_subplot(111)

        for method in methods.keys():
            means = pivot_cost[('cost_mean', method)].values
            stds = pivot_cost[('cost_std', method)].values
            ax1.errorbar(K_values, means, yerr=stds,
                         label=method, color=method_colors[method],
                         marker=method_markers[method],
                         capsize=5, linewidth=2, markersize=8)

        ax1.set_xlabel("Nombre de canaux K", fontsize=12)
        ax1.set_ylabel("Coût global moyen J(x)", fontsize=12)
        ax1.set_title(f"E3 - Impact de K sur le coût moyen - {model_name}",
                      fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        fig1.subplots_adjust(top=0.92, bottom=0.10, left=0.09, right=0.97)
        fig1.savefig(str(model_fig_dir / f"cost_vs_K_{model_name}.png"), dpi=300)
        print(f"[INFO] Figure E3/{folder}/cost_vs_K_{model_name}.png saved.")

        # ---- Figure 2 : Conflicts vs K ----
        # [FIX-LAYOUT] Figure() au lieu de plt.figure().
        fig2, canvas2 = _new_figure((10, 6))
        ax2 = fig2.add_subplot(111)

        for method in methods.keys():
            means = pivot_conflicts[('conflicts_total_mean', method)].values
            stds = pivot_conflicts[('conflicts_total_std', method)].values
            ax2.errorbar(K_values, means, yerr=stds,
                         label=method, color=method_colors[method],
                         marker=method_markers[method],
                         capsize=5, linewidth=2, markersize=8)

        ax2.set_xlabel("Nombre de canaux K", fontsize=12)
        ax2.set_ylabel("Conflits totaux moyens", fontsize=12)
        ax2.set_title(f"E3 - Impact de K sur les conflits moyens - {model_name}",
                      fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)
        fig2.subplots_adjust(top=0.92, bottom=0.10, left=0.09, right=0.97)
        fig2.savefig(str(model_fig_dir / f"conflicts_vs_K_{model_name}.png"), dpi=300)
        print(f"[INFO] Figure E3/{folder}/conflicts_vs_K_{model_name}.png saved.")

    _validate_if_available("E3")
    print("[SUCCESS] Experiment E3 completed.")


# ============================================================================
# E4 - IMPACT DE LA DENSITÉ
# ============================================================================
def run_experiment_E4(verbose=False):
    plt.close('all')
    scenario_name = "S3"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"[ERROR] Scenario {scenario_name} missing from database."); return

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
                        n_sweeps = np.nan
                    elif method_name == "Greedy":
                        np.random.seed(seed)
                        order = np.random.permutation(N).tolist()
                        x = method_func(N, K, W, order=order, M=M)
                        n_sweeps = np.nan
                    elif method_name == "DSATUR":
                        x = method_func(N, K, W, M=M)
                        n_sweeps = np.nan
                    else:
                        x, _, _, _, _, best_sweep = method_func(
                            N, K, W, M=M,
                            num_restarts=config.NUM_RESTARTS,
                            max_sweeps=config.MAX_SWEEPS_BD,
                            random_order=True,
                            seed=seed,
                            verbose=False
                        )
                        n_sweeps = best_sweep
                    elapsed = time.perf_counter() - start_time

                    cost = _compute_cost(x, W, M)
                    c_cci, c_aci, c_tot = _compute_all_conflicts(x, W, M)

                    raw_data.append({
                        "density_label": label,
                        "threshold": th,
                        "seed": seed,
                        "method": method_name,
                        "cost": cost,
                        "conflicts_cci": c_cci,
                        "conflicts_aci": c_aci,
                        "conflicts_total": c_tot,
                        "used_channels": len(set(x)),
                        "time": elapsed,
                        "n_sweeps": n_sweeps
                    })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(str(model_csv_dir / "raw_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E4/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["density_label", "threshold", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts_cci": ["mean", "std", "min", "max", "median"],
            "conflicts_aci": ["mean", "std", "min", "max", "median"],
            "conflicts_total": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"],
            "n_sweeps": ["mean", "std", "min", "max", "median"]
        }).reset_index()

        summary.columns = [
            'density_label', 'threshold', 'method',
            'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
            'conflicts_cci_mean', 'conflicts_cci_std', 'conflicts_cci_min',
            'conflicts_cci_max', 'conflicts_cci_median',
            'conflicts_aci_mean', 'conflicts_aci_std', 'conflicts_aci_min',
            'conflicts_aci_max', 'conflicts_aci_median',
            'conflicts_total_mean', 'conflicts_total_std', 'conflicts_total_min',
            'conflicts_total_max', 'conflicts_total_median',
            'used_channels_mean', 'used_channels_std', 'used_channels_min',
            'used_channels_max', 'used_channels_median',
            'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
            'n_sweeps_mean', 'n_sweeps_std', 'n_sweeps_min', 'n_sweeps_max', 'n_sweeps_median'
        ]

        for col in summary.columns[3:]:
            summary[col] = summary[col].round(1)
            if any(k in col for k in ['conflicts', 'used_channels']) and 'std' not in col:
                summary[col] = summary[col].fillna(0).round(0).astype(int)

        summary.to_csv(str(model_csv_dir / "summary_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E4/{folder}/summary_metrics.csv saved.")

        pivot_cost = summary.pivot(index='threshold', columns='method',
                                   values=['cost_mean', 'cost_std'])
        x_vals = sorted(summary['threshold'].unique())
        density_labels = [d["label"] for d in density_configs]

        # [FIX-LAYOUT] Figure() au lieu de plt.figure().
        # Cas sensible : combine ax.twiny() + ax.legend() + marges explicites.
        # On utilise add_axes() pour fixer la position en coordonnees figure,
        # ce qui empeche tout moteur de layout de recalculer les marges.
        fig, canvas = _new_figure((10, 6))
        ax = fig.add_axes([0.09, 0.22, 0.88, 0.68])  # bottom=0.22 pour twiny
        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen',
                         'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
        method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}

        for method in methods.keys():
            means = pivot_cost[('cost_mean', method)].values
            stds = pivot_cost[('cost_std', method)].values
            ax.errorbar(x_vals, means, yerr=stds,
                        label=method, color=method_colors[method],
                        marker=method_markers[method],
                        capsize=5, linewidth=2, markersize=8)

        ax.set_xlabel("Seuil d'interférence (threshold) - Densité croissante", fontsize=12)
        ax.set_ylabel("Coût global moyen J(x)", fontsize=12)
        ax.set_title(f"E4 - Impact de la densité sur le coût moyen - {model_name}",
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{th}" for th in x_vals])

        ax2 = ax.twiny()
        ax2.set_xticks(x_vals)
        ax2.set_xticklabels(density_labels)
        ax2.set_xlabel("Niveau de densité")
        ax2.xaxis.set_label_position('bottom')
        ax2.xaxis.tick_bottom()
        ax2.xaxis.set_ticks_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 40))

        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)
        # [FIX-LAYOUT] subplots_adjust supprime : position deja fixee par add_axes.
        fig.savefig(str(model_fig_dir / f"cost_vs_density_{model_name}.png"), dpi=300)

    _validate_if_available("E4")
    print("[SUCCESS] Experiment E4 completed.")


# ============================================================================
# E5 - CONVERGENCE DYNAMICS
# ============================================================================
def run_experiment_E5(verbose=False):
    plt.close('all')
    scenarios_to_run = ["S2", "S3"]
    horizon = config.MAX_SWEEPS_BD

    e5_fig_dir = config.FIGURES_DIR / "E5"
    e5_csv_dir = config.CSV_DIR / "E5"
    os.makedirs(e5_fig_dir, exist_ok=True)
    os.makedirs(e5_csv_dir, exist_ok=True)

    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}
    ]

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e5_fig_dir / folder
        model_csv_dir = e5_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        scenario_trajectories = {}
        scenario_sweeps = {}
        scenario_cost_initial = {}
        scenario_cost_final = {}

        for scenario_name in scenarios_to_run:
            instances = all_instances.get(scenario_name, {})
            if not instances:
                print(f"[ERROR] Scenario {scenario_name} missing from database."); continue

            first_inst = list(instances.values())[0]
            N, K = first_inst["N"], first_inst["K"]
            M = create_channel_interference_matrix(K) if model_name == "adjacent" else None
            seeds = sorted([int(s) for s in instances.keys() if s.isdigit()])

            trajectories_matrix = np.zeros((len(seeds), horizon + 1))
            sweeps_per_seed = []
            cost_initial_per_seed = []
            cost_final_per_seed = []

            print(f"[INFO] E5 - Scenario {scenario_name} ({model_name}): "
                  f"evaluating convergence on 30 certified topologies...")

            for idx, seed in enumerate(seeds):
                inst = instances.get(str(seed))
                if inst is None: continue
                W = np.array(inst["W"])

                x_final, history, elapsed, _, _, best_sweep = bdcenn_allocation(
                    N, K, W, M=M,
                    num_restarts=config.NUM_RESTARTS,
                    max_sweeps=horizon,
                    random_order=True,
                    seed=seed,
                    verbose=False
                )

                raw_trajectory = [entry[1] for entry in history]
                actual_length = len(raw_trajectory)

                if actual_length > 0:
                    cost_initial = raw_trajectory[0]
                    cost_final = raw_trajectory[-1]

                    filled = np.zeros(horizon + 1)
                    for t in range(horizon + 1):
                        filled[t] = raw_trajectory[t] if t < len(raw_trajectory) else raw_trajectory[-1]
                    trajectories_matrix[idx, :] = filled

                    sweeps_per_seed.append(best_sweep)
                    cost_initial_per_seed.append(cost_initial)
                    cost_final_per_seed.append(cost_final)

            scenario_trajectories[scenario_name] = {
                "mean": np.mean(trajectories_matrix, axis=0),
                "std": np.std(trajectories_matrix, axis=0, ddof=1),
                "N": N, "K": K}
            scenario_sweeps[scenario_name] = np.array(sweeps_per_seed)
            scenario_cost_initial[scenario_name] = np.array(cost_initial_per_seed)
            scenario_cost_final[scenario_name] = np.array(cost_final_per_seed)

        # ---- Figure 1 : Convergence curves ----
        for scenario_name in scenarios_to_run:
            if scenario_name not in scenario_trajectories: continue
            data = scenario_trajectories[scenario_name]
            # [FIX-LAYOUT] Figure() au lieu de plt.figure().
            fig, canvas = _new_figure((10, 6))
            ax = fig.add_subplot(111)
            iters = np.arange(horizon + 1)
            ax.plot(iters, data["mean"], color='firebrick', linewidth=2.5,
                    label='Cout moyen J(t)', zorder=3)
            ax.fill_between(iters, data["mean"] - data["std"],
                            data["mean"] + data["std"],
                            color='firebrick', alpha=0.20,
                            label='Plus ou moins 1 SD')
            ax.set_xlabel("Sweeps (balayages complets)", fontsize=12)
            ax.set_ylabel("Cout global J(x)", fontsize=12)
            ax.set_title(f"E5 - Convergence BD-CeNN - {scenario_name} "
                         f"(N={data['N']}, K={data['K']}) - {model_name}",
                         fontsize=13, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='upper right', fontsize=11)
            ax.set_xlim([0, horizon])
            fig.subplots_adjust(top=0.90, bottom=0.10, left=0.09, right=0.97)
            fig.savefig(str(model_fig_dir / f"convergence_curve_{scenario_name}_{model_name}.png"), dpi=300)

        # ---- Figure 2 : Sweeps histogram ----
        # [FIX-LAYOUT] Figure() au lieu de plt.figure().
        fig, canvas = _new_figure((9, 6))
        ax = fig.add_subplot(111)
        bar_labels, bar_means, bar_stds = [], [], []
        for scenario_name in scenarios_to_run:
            if scenario_name not in scenario_sweeps: continue
            sw = scenario_sweeps[scenario_name]
            data = scenario_trajectories[scenario_name]
            bar_labels.append(f"{scenario_name}\n(N={data['N']}, K={data['K']})")
            bar_means.append(int(round(np.mean(sw))) if len(sw) > 0 else 0)
            bar_stds.append(float(np.std(sw, ddof=1)) if len(sw) > 1 else 0.0)
        x_pos = np.arange(len(bar_labels))
        bars = ax.bar(x_pos, bar_means, yerr=bar_stds, capsize=10,
                      color=['steelblue', 'darkorange'][:len(bar_labels)],
                      edgecolor='black', linewidth=1.2, alpha=0.85)
        for i, (bar, m, s) in enumerate(zip(bars, bar_means, bar_stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                    f"{m} +/- {s:.1f} sweeps", ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos); ax.set_xticklabels(bar_labels, fontsize=11)
        ax.set_ylabel("Sweeps moyens vers le minimum", fontsize=12)
        ax.set_title(f"E5 - Sweeps moyens - {model_name}", fontsize=13, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if bar_means:
            ax.set_ylim([0, max(bar_means) + max(bar_stds)*2 + 5])
        fig.subplots_adjust(top=0.90, bottom=0.12, left=0.10, right=0.97)
        fig.savefig(str(model_fig_dir / f"sweeps_histogram_{model_name}.png"), dpi=300)

        # ---- CSV convergence metrics ----
        rows = []
        for scenario_name in scenarios_to_run:
            if scenario_name not in scenario_sweeps: continue
            sw = scenario_sweeps[scenario_name]
            ci = scenario_cost_initial[scenario_name]
            cf = scenario_cost_final[scenario_name]
            data = scenario_trajectories[scenario_name]
            def _s(a):
                if len(a) == 0: return dict(mean=0,std=0,median=0,min=0,max=0)
                return dict(mean=float(np.mean(a)),
                            std=float(np.std(a,ddof=1)) if len(a)>1 else 0.0,
                            median=float(np.median(a)),
                            min=float(np.min(a)), max=float(np.max(a)))
            ss, cis, cfs = _s(sw), _s(ci), _s(cf)
            rows.append({"scenario": scenario_name, "N": data["N"], "K": data["K"],
                         "n_seeds": len(sw),
                         "n_sweeps_mean": round(ss["mean"],2),
                         "n_sweeps_std": round(ss["std"],2),
                         "n_sweeps_median": round(ss["median"],2),
                         "n_sweeps_min": int(ss["min"]),
                         "n_sweeps_max": int(ss["max"]),
                         "cost_initial_mean": round(cis["mean"],4),
                         "cost_initial_std": round(cis["std"],4),
                         "cost_final_mean": round(cfs["mean"],4),
                         "cost_final_std": round(cfs["std"],4)})
        pd.DataFrame(rows).to_csv(str(model_csv_dir / "convergence_metrics.csv"),
                                  index=False, float_format="%.6f")

        for scenario_name in scenarios_to_run:
            if scenario_name not in scenario_trajectories: continue
            data = scenario_trajectories[scenario_name]
            pd.DataFrame({"sweep": np.arange(horizon+1),
                          "mean_cost": data["mean"], "std_cost": data["std"]
                          }).to_csv(str(model_csv_dir / f"trajectory_{scenario_name}.csv"),
                                    index=False, float_format="%.6f")

    _validate_if_available("E5")
    print("[SUCCESS] Experiment E5 completed.")


# ============================================================================
# E6 - SCALABILITE
# ============================================================================
def run_experiment_E6(verbose=False):
    plt.close('all')
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
                        n_sweeps = np.nan
                    elif method_name == "Greedy":
                        np.random.seed(seed)
                        order = np.random.permutation(N).tolist()
                        x = method_func(N, K, W, order=order, M=M)
                        n_sweeps = np.nan
                    elif method_name == "DSATUR":
                        x = method_func(N, K, W, M=M)
                        n_sweeps = np.nan
                    else:
                        x, _, _, _, _, best_sweep = method_func(
                            N, K, W, M=M,
                            num_restarts=config.NUM_RESTARTS,
                            max_sweeps=config.MAX_SWEEPS_BD,
                            random_order=True,
                            seed=seed,
                            verbose=False
                        )
                        n_sweeps = best_sweep
                    elapsed = time.perf_counter() - start_time

                    cost = _compute_cost(x, W, M)
                    c_cci, c_aci, c_tot = _compute_all_conflicts(x, W, M)
                    normalized_cost = cost / sum_weights if sum_weights > 0 else np.nan

                    raw_data.append({
                        "N": N,
                        "seed": seed,
                        "method": method_name,
                        "cost": cost,
                        "normalized_cost": normalized_cost,
                        "conflicts_cci": c_cci,
                        "conflicts_aci": c_aci,
                        "conflicts_total": c_tot,
                        "used_channels": len(set(x)),
                        "time": elapsed,
                        "n_sweeps": n_sweeps
                    })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(str(model_csv_dir / "raw_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E6/{folder}/raw_metrics.csv saved.")

        summary = df_raw.groupby(["N", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "normalized_cost": ["mean", "std", "min", "max", "median"],
            "conflicts_cci": ["mean", "std", "min", "max", "median"],
            "conflicts_aci": ["mean", "std", "min", "max", "median"],
            "conflicts_total": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"],
            "n_sweeps": ["mean", "std", "min", "max", "median"]
        }).reset_index()

        summary.columns = [
            'N', 'method',
            'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
            'normalized_cost_mean', 'normalized_cost_std', 'normalized_cost_min',
            'normalized_cost_max', 'normalized_cost_median',
            'conflicts_cci_mean', 'conflicts_cci_std', 'conflicts_cci_min',
            'conflicts_cci_max', 'conflicts_cci_median',
            'conflicts_aci_mean', 'conflicts_aci_std', 'conflicts_aci_min',
            'conflicts_aci_max', 'conflicts_aci_median',
            'conflicts_total_mean', 'conflicts_total_std', 'conflicts_total_min',
            'conflicts_total_max', 'conflicts_total_median',
            'used_channels_mean', 'used_channels_std', 'used_channels_min',
            'used_channels_max', 'used_channels_median',
            'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
            'n_sweeps_mean', 'n_sweeps_std', 'n_sweeps_min', 'n_sweeps_max', 'n_sweeps_median'
        ]

        for col in summary.columns[2:]:
            summary[col] = summary[col].round(1)

        for col in summary.columns:
            if any(k in col for k in ['conflicts', 'used_channels']) and 'std' not in col:
                summary[col] = summary[col].fillna(0).round(0).astype(int)

        summary.to_csv(str(model_csv_dir / "summary_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E6/{folder}/summary_metrics.csv saved.")

        metrics_to_plot = [
            ('cost_mean', 'cost_std', 'Cout global moyen J(x)'),
            ('time_mean', 'time_std', "Temps d'execution moyen (s)"),
            ('n_sweeps_mean', 'n_sweeps_std', "Sweeps moyens (BD-CeNN)")
        ]

        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen',
                         'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
        method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}

        for metric_col, std_col, ylabel in metrics_to_plot:
            # [FIX-LAYOUT] Figure() au lieu de plt.figure().
            fig, canvas = _new_figure((10, 6))
            ax = fig.add_subplot(111)

            if 'sweeps' in metric_col:
                sub = summary[summary['method'] == 'BD-CeNN'].set_index('N').reindex(N_values).reset_index()
                ax.errorbar(sub['N'].values, sub[metric_col].values, yerr=sub[std_col].values,
                            fmt='o-', capsize=5, color='firebrick', label='BD-CeNN',
                            linewidth=2, markersize=8)
                ax.set_title(f"E6 - Complexite dynamique - {ylabel} (BD-CeNN)")
            else:
                for method in methods.keys():
                    sub = summary[summary['method'] == method].set_index('N').reindex(N_values).reset_index()
                    ax.errorbar(sub['N'].values, sub[metric_col].values, yerr=sub[std_col].values,
                                fmt=method_markers[method]+'-', capsize=5,
                                color=method_colors[method], label=method,
                                linewidth=2, markersize=8)
                ax.set_title(f"E6 - Complexite structurelle - {ylabel}")

            ax.set_xlabel("Nombre de cellules N", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            fig.subplots_adjust(top=0.92, bottom=0.10, left=0.09, right=0.97)

            fname = f"{metric_col.replace('_mean','')}_vs_N_{model_name}.png"
            fig.savefig(str(model_fig_dir / fname), dpi=300)
            print(f"[INFO] Figure E6/{folder}/{fname} saved.")

        # ---- Figure 4 : Normalized Cost vs N ----
        # [FIX-LAYOUT] Figure() au lieu de plt.figure().
        fig, canvas = _new_figure((10, 6))
        ax = fig.add_subplot(111)

        for method in methods.keys():
            sub = summary[summary['method'] == method].set_index('N').reindex(N_values).reset_index()
            ax.errorbar(sub['N'].values, sub['normalized_cost_mean'].values,
                        yerr=sub['normalized_cost_std'].values,
                        fmt=method_markers[method]+'-', capsize=5,
                        color=method_colors[method], label=method,
                        linewidth=2, markersize=8)

        ax.set_xlabel("Nombre de cellules N", fontsize=12)
        ax.set_ylabel("Cout normalise moyen", fontsize=12)
        ax.set_title(f"E6 - Cout normalise moyen - {model_name}",
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.subplots_adjust(top=0.92, bottom=0.10, left=0.09, right=0.97)

        fname = f"normalized_cost_vs_N_{model_name}.png"
        fig.savefig(str(model_fig_dir / fname), dpi=300)
        print(f"[INFO] Figure E6/{folder}/{fname} saved.")
        print(f"[INFO] Operational charts saved inside E6/{folder}/.")

    _validate_if_available("E6")
    print("[SUCCESS] Experiment E6 completed.")


# ============================================================================
# E7 - ROBUSTESSE AU BRUIT
# ============================================================================
def run_experiment_E7(verbose=False):
    plt.close('all')
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

            x_ref, _, _, _, _, _ = method_func(
                N, K, W_clean, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_sweeps=config.MAX_SWEEPS_BD,
                random_order=True,
                seed=seed,
                verbose=False
            )
            ref_x[seed] = x_ref
            ref_cost[seed] = _compute_cost(x_ref, W_clean, M)

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

                x_noisy, _, _, _, _, _ = method_func(
                    N, K, W_noisy, M=M,
                    num_restarts=config.NUM_RESTARTS,
                    max_sweeps=config.MAX_SWEEPS_BD,
                    random_order=True,
                    seed=seed_noisy,
                    verbose=False
                )
                cost_noisy = _compute_cost(x_noisy, W_noisy, M)

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
        df_raw.to_csv(str(model_csv_dir / "raw_metrics.csv"), index=False, float_format='%.6f')
        print(f"[SUCCESS] CSV E7/{folder}/raw_metrics.csv saved.")

        # [FIX-E7] Ordre des statistiques aligne sur la convention du memoire :
        # moyenne -> ecart-type -> mediane -> minimum -> maximum
        summary = df_raw.groupby("noise_level").agg({
            "cost_relatif": ["mean", "std", "median", "min", "max"],
            "change_rate": ["mean", "std", "median", "min", "max"]
        }).reset_index()
        summary.columns = ['noise_level',
                           'cost_rel_mean', 'cost_rel_std', 'cost_rel_median',
                           'cost_rel_min', 'cost_rel_max',
                           'change_mean', 'change_std', 'change_median',
                           'change_min', 'change_max']
        for col in summary.columns[1:]:
            summary[col] = summary[col].round(1)

        summary.to_csv(str(model_csv_dir / "summary_metrics.csv"), index=False, float_format='%.1f')
        print(f"[SUCCESS] CSV E7/{folder}/summary_metrics.csv saved.")

        # [FIX] Pattern robuste + correction du bug : 'change_rate_std' -> 'change_std'
        # [FIX-LAYOUT] Figure() au lieu de plt.figure().
        fig, canvas = _new_figure((10, 6))
        ax = fig.add_subplot(111)
        noise_vals = [0] + sorted(summary['noise_level'].unique())
        cost_means = [0] + summary['cost_rel_mean'].tolist()
        cost_stds = [0] + summary['cost_rel_std'].tolist()
        change_means = [0] + summary['change_mean'].tolist()
        change_stds = [0] + summary['change_std'].tolist()   # <-- CORRECTION ICI

        ax.errorbar(noise_vals, cost_means, yerr=cost_stds, fmt='o-', capsize=6,
                    color='red', label='Degradation relative du cout (%)',
                    linewidth=2, markersize=8)
        ax.errorbar(noise_vals, change_means, yerr=change_stds, fmt='s-', capsize=6,
                    color='blue', label='Taux de changement de canal (%)',
                    linewidth=2, markersize=8)

        ax.set_xlabel("Niveau de bruit injecte (%)", fontsize=11)
        ax.set_ylabel("Pourcentage (%)", fontsize=11)
        ax.set_title(f"E7 - Robustesse au bruit de mesure - {model_name}",
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(-50, 100)

        fig.subplots_adjust(top=0.90, bottom=0.12, left=0.10, right=0.97)
        fname = f"robustness_{model_name}.png"
        fig.savefig(str(model_fig_dir / fname), dpi=300)
        print(f"[INFO] Figure E7/{folder}/{fname} saved.")

    _validate_if_available("E7")
    print("[SUCCESS] Experiment E7 completed.")


# ============================================================================
# E8 - RESEAU DYNAMIQUE
# ============================================================================
def run_experiment_E8(verbose=False):
    plt.close('all')
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

            x_init, _, _, _, _, _ = method_func(
                N, K, W_orig, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_sweeps=config.MAX_SWEEPS_BD,
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

                start = time.perf_counter()
                x_warm, _, _, _, _, _ = method_func(
                    N, K, W_dyn, M=M,
                    num_restarts=1,
                    max_sweeps=config.MAX_SWEEPS_BD,
                    random_order=True,
                    seed=seed + int(b_mod * 1000) + 300,
                    verbose=False
                )
                time_warm = time.perf_counter() - start

                start = time.perf_counter()
                x_cold, _, _, _, _, _ = method_func(
                    N, K, W_dyn, M=M,
                    num_restarts=config.NUM_RESTARTS,
                    max_sweeps=config.MAX_SWEEPS_BD,
                    random_order=True,
                    seed=seed + int(b_mod * 1000) + 400,
                    verbose=False
                )
                time_cold = time.perf_counter() - start

                changes_warm = np.sum(x_warm != ref_x[seed])
                changes_cold = np.sum(x_cold != ref_x[seed])

                cost_warm = _compute_cost(x_warm, W_dyn, M)
                cost_cold = _compute_cost(x_cold, W_dyn, M)

                raw_data.append({
                    "mod_level": b_mod * 100, "seed": seed, "mode": "warm",
                    "time": time_warm, "changes": changes_warm, "cost": cost_warm
                })
                raw_data.append({
                    "mod_level": b_mod * 100, "seed": seed, "mode": "cold",
                    "time": time_cold, "changes": changes_cold, "cost": cost_cold
                })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(str(model_csv_dir / "raw_metrics.csv"), index=False, float_format='%.6f')
        print(f"[SUCCESS] CSV E8/{folder}/raw_metrics.csv saved.")

        # [FIX-E8] Ordre des statistiques aligne sur la convention du memoire :
        # moyenne -> ecart-type -> mediane -> minimum -> maximum
        summary = df_raw.groupby(["mod_level", "mode"]).agg({
            "time": ["mean", "std", "median", "min", "max"],
            "changes": ["mean", "std", "median", "min", "max"],
            "cost": ["mean", "std", "median", "min", "max"]
        }).reset_index()
        summary.columns = ['mod_level', 'mode',
                           'time_mean', 'time_std', 'time_median', 'time_min', 'time_max',
                           'changes_mean', 'changes_std', 'changes_median', 'changes_min', 'changes_max',
                           'cost_mean', 'cost_std', 'cost_median', 'cost_min', 'cost_max']
        for col in summary.columns:
            if col not in ['mod_level', 'mode']:
                summary[col] = summary[col].round(1) if ('changes' in col or 'cost' in col) else summary[col].round(6)

        summary.to_csv(str(model_csv_dir / "summary_metrics.csv"), index=False, float_format='%.6f')
        print(f"[SUCCESS] CSV E8/{folder}/summary_metrics.csv saved.")

        # [FIX-LAYOUT] Figure() au lieu de plt.figure().
        fig, canvas = _new_figure((10, 6))
        ax = fig.add_subplot(111)
        warm_data = summary[summary['mode'] == 'warm']
        cold_data = summary[summary['mode'] == 'cold']

        mod_vals_warm = [0] + sorted(warm_data['mod_level'].unique())
        changes_warm_means = [0] + warm_data['changes_mean'].tolist()
        changes_warm_stds = [0] + warm_data['changes_std'].tolist()

        mod_vals_cold = [0] + sorted(cold_data['mod_level'].unique())
        changes_cold_means = [0] + cold_data['changes_mean'].tolist()
        changes_cold_stds = [0] + cold_data['changes_std'].tolist()

        ax.errorbar(mod_vals_warm, changes_warm_means, yerr=changes_warm_stds,
                    fmt='o-', capsize=6, color='green',
                    label='Warm Start Optimization', linewidth=2, markersize=8)
        ax.errorbar(mod_vals_cold, changes_cold_means, yerr=changes_cold_stds,
                    fmt='s-', capsize=6, color='purple',
                    label='Cold Start Optimization', linewidth=2, markersize=8)

        ax.set_xlabel("Taux de modification dynamique (%)", fontsize=11)
        ax.set_ylabel("Nombre moyen de cellules reaffectees", fontsize=11)
        ax.set_title(f"E8 - Convergence en environnement instable - {model_name}",
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        y_min = min(changes_warm_means + changes_cold_means) - 2
        y_max = max(changes_warm_means + changes_cold_means) + 2
        ax.set_ylim([max(y_min, 0), y_max + 5])
        fig.subplots_adjust(top=0.90, bottom=0.12, left=0.10, right=0.97)
        fname = f"adaptation_{model_name}.png"
        fig.savefig(str(model_fig_dir / fname), dpi=300)
        print(f"[INFO] Figure E8/{folder}/{fname} saved.")

    _validate_if_available("E8")
    print("[SUCCESS] Experiment E8 completed.")


# ============================================================================
# E9 - MINIMA LOCAUX
# ============================================================================
def run_experiment_E9(verbose=False):
    plt.close('all')
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

            for num_restarts in values:
                start = time.perf_counter()
                x_bd, _, _, n_cci, n_aci, _ = method_bd(
                    N, K, W, M=M,
                    num_restarts=num_restarts,
                    max_sweeps=config.MAX_SWEEPS_BD,
                    random_order=True,
                    seed=seed + num_restarts * 1000,
                    verbose=False
                )
                elapsed = time.perf_counter() - start
                cost = _compute_cost(x_bd, W, M)

                raw_data.append({
                    "seed": seed, "method": "BD-CeNN", "value": num_restarts,
                    "cost": cost, "conflicts_cci": n_cci, "conflicts_aci": n_aci,
                    "conflicts_total": n_cci + n_aci, "time": elapsed
                })

            for num_orders in values:
                best_cost = float('inf')
                best_conf_cci, best_conf_aci = 0, 0
                best_time = 0.0
                start_total = time.perf_counter()
                for order_idx in range(num_orders):
                    np.random.seed(seed + order_idx * 100 + num_orders * 1000)
                    order = np.random.permutation(N).tolist()
                    start = time.perf_counter()
                    x_g = method_greedy(N, K, W, order=order, M=M)
                    best_time += (time.perf_counter() - start)
                    cost_g = _compute_cost(x_g, W, M)
                    c_cci, c_aci, _ = _compute_all_conflicts(x_g, W, M)
                    if cost_g < best_cost:
                        best_cost = cost_g
                        best_conf_cci = c_cci
                        best_conf_aci = c_aci

                raw_data.append({
                    "seed": seed, "method": "Greedy", "value": num_orders,
                    "cost": best_cost, "conflicts_cci": best_conf_cci,
                    "conflicts_aci": best_conf_aci,
                    "conflicts_total": best_conf_cci + best_conf_aci,
                    "time": time.perf_counter() - start_total
                })

        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(str(model_csv_dir / "raw_metrics.csv"), index=False)
        print(f"[SUCCESS] CSV E9/{folder}/raw_metrics.csv saved.")

        # [FIX-E9-AMELIORATION] Aggregation complete incluant median/min/max
        # pour conflicts_cci et conflicts_aci (necessaire pour la table
        # standalone adjacent en 5 statistiques completes).
        summary = df_raw.groupby(["value", "method"]).agg({
            "cost": ["mean", "std", "median", "min", "max"],
            "conflicts_cci": ["mean", "std", "median", "min", "max"],
            "conflicts_aci": ["mean", "std", "median", "min", "max"],
            "conflicts_total": ["mean", "std", "median", "min", "max"],
            "time": ["mean", "std", "median", "min", "max"]
        }).reset_index()
        summary.columns = [
            'value', 'method',
            'cost_mean', 'cost_std', 'cost_median', 'cost_min', 'cost_max',
            'conflicts_cci_mean', 'conflicts_cci_std',
            'conflicts_cci_median', 'conflicts_cci_min', 'conflicts_cci_max',
            'conflicts_aci_mean', 'conflicts_aci_std',
            'conflicts_aci_median', 'conflicts_aci_min', 'conflicts_aci_max',
            'conflicts_total_mean', 'conflicts_total_std',
            'conflicts_total_median', 'conflicts_total_min', 'conflicts_total_max',
            'time_mean', 'time_std', 'time_median', 'time_min', 'time_max'
        ]
        for col in summary.columns:
            if col not in ['value', 'method']:
                summary[col] = summary[col].round(1) if ('cost' in col or 'conflicts' in col) else summary[col].round(6)

        # [FIX-E9-AMELIORATION] Calcul du Gain(R) et du CostRatio(R)
        # pour chaque methode, par rapport a R=1 (baseline).
        summary = summary.sort_values(["method", "value"]).reset_index(drop=True)
        gains = []
        ratios = []
        for _, row in summary.iterrows():
            ref = summary[
                (summary["method"] == row["method"]) &
                (summary["value"] == 1)
            ]
            if ref.empty:
                gains.append(0.0)
                ratios.append(1.0)
                continue
            J1 = float(ref.iloc[0]["cost_mean"])
            T1 = float(ref.iloc[0]["time_mean"])
            JR = float(row["cost_mean"])
            TR = float(row["time_mean"])
            gain = ((J1 - JR) / J1 * 100.0) if J1 > 0 else 0.0
            ratio = (TR / T1) if T1 > 0 else 1.0
            gains.append(round(gain, 1))
            ratios.append(round(ratio, 2))

        summary["gain_pct"] = gains
        summary["cost_ratio"] = ratios

        summary.to_csv(str(model_csv_dir / "summary_metrics.csv"), index=False, float_format='%.6f')
        print(f"[SUCCESS] CSV E9/{folder}/summary_metrics.csv saved.")

        # [FIX-E9-AMELIORATION] Export dedie des metriques de compromis
        trade_off = summary[[
            "value", "method", "cost_mean", "time_mean",
            "gain_pct", "cost_ratio"
        ]].copy()
        trade_off.to_csv(
            str(model_csv_dir / "trade_off_metrics.csv"),
            index=False, float_format="%.4f"
        )
        print(f"[SUCCESS] CSV E9/{folder}/trade_off_metrics.csv saved.")

        # [FIX-LAYOUT] Figure() au lieu de plt.figure().
        fig, canvas = _new_figure((10, 6))
        ax = fig.add_subplot(111)
        bd_data = summary[summary['method'] == 'BD-CeNN'].sort_values('value')
        greedy_data = summary[summary['method'] == 'Greedy'].sort_values('value')

        x_vals = values
        ax.errorbar(x_vals, bd_data['cost_mean'], yerr=bd_data['cost_std'],
                    fmt='o-', capsize=6, color='red', label='BD-CeNN Solver',
                    linewidth=2, markersize=8)
        ax.errorbar(x_vals, greedy_data['cost_mean'], yerr=greedy_data['cost_std'],
                    fmt='s-', capsize=6, color='blue', label='Greedy Permutations',
                    linewidth=2, markersize=8)

        ax.set_xlabel("Nombre de tentatives de reallocation (Restarts pour CeNN, Essais pour Greedy)",
                      fontsize=11)
        ax.set_ylabel("Cout moyen de l'optimum local J(x)", fontsize=11)
        ax.set_title(f"E9 - Robustesse aux minima locaux - {model_name}",
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks(values)
        ax.set_xticklabels([str(v) for v in values])

        fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.97)
        fname = f"minima_locaux_{model_name}.png"
        fig.savefig(str(model_fig_dir / fname), dpi=300)
        print(f"[INFO] Figure E9/{folder}/{fname} saved.")

        # [FIX-E9-AMELIORATION] Figure de compromis qualite / cout computationnel
        # Deux panneaux : Gain(%) vs R et CostRatio vs R, avec Greedy et BD-CeNN.
        fig, canvas = _new_figure((13, 6))

        ax_gain = fig.add_axes([0.06, 0.15, 0.42, 0.75])
        ax_ratio = fig.add_axes([0.55, 0.15, 0.42, 0.75])

        method_colors = {"Greedy": "forestgreen", "BD-CeNN": "firebrick"}
        method_markers = {"Greedy": "s", "BD-CeNN": "D"}

        for method in ["Greedy", "BD-CeNN"]:
            sub = trade_off[trade_off["method"] == method].sort_values("value")
            if sub.empty:
                continue

            ax_gain.plot(
                sub["value"].values, sub["gain_pct"].values,
                color=method_colors[method],
                marker=method_markers[method],
                linewidth=2.2, markersize=9,
                label=method
            )
            ax_ratio.plot(
                sub["value"].values, sub["cost_ratio"].values,
                color=method_colors[method],
                marker=method_markers[method],
                linewidth=2.2, markersize=9,
                label=method
            )

        for ax in (ax_gain, ax_ratio):
            ax.set_xticks(values)
            ax.set_xticklabels([str(v) for v in values])
            ax.set_xlabel("Nombre de restarts R", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(fontsize=10, loc="best")

        ax_gain.set_ylabel("Gain relatif du coût J(x) (%)", fontsize=11)
        ax_gain.set_title(
            r"Gain$(R) = (J_1 - J_R)\,/\,J_1 \times 100$",
            fontsize=12, fontweight="bold"
        )

        ax_ratio.set_ylabel(r"Ratio de temps $T_R / T_1$", fontsize=11)
        ax_ratio.set_title(
            r"CostRatio$(R) = T_R / T_1$",
            fontsize=12, fontweight="bold"
        )

        fig.suptitle(
            f"E9 — Compromis qualité de solution / coût computationnel ({model_name})",
            fontsize=13, fontweight="bold", y=0.97
        )

        fname = f"trade_off_{model_name}.png"
        fig.savefig(str(model_fig_dir / fname), dpi=300)
        print(f"[INFO] Figure E9/{folder}/{fname} saved.")

    _validate_if_available("E9")
    print("[SUCCESS] Experiment E9 completed.")


# ============================================================================
# E10 - AUDIT DE FIDELITE NEURO-SYMBOLIQUE
# ============================================================================
def run_experiment_E10(verbose=False):
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
            x_bd, history_bd, _, n_cci, n_aci, best_sweep = bdcenn_allocation(
                N, K, W, M=M,
                num_restarts=num_restarts,
                max_sweeps=config.MAX_SWEEPS_BD,
                random_order=True,
                seed=seed,
                verbose=False,
            )
            time_bd = time.perf_counter() - start_t

            if history_bd:
                alloc_init = history_bd[0][2]
                cost_init = _compute_cost(alloc_init, W, M)
            else:
                cost_init = 0.0

            cost_final = _compute_cost(x_bd, W, M)
            n_cci_f, n_aci_f, n_total_f = _compute_all_conflicts(x_bd, W, M)
            used_channels = len(set(x_bd))

            np.random.seed(seed)
            x_rand = random_allocation(N, K)
            np.random.seed(seed)
            order_greedy = np.random.permutation(N).tolist()
            x_greedy = greedy_allocation(N, K, W, order=order_greedy, M=M)
            x_dsatur = dsatur_allocation(N, K, W, M=M)

            baselines = {}
            for name, x_b, t_b in [("Random", x_rand, 0.0), ("Greedy", x_greedy, 0.0), ("DSATUR", x_dsatur, 0.0)]:
                c = _compute_cost(x_b, W, M)
                c_cci_b, c_aci_b, _ = _compute_all_conflicts(x_b, W, M)
                baselines[name] = {
                    "cost": float(c),
                    "conflicts_cci": int(c_cci_b),
                    "conflicts_aci": int(c_aci_b),
                    "conflicts_total": int(c_cci_b + c_aci_b),
                    "time": float(t_b)
                }

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
            cell_summary = [{"cell": int(idx),
                             "degree": int(cell_degree[idx]),
                             "strength": int(cell_strength[idx])} for idx in top_cells_idx]

            threshold_value = float(inst["threshold"]) if (inst is not None and "threshold" in inst) else 0.0
            topology_meta = {"seed": int(seed), "threshold": threshold_value, "N": int(N), "K": int(K)}

            model_label = "CCI-only" if model_name == "cochannel" else "CCI+ACI"

            case_data = {
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "N": N, "K": K, "seed": seed,
                "context": case["context"] + f" [{model_label}]",
                "model_label": model_label,
                "metrics": {
                    "cost_initial": float(cost_init),
                    "cost_final": float(cost_final),
                    "conflicts": int(n_total_f),
                    "conflicts_cci": int(n_cci_f),
                    "conflicts_aci": int(n_aci_f),
                    "time_seconds": float(time_bd),
                    "iterations": int(best_sweep),
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
        df_eval.to_csv(str(model_csv_dir / "llm_fidelity_evaluation.csv"), index=False, float_format="%.2f")
        print(f"[SUCCESS] CSV E10/{folder}/llm_fidelity_evaluation.csv saved.")

    _validate_if_available("E10")
    print("\n" + "=" * 80)
    print("[SUCCESS] GLOBAL EXPERIMENT E10 COMPLETED.")
    print("=" * 80)


# ============================================================================
# MASTER ORCHESTRATOR
# ============================================================================
def run_all_experiments(verbose=False):
    print("\n[START] Executing standard analysis pipeline...")
    run_experiment_E1(verbose)
    run_experiment_E2(verbose)
    run_experiment_E3(verbose)
    run_experiment_E4(verbose)
    run_experiment_E5(verbose)
    run_experiment_E6(verbose)
    run_experiment_E7(verbose)
    run_experiment_E8(verbose)
    run_experiment_E9(verbose)
    run_experiment_E10(verbose)
    print("\n[FINISH] All execution pipelines fully completed.")


if __name__ == "__main__":
    run_all_experiments()