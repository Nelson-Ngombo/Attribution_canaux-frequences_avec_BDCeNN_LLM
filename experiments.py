# experiments.py

import numpy as np
import pandas as pd
import matplotlib          # ← AJOUTER cette ligne
import matplotlib.pyplot as plt
import os
import time
from llm_assistant import audit_case, LLM_REPORTS_DIR, LLM_LOGS_DIR
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
# E1 – VÉRIFICATION VISUELLE (Petit graphe) - VERSION AMÉLIORÉE
# ============================================================================
def run_experiment_E1(verbose=False):
    """
    E1 - Vérification visuelle
    Produit les sorties obligatoires :
        - Graphe d'interférence avec cellules numérotées
        - Matrice W
        - Pour chaque méthode (Random, Greedy, DSATUR, BD-CeNN) :
            - Graphe colorié selon le canal attribué
            - BD-CeNN : affectation initiale et finale
        - Tables CSV : (1) cellule -> canal par méthode, (2) métriques par méthode
        - Tout cela pour deux modèles : co-canal seulement (sans M) et adjacent (avec M)
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Chargement du scénario S1 (seed=1)
    scenario_name = "S1"
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W = np.array(data["W"])
    positions = data["positions"]
    G = data["graph"]
    seed = data["seed"]  # seed=1

    # Palette de couleurs pour les canaux
    channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = {c: channel_colors[c % len(channel_colors)] for c in range(K)}
    pos_dict = {i: tuple(positions[i]) for i in range(N)}

    # Dossiers de sortie
    e1_fig_dir = config.FIGURES_DIR / "E1"
    e1_csv_dir = config.CSV_DIR / "E1"
    os.makedirs(e1_fig_dir, exist_ok=True)
    os.makedirs(e1_csv_dir, exist_ok=True)

    # ================================================================
    # 1. Figures communes : Graphe d'interférence nu et matrice W
    # ================================================================
    # Graphe d'interférence avec numéros de cellules
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos_dict, ax=ax, node_color='lightblue', node_size=500, edgecolors='black', linewidths=1)
    nx.draw_networkx_edges(G, pos_dict, ax=ax, edge_color='gray', width=2)
    nx.draw_networkx_labels(G, pos_dict, ax=ax, font_size=10, font_weight='bold')
    ax.set_title(f"Scénario {scenario_name} - Graphe d'interférence (N={N}, K={K}, seed={seed})", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(e1_fig_dir / "interference_graph.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Figure E1/interference_graph.png sauvegardée.")

    # Matrice W
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(W, cmap='Reds', interpolation='nearest', vmin=0, vmax=4)
    for i in range(N):
        for j in range(N):
            if W[i, j] > 0:
                ax.text(j, i, int(W[i, j]), ha='center', va='center', color='black', fontsize=9, fontweight='bold')
    plt.colorbar(im, label="Niveau d'interférence", shrink=0.8)
    ax.set_title(f"Scénario {scenario_name} - Matrice W (N={N}, seed={seed})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Cellule j")
    ax.set_ylabel("Cellule i")
    plt.tight_layout()
    plt.savefig(e1_fig_dir / "matrix_W.png", dpi=300)
    plt.close(fig)
    print("✅ Figure E1/matrix_W.png sauvegardée.")

    # ================================================================
    # Fonctions auxiliaires pour dessiner les graphes coloriés
    # ================================================================
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
        legend_elements = [Patch(facecolor=colors[c], edgecolor='black', label=f'Canal {c}') for c in range(K)]
        fig.legend(handles=legend_elements, loc='lower center', ncol=K, fontsize=10, bbox_to_anchor=(0.5, -0.05))
        plt.suptitle(f"E1 - {fig_dir.name} - {title} - S1 (N={N}, K={K}, seed={seed})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = f"{method_name.lower().replace(' ', '_')}{label_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(fig_dir / subfolder / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Figure E1/{subfolder}/{filename} sauvegardée.")

    # ================================================================
    # 2. Exécution pour les deux modèles d'interférence
    # ================================================================
    models = [
        {"name": "cochannel", "M": None, "suffix": " (co-canal)", "folder": "cochannel"},
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

        # Exécution des méthodes
        np.random.seed(seed)
        x_rand = random_allocation(N, K)

        x_greedy = greedy_allocation(N, K, W, M=M)

        x_dsatur = dsatur_allocation(N, K, W, M=M)

        # BD-CeNN : on récupère l'historique pour avoir l'état initial et final
        x_bd_final, history_bd, t_bd, conf_bd, iter_best = bdcenn_allocation(
            N, K, W, M=M,
            num_restarts=config.NUM_RESTARTS,
            max_iter=config.MAX_ITER_BD,
            random_order=True,
            seed=seed,
            verbose=False
        )
        # État initial = première allocation de l'historique (itération 0)
        x_bd_initial = history_bd[0][2]

        # Calcul des métriques pour les tables
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

        # --- 2.1 Graphes coloriés ---
        save_colored_graph("Random", x_rand, e1_fig_dir, folder, suffix)
        save_colored_graph("Greedy", x_greedy, e1_fig_dir, folder, suffix)
        save_colored_graph("DSATUR", x_dsatur, e1_fig_dir, folder, suffix)
        save_colored_graph("BD-CeNN initial", x_bd_initial, e1_fig_dir, folder, suffix + " (initial)")
        save_colored_graph("BD-CeNN final", x_bd_final, e1_fig_dir, folder, suffix + " (final)")

        # --- 2.2 Table 1 : Cellule -> Canal par méthode ---
        df_cell_channels = pd.DataFrame({
            "Cellule": list(range(N)),
            "Random": x_rand,
            "Greedy": x_greedy,
            "DSATUR": x_dsatur,
            "BD-CeNN_initial": x_bd_initial,
            "BD-CeNN_final": x_bd_final
        })
        df_cell_channels.to_csv(model_csv_dir / "cell_channels.csv", index=False)
        print(f"✅ CSV E1/{folder}/cell_channels.csv sauvegardé.")

        # --- 2.3 Table 2 : Métriques par méthode ---
        df_metrics = pd.DataFrame({
            "Méthode": ["Random", "Greedy", "DSATUR", "BD-CeNN_initial", "BD-CeNN_final"],
            "Coût global": [cost_rand, cost_greedy, cost_dsatur, cost_bd_initial, cost_bd],
            "Conflits": [conf_rand, conf_greedy, conf_dsatur, conf_bd_initial, conf_bd],
            "Canaux utilisés": [used_rand, used_greedy, used_dsatur, len(set(x_bd_initial)), used_bd]
        })
        df_metrics.to_csv(model_csv_dir / "method_metrics.csv", index=False)
        print(f"✅ CSV E1/{folder}/method_metrics.csv sauvegardé.")

    print("✅ E1 terminée.")


# ============================================================================
# E2 – COMPARAISON GLOBALE (Campagne factorielle)
# ============================================================================
def run_experiment_E2(verbose=False):
    """
    E2 - Comparaison globale (campagne factorielle)
    Étudie l'impact de N, K et de la densité sur les performances des 4 méthodes.
    Les configurations sont : N ∈ {30, 50}, K ∈ {4, 6}, densité ∈ {Faible, Moyenne, Forte}.
    Pour chaque combinaison (N, K, densité), on exécute les 4 méthodes sur les 30 topologies
    (seeds 1 à 30) du scénario correspondant (S2 pour N=30, S3 pour N=50).
    Les métriques sont moyennées sur les seeds.
    Les résultats sont séparés pour co-canal (sans M) et adjacent (avec M).
    Produit :
        - CSV bruts et résumés par modèle
        - Figures condensées : 4 figures par modèle (coût, conflits, canaux utilisés, temps)
          avec une grille 2x2 pour les combinaisons (N, K), chaque sous-figure montrant
          les 4 méthodes avec barres d'erreur pour les 3 densités.
    """
    import networkx as nx
    import time

    # Charger les instances des scénarios S2 et S3
    sc_instances = {}
    for sc_name in ["S2", "S3"]:
        inst = all_instances.get(sc_name, {})
        if not inst:
            print(f"❌ Scénario {sc_name} introuvable dans scenarios_data.json")
            return
        sc_instances[sc_name] = inst

    seeds = [int(s) for s in list(sc_instances["S2"].keys()) if s.isdigit()]
    seeds = sorted(seeds)  # 1..30

    # Définition des 12 configurations : (N, K, scenario_name, area)
    configs = [
        {"N": 30, "K": 4, "scenario": "S2", "area": 150},
        {"N": 30, "K": 6, "scenario": "S2", "area": 150},
        {"N": 50, "K": 4, "scenario": "S3", "area": 200},
        {"N": 50, "K": 6, "scenario": "S3", "area": 200}
    ]

    # Densités
    density_configs = [
        {"label": "Faible", "threshold": 30},
        {"label": "Moyenne", "threshold": 50},
        {"label": "Forte", "threshold": 70}
    ]

    # Dossiers de sortie
    e2_fig_dir = config.FIGURES_DIR / "E2"
    e2_csv_dir = config.CSV_DIR / "E2"
    os.makedirs(e2_fig_dir, exist_ok=True)
    os.makedirs(e2_csv_dir, exist_ok=True)

    # Modèles d'interférence
    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}  # M défini dans la boucle
    ]

    # Méthodes
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

        # Boucle sur les configurations (N, K)
        for cfg in configs:
            N = cfg["N"]
            K = cfg["K"]
            sc_name = cfg["scenario"]
            area = cfg["area"]
            inst_sc = sc_instances[sc_name]

            # Boucle sur les densités
            for dens in density_configs:
                th = dens["threshold"]
                label = dens["label"]

                # Créer M pour ce K (si modèle adjacent)
                if model_name == "adjacent":
                    M = create_channel_interference_matrix(K)
                else:
                    M = None

                # Boucle sur les seeds
                for seed in seeds:
                    inst = inst_sc.get(str(seed))
                    if inst is None:
                        continue
                    positions = np.array(inst["positions"])
                    # Recalculer W avec le nouveau threshold
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

                    # Exécuter chaque méthode
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
                        else:  # BD-CeNN
                            x, _, _, _, best_iter = method_func(
                                N, K, W, M=M,
                                num_restarts=config.NUM_RESTARTS,
                                max_iter=config.MAX_ITER_BD,
                                random_order=True,
                                seed=seed,
                                verbose=False
                            )
                        elapsed = time.perf_counter() - start_time

                        # Calcul des métriques selon le modèle
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

        # Convertir en DataFrame
        df_raw = pd.DataFrame(raw_data)

        # Sauvegarder les données brutes
        raw_csv_path = model_csv_dir / "raw_metrics.csv"
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"✅ CSV E2/{folder}/raw_metrics.csv sauvegardé.")

        # Calcul du résumé par (N, K, density_label, threshold, method)
        summary = df_raw.groupby(["N", "K", "density_label", "threshold", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"],
            "iterations": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        # Aplatir les colonnes
        summary.columns = ['N', 'K', 'density_label', 'threshold', 'method',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
                           'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']

        # Arrondir les colonnes appropriées
        for col in ['conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                    'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']:
            summary[col] = summary[col].round(1)

        # Convertir en entier les colonnes de conflits et canaux utilisés (moyennes, min, max, median)
        for col in ['conflicts_mean', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_min', 'used_channels_max', 'used_channels_median']:
            summary[col] = summary[col].fillna(0).round(0).astype(int)

        # Sauvegarder le résumé
        summary_csv_path = model_csv_dir / "summary_metrics.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"✅ CSV E2/{folder}/summary_metrics.csv sauvegardé.")

        # ================================================================
        # NOUVELLE SECTION : FIGURES CONDENSÉES (4 figures par modèle)
        # ================================================================

        # Définition des métriques à tracer
        metrics_to_plot = [
            ('cost', 'Coût global moyen J(x)'),
            ('conflicts', 'Conflits moyens'),
            ('used_channels', 'Canaux utilisés moyens'),
            ('time', "Temps d'exécution moyen (s)")
        ]

        # Combinaisons (N, K) pour les sous-graphiques
        subplots_config = [
            (30, 4, "N=30, K=4"),
            (30, 6, "N=30, K=6"),
            (50, 4, "N=50, K=4"),
            (50, 6, "N=50, K=6")
        ]

        # Couleurs des méthodes
        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}

        # Pour chaque métrique, créer une figure avec 4 sous-graphiques (2x2)
        for metric_col, ylabel in metrics_to_plot:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            # Aplatir les axes pour itérer facilement
            axes_flat = axes.flatten()

            # Pour chaque sous-graphique, on extrait les données correspondantes
            for ax, (N, K, title) in zip(axes_flat, subplots_config):
                # Filtrer les données pour ce N, K
                sub = summary[(summary['N'] == N) & (summary['K'] == K)]
                if sub.empty:
                    ax.set_title(f"{title} (pas de données)")
                    ax.axis('off')
                    continue

                # Réindexer par densité pour avoir l'ordre (Faible, Moyenne, Forte)
                density_order = [d["label"] for d in density_configs]
                # Grouper par méthode et densité
                # Créer un DataFrame pivot pour les moyennes et écarts-types
                pivot_mean = sub.pivot(index='density_label', columns='method', values=f'{metric_col}_mean')
                pivot_std = sub.pivot(index='density_label', columns='method', values=f'{metric_col}_std')
                # Réindexer pour l'ordre des densités
                pivot_mean = pivot_mean.reindex(density_order)
                pivot_std = pivot_std.reindex(density_order)

                # Paramètres du barplot
                x = np.arange(len(density_order))
                width = 0.2
                # Pour chaque méthode, tracer les barres
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
                ax.set_xticklabels(density_order, rotation=15, fontsize=8)
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=9)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                # Ajuster l'échelle pour les canaux utilisés (valeurs entières)
                if metric_col == 'used_channels':
                    # Forcer les ticks à des entiers
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Ajouter une légende commune (en dessous ou à droite)
            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(methods), fontsize=10, bbox_to_anchor=(0.5, -0.02))
            # Titre général
            fig.suptitle(f"E2 - {ylabel} - {model_name}", fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # laisser de la place pour la légende
            # Sauvegarde
            fname = f"E2_{metric_col}_{model_name}.png"
            plt.savefig(model_fig_dir / fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Figure E2/{folder}/{fname} sauvegardée.")

    print("✅ E2 terminée.")



# ============================================================================
# E3 – IMPACT DU NOMBRE DE CANAUX (K) - VERSION CORRIGÉE
# ============================================================================

def run_experiment_E3(verbose=False):
    """
    E3 - Impact du nombre de canaux (K)
    Mesure l'évolution du coût, des conflits et des canaux utilisés en fonction de K.
    Pour chaque K, on exécute les 4 méthodes sur les 30 topologies (seeds 1 à 30)
    du scénario S4 (N=50). Les métriques sont moyennées sur les seeds.
    Les résultats sont séparés pour co-canal (sans M) et adjacent (avec M).
    Produit :
        - CSV bruts et résumés par modèle
        - Figures : coût moyen vs K et conflits moyens vs K avec barres d'erreur
    """
    import networkx as nx  # utilisé pour reconstruire les graphes

    # Charger les instances du scénario S4 (N=50)
    scenario_name = "S4"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"❌ Scénario {scenario_name} introuvable dans scenarios_data.json")
        return

    # Paramètres fixes de S4
    first_inst = instances.get("1") or list(instances.values())[0]
    N = first_inst["N"]          # 50
    K_values = [2, 3, 4, 6, 8]   # K variable
    seeds = [int(s) for s in instances.keys() if s.isdigit()]
    # Trier les seeds
    seeds = sorted(seeds)

    # Dossiers de sortie
    e3_fig_dir = config.FIGURES_DIR / "E3"
    e3_csv_dir = config.CSV_DIR / "E3"
    os.makedirs(e3_fig_dir, exist_ok=True)
    os.makedirs(e3_csv_dir, exist_ok=True)

    # Modèles d'interférence
    models = [
        {"name": "cochannel", "suffix": "_co", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "suffix": "_adj", "M": None, "folder": "adjacent"}  # M sera défini dans la boucle K
    ]

    # Méthodes et leurs fonctions
    methods = {
        "Random": random_allocation,
        "Greedy": greedy_allocation,
        "DSATUR": dsatur_allocation,
        "BD-CeNN": bdcenn_allocation
    }

    # Pour chaque modèle
    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e3_fig_dir / folder
        model_csv_dir = e3_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        # Liste pour stocker les données brutes
        raw_data = []

        # Boucle sur K
        for K in K_values:
            # Créer la matrice M pour ce K (si modèle adjacent)
            if model_name == "adjacent":
                M = create_channel_interference_matrix(K)
            else:
                M = None

            # Boucle sur les seeds
            for seed in seeds:
                # Charger la topologie pour cette seed
                inst = instances.get(str(seed))
                if inst is None:
                    continue
                W = np.array(inst["W"])

                # Exécuter chaque méthode
                for method_name, method_func in methods.items():
                    if method_name == "Random":
                        np.random.seed(seed)
                        x = method_func(N, K)   # Random ne prend pas M
                    elif method_name == "Greedy":
                        np.random.seed(seed)
                        order = np.random.permutation(N).tolist()
                        x = method_func(N, K, W, order=order, M=M)
                    elif method_name == "DSATUR":
                        x = method_func(N, K, W, M=M)
                    else:  # BD-CeNN
                        x, _, _, _, _ = method_func(
                            N, K, W, M=M,
                            num_restarts=config.NUM_RESTARTS,
                            max_iter=config.MAX_ITER_BD,
                            random_order=True,
                            seed=seed,
                            verbose=False
                        )

                    # Calcul des métriques selon le modèle
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

        # Convertir en DataFrame
        df_raw = pd.DataFrame(raw_data)

        # Sauvegarder les données brutes
        raw_csv_path = model_csv_dir / "raw_metrics.csv"
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"✅ CSV E3/{folder}/raw_metrics.csv sauvegardé.")

        # Calcul du résumé par K et méthode
        summary = df_raw.groupby(["K", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        # Aplatir les colonnes
        summary.columns = ['K', 'method', 
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median']
        # Arrondir les colonnes appropriées
        for col in ['conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median']:
            summary[col] = summary[col].round(1)  # on garde une décimale pour std
        # Pour les conflits et canaux, on arrondit à l'entier
        for col in ['conflicts_mean', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_min', 'used_channels_max', 'used_channels_median']:
            summary[col] = summary[col].round(0).astype(int)

        # Sauvegarder le résumé
        summary_csv_path = model_csv_dir / "summary_metrics.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"✅ CSV E3/{folder}/summary_metrics.csv sauvegardé.")

        # --- Figures ---
        # Préparer les données pour les figures
        pivot_cost = summary.pivot(index='K', columns='method', values=['cost_mean', 'cost_std'])
        pivot_conflicts = summary.pivot(index='K', columns='method', values=['conflicts_mean', 'conflicts_std'])

        # Figure 1 : Coût global moyen vs K avec barres d'erreur
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
        method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}

        for method in methods.keys():
            means = pivot_cost[('cost_mean', method)].values
            stds = pivot_cost[('cost_std', method)].values
            ax1.errorbar(K_values, means, yerr=stds, 
                         label=method, color=method_colors[method], marker=method_markers[method],
                         capsize=5, linewidth=2, markersize=8)

        ax1.set_xlabel("Nombre de canaux K", fontsize=12)
        ax1.set_ylabel("Coût global moyen J(x)", fontsize=12)
        ax1.set_title(f"E3 - Impact de K sur le coût moyen - {model_name}", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        fig1_filename = model_fig_dir / f"cost_vs_K_{model_name}.png"
        plt.savefig(fig1_filename, dpi=300)
        plt.close(fig1)
        print(f"✅ Figure E3/{folder}/cost_vs_K_{model_name}.png sauvegardée.")

        # Figure 2 : Conflits moyens vs K avec barres d'erreur
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for method in methods.keys():
            means = pivot_conflicts[('conflicts_mean', method)].values
            stds = pivot_conflicts[('conflicts_std', method)].values
            ax2.errorbar(K_values, means, yerr=stds,
                         label=method, color=method_colors[method], marker=method_markers[method],
                         capsize=5, linewidth=2, markersize=8)

        ax2.set_xlabel("Nombre de canaux K", fontsize=12)
        ax2.set_ylabel("Conflits moyens", fontsize=12)
        ax2.set_title(f"E3 - Impact de K sur les conflits moyens - {model_name}", fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        fig2_filename = model_fig_dir / f"conflicts_vs_K_{model_name}.png"
        plt.savefig(fig2_filename, dpi=300)
        plt.close(fig2)
        print(f"✅ Figure E3/{folder}/conflicts_vs_K_{model_name}.png sauvegardée.")

    print("✅ E3 terminée.")


# ============================================================================
# E4 – IMPACT DE LA DENSITÉ - VERSION CORRIGÉE AVEC SÉPARATION COCANAL/ADJACENT
# ============================================================================
def run_experiment_E4(verbose=False):
    """
    E4 - Impact de la densité du graphe sur les performances.
    Utilise le scénario S3 (N=50, K=6) avec trois niveaux de densité :
        - Faible (threshold=30)
        - Moyenne (threshold=50)
        - Forte (threshold=70)
    Pour chaque densité, on exécute les 4 méthodes sur les 30 topologies (seeds 1 à 30)
    du scénario S3. Les métriques sont moyennées sur les seeds.
    Les résultats sont séparés pour co-canal (sans M) et adjacent (avec M).
    Produit :
        - CSV bruts et résumés par modèle
        - Figure : coût moyen vs densité avec barres d'erreur
    """
    import networkx as nx
    import time

    # Charger les instances du scénario S3 (N=50, K=6)
    scenario_name = "S3"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"❌ Scénario {scenario_name} introuvable dans scenarios_data.json")
        return

    # Paramètres fixes de S3
    first_inst = instances.get("1") or list(instances.values())[0]
    N = first_inst["N"]          # 50
    K = first_inst["K"]          # 6
    area = first_inst.get("area", 200)  # on le récupère de config si besoin
    # Les thresholds pour les trois densités
    density_configs = [
        {"label": "Faible", "threshold": 30},
        {"label": "Moyenne", "threshold": 50},
        {"label": "Forte", "threshold": 70}
    ]

    seeds = [int(s) for s in instances.keys() if s.isdigit()]
    seeds = sorted(seeds)  # 1..30

    # Dossiers de sortie
    e4_fig_dir = config.FIGURES_DIR / "E4"
    e4_csv_dir = config.CSV_DIR / "E4"
    os.makedirs(e4_fig_dir, exist_ok=True)
    os.makedirs(e4_csv_dir, exist_ok=True)

    # Modèles d'interférence
    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}  # M sera défini dans la boucle densité
    ]

    # Méthodes et leurs fonctions
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

        # Boucle sur les densités
        for dens in density_configs:
            th = dens["threshold"]
            label = dens["label"]

            # Boucle sur les seeds
            for seed in seeds:
                inst = instances.get(str(seed))
                if inst is None:
                    continue
                # Positions fixes pour cette seed
                positions = np.array(inst["positions"])
                # Recalculer W avec le nouveau threshold
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

                # Créer M pour ce K (si modèle adjacent)
                if model_name == "adjacent":
                    M = create_channel_interference_matrix(K)
                else:
                    M = None

                # Exécuter chaque méthode
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
                    else:  # BD-CeNN
                        x, _, _, _, best_iter = method_func(
                            N, K, W, M=M,
                            num_restarts=config.NUM_RESTARTS,
                            max_iter=config.MAX_ITER_BD,
                            random_order=True,
                            seed=seed,
                            verbose=False
                        )
                    elapsed = time.perf_counter() - start_time

                    # Calcul des métriques selon le modèle
                    if M is None:
                        cost = compute_cochannel_cost(x, W)
                        conflicts = count_cochannel_conflicts(x, W)
                    else:
                        cost = compute_adjacent_cost(x, W, M)
                        conflicts = count_adjacent_conflicts(x, W, M)
                    used_channels = len(set(x))

                    # Pour BD-CeNN, on récupère l'itération du meilleur coût
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

        # Convertir en DataFrame
        df_raw = pd.DataFrame(raw_data)

        # Sauvegarder les données brutes
        raw_csv_path = model_csv_dir / "raw_metrics.csv"
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"✅ CSV E4/{folder}/raw_metrics.csv sauvegardé.")

        # Calcul du résumé par densité et méthode
        summary = df_raw.groupby(["density_label", "threshold", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"],
            "iterations": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        # Aplatir les colonnes
        summary.columns = ['density_label', 'threshold', 'method',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
                           'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']

        # Arrondir les colonnes appropriées (sauf les itérations qui contiennent des NaN)
        # Pour les conflits et canaux utilisés, on arrondit à 1 décimale puis à l'entier pour les moyennes/min/max/median
        for col in ['conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median']:
            summary[col] = summary[col].round(1)
            if 'std' not in col:
                summary[col] = summary[col].round(0).astype(int)

        # Pour les itérations, on les arrondit à 1 décimale (sans conversion en int pour éviter les NaN)
        for col in ['iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']:
            summary[col] = summary[col].round(1)

        # Sauvegarder le résumé
        summary_csv_path = model_csv_dir / "summary_metrics.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"✅ CSV E4/{folder}/summary_metrics.csv sauvegardé.")

        # --- Figure : coût moyen vs densité avec barres d'erreur ---
        # Préparer les données : pivot pour avoir les méthodes en colonnes
        pivot_cost = summary.pivot(index='threshold', columns='method', values=['cost_mean', 'cost_std'])
        # Les thresholds sont 30, 50, 70; on les trie
        x_vals = sorted(summary['threshold'].unique())
        # Labels pour l'axe des x
        x_labels = [f"{th}" for th in x_vals]
        # On peut ajouter le label de densité en dessous

        fig, ax = plt.subplots(figsize=(10, 6))
        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
        method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}

        for method in methods.keys():
            means = pivot_cost[('cost_mean', method)].values
            stds = pivot_cost[('cost_std', method)].values
            ax.errorbar(x_vals, means, yerr=stds,
                         label=method, color=method_colors[method], marker=method_markers[method],
                         capsize=5, linewidth=2, markersize=8)

        ax.set_xlabel("Seuil d'interférence (threshold) - Densité croissante", fontsize=12)
        ax.set_ylabel("Coût global moyen J(x)", fontsize=12)
        ax.set_title(f"E4 - Impact de la densité sur le coût moyen - {model_name}", fontsize=14, fontweight='bold')
        # Ajouter les labels de densité sous l'axe
        density_labels = [f"{label}" for label, th in zip([d["label"] for d in density_configs], x_vals)]
        # Décaler les ticks
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{th}" for th in x_vals])
        # Ajouter une deuxième ligne de ticks pour les labels de densité
        ax2 = ax.twiny()
        ax2.set_xticks(x_vals)
        ax2.set_xticklabels(density_labels)
        ax2.set_xlabel("Niveau de densité")
        ax2.xaxis.set_label_position('bottom')
        ax2.xaxis.tick_bottom()
        # Désactiver les ticks de ax2 pour éviter le chevauchement
        ax2.xaxis.set_ticks_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 40))

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        fig1_filename = model_fig_dir / f"cost_vs_density_{model_name}.png"
        plt.savefig(fig1_filename, dpi=300)
        plt.close(fig)
        print(f"✅ Figure E4/{folder}/cost_vs_density_{model_name}.png sauvegardée.")

    print("✅ E4 terminée.")


# ============================================================================
# E6 – SCALABILITÉ (temps, coût, coût normalisé, itérations vs N)
# ============================================================================
def run_experiment_E6(verbose=False):
    """
    E6 - Scalabilité
    Étudie l'impact de la taille du réseau N sur les performances des 4 méthodes.
    N ∈ {20, 30, 50, 100, 200}, K = 8 fixe.
    Pour chaque N, on exécute les 4 méthodes sur les 30 topologies (seeds 1 à 30)
    générées avec les paramètres (area=300, threshold=50).
    Les métriques sont moyennées sur les seeds.
    Les résultats sont séparés pour co-canal (sans M) et adjacent (avec M).
    Produit :
        - CSV bruts et résumés par modèle
        - Figures : coût moyen vs N, temps moyen vs N, itérations moyennes vs N
          avec barres d'erreur.
    """
    import networkx as nx
    import time

    # Paramètres
    K = 8
    area = 300
    threshold = 50
    N_values = [20, 30, 50, 100, 200]
    seeds = list(range(1, config.NUM_RUNS + 1))  # 1..30

    # Dossiers de sortie
    e6_fig_dir = config.FIGURES_DIR / "E6"
    e6_csv_dir = config.CSV_DIR / "E6"
    os.makedirs(e6_fig_dir, exist_ok=True)
    os.makedirs(e6_csv_dir, exist_ok=True)

    # Modèles d'interférence
    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}  # M défini dans la boucle
    ]

    # Méthodes
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

        # Boucle sur N
        for N in N_values:
            # Créer M pour ce K (si modèle adjacent)
            if model_name == "adjacent":
                M = create_channel_interference_matrix(K)
            else:
                M = None

            # Boucle sur les seeds
            for seed in seeds:
                # Générer la topologie pour ce N, seed
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
                # Somme des poids pour normalisation
                sum_weights = np.sum(W)

                # Exécuter chaque méthode
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
                    else:  # BD-CeNN
                        x, _, _, _, best_iter = method_func(
                            N, K, W, M=M,
                            num_restarts=config.NUM_RESTARTS,
                            max_iter=config.MAX_ITER_BD,
                            random_order=True,
                            seed=seed,
                            verbose=False
                        )
                    elapsed = time.perf_counter() - start_time

                    # Calcul des métriques selon le modèle
                    if M is None:
                        cost = compute_cochannel_cost(x, W)
                        conflicts = count_cochannel_conflicts(x, W)
                    else:
                        cost = compute_adjacent_cost(x, W, M)
                        conflicts = count_adjacent_conflicts(x, W, M)
                    used_channels = len(set(x))
                    # Coût normalisé
                    if sum_weights > 0:
                        normalized_cost = cost / sum_weights
                    else:
                        normalized_cost = np.nan

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

        # Convertir en DataFrame
        df_raw = pd.DataFrame(raw_data)

        # Sauvegarder les données brutes
        raw_csv_path = model_csv_dir / "raw_metrics.csv"
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"✅ CSV E6/{folder}/raw_metrics.csv sauvegardé.")

        # Calcul du résumé par N et méthode
        summary = df_raw.groupby(["N", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "normalized_cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "used_channels": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"],
            "iterations": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        # Aplatir les colonnes
        summary.columns = ['N', 'method',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'normalized_cost_mean', 'normalized_cost_std', 'normalized_cost_min', 'normalized_cost_max', 'normalized_cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
                           'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']

        # Arrondir les colonnes appropriées
        for col in ['conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                    'iterations_mean', 'iterations_std', 'iterations_min', 'iterations_max', 'iterations_median']:
            summary[col] = summary[col].round(1)

        # Convertir en entier les moyennes, min, max, median
        for col in ['conflicts_mean', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                    'used_channels_mean', 'used_channels_min', 'used_channels_max', 'used_channels_median',
                    'iterations_mean', 'iterations_min', 'iterations_max', 'iterations_median']:
            summary[col] = summary[col].fillna(0).round(0).astype(int)

        # Sauvegarder le résumé
        summary_csv_path = model_csv_dir / "summary_metrics.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"✅ CSV E6/{folder}/summary_metrics.csv sauvegardé.")

        # --- Figures ---
        # On prépare les données pour le tracé : on veut les moyennes et écarts-types par N pour chaque méthode
        # Pour cost_mean, time_mean, iterations_mean (seulement pour BD-CeNN)
        metrics_to_plot = [
            ('cost_mean', 'cost_std', 'Coût global moyen J(x)'),
            ('time_mean', 'time_std', "Temps d'exécution moyen (s)"),
            ('iterations_mean', 'iterations_std', "Itérations moyennes (BD-CeNN)")
        ]

        # Pour les itérations, on ne garde que la méthode BD-CeNN
        for metric_col, std_col, ylabel in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            if 'iterations' in metric_col:
                # On filtre uniquement BD-CeNN
                sub = summary[summary['method'] == 'BD-CeNN']
                # On s'assure que les N sont dans l'ordre
                sub = sub.set_index('N').reindex(N_values).reset_index()
                x = sub['N'].values
                means = sub[metric_col].values
                stds = sub[std_col].values
                # On peut ajouter les autres méthodes si on veut, mais la consigne demande seulement BD-CeNN pour les itérations
                # On affiche avec un style
                ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=5,
                            color='firebrick', label='BD-CeNN', linewidth=2, markersize=8)
                ax.set_title(f"E6 - Scalabilité - {ylabel} (BD-CeNN)")
            else:
                # Pour cost et time, on affiche les 4 méthodes
                method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
                method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}
                for method in methods.keys():
                    sub = summary[summary['method'] == method]
                    sub = sub.set_index('N').reindex(N_values).reset_index()
                    x = sub['N'].values
                    means = sub[metric_col].values
                    stds = sub[std_col].values
                    ax.errorbar(x, means, yerr=stds, fmt=method_markers[method]+'-', capsize=5,
                                color=method_colors[method], label=method, linewidth=2, markersize=8)
                ax.set_title(f"E6 - Scalabilité - {ylabel}")

            ax.set_xlabel("Nombre de cellules N", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            # Nom du fichier
            fname = f"{metric_col.replace('_mean','')}_vs_N_{model_name}.png"
            plt.savefig(model_fig_dir / fname, dpi=300)
            plt.close(fig)
            print(f"✅ Figure E6/{folder}/{fname} sauvegardée.")

        # Figure supplémentaire : coût normalisé moyen vs N
        # On le fait également
        fig, ax = plt.subplots(figsize=(10, 6))
        method_colors = {'Random': 'royalblue', 'Greedy': 'forestgreen', 'DSATUR': 'darkorange', 'BD-CeNN': 'firebrick'}
        method_markers = {'Random': 'o', 'Greedy': 's', 'DSATUR': '^', 'BD-CeNN': 'D'}
        for method in methods.keys():
            sub = summary[summary['method'] == method]
            sub = sub.set_index('N').reindex(N_values).reset_index()
            x = sub['N'].values
            means = sub['normalized_cost_mean'].values
            stds = sub['normalized_cost_std'].values
            ax.errorbar(x, means, yerr=stds, fmt=method_markers[method]+'-', capsize=5,
                        color=method_colors[method], label=method, linewidth=2, markersize=8)
        ax.set_xlabel("Nombre de cellules N", fontsize=12)
        ax.set_ylabel("Coût normalisé moyen", fontsize=12)
        ax.set_title(f"E6 - Scalabilité - Coût normalisé moyen - {model_name}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        fname = f"normalized_cost_vs_N_{model_name}.png"
        plt.savefig(model_fig_dir / fname, dpi=300)
        plt.close(fig)
        print(f"✅ Figure E6/{folder}/{fname} sauvegardée.")

    print("✅ E6 terminée.")


# ============================================================================
# E7 – ROBUSTESSE AU BRUIT 
# ============================================================================
def run_experiment_E7(verbose=False):
    """
    E7 - Robustesse au bruit sur la matrice d'interférence W.
    Utilise le scénario S6 (N=50, K=4) avec 30 topologies de référence.
    Pour chaque topologie, on exécute BD-CeNN sur W propre (0% bruit) pour obtenir
    la solution de référence (x_ref, J_ref). Ensuite, pour 3 niveaux de bruit (5%, 10%, 20%),
    on perturbe W et on exécute BD-CeNN pour obtenir x_bruitee.
    Métriques : coût relatif (dégradation) et taux de changement de canaux.
    Les résultats sont séparés pour co-canal (sans M) et adjacent (avec M).
    Produit :
        - CSV bruts et résumés par modèle
        - Figure : deux courbes (coût relatif et taux de changement) vs niveau de bruit,
          avec barres d'erreur, sur un même axe Y en pourcentage (échelle -50% à +100%).
    """
    import networkx as nx
    import time

    # Paramètres du scénario S6
    scenario_name = "S6"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"❌ Scénario {scenario_name} introuvable dans scenarios_data.json")
        return

    first_inst = instances.get("1") or list(instances.values())[0]
    N = first_inst["N"]          # 50
    K = first_inst["K"]          # 4
    area = 200                   # issu de config
    seeds = [int(s) for s in instances.keys() if s.isdigit()]
    seeds = sorted(seeds)  # 1..30

    # Niveaux de bruit (en fraction)
    noise_levels = [0.05, 0.10, 0.20]
    noise_labels = ["5%", "10%", "20%"]

    # Dossiers de sortie
    e7_fig_dir = config.FIGURES_DIR / "E7"
    e7_csv_dir = config.CSV_DIR / "E7"
    os.makedirs(e7_fig_dir, exist_ok=True)
    os.makedirs(e7_csv_dir, exist_ok=True)

    # Modèles d'interférence
    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}  # M défini dans la boucle
    ]

    # Méthodes : on n'utilise que BD-CeNN
    method_func = bdcenn_allocation

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e7_fig_dir / folder
        model_csv_dir = e7_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []  # pour stocker les résultats bruts

        # Pour chaque seed, on va d'abord obtenir la référence (0% bruit)
        # On stocke les références pour chaque seed
        ref_x = {}
        ref_cost = {}
        ref_W = {}

        for seed in seeds:
            # Générer la topologie propre (W propre)
            inst = instances.get(str(seed))
            if inst is None:
                continue
            positions = np.array(inst["positions"])
            W_clean = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    # Utiliser le threshold de S6 (35) pour générer W
                    threshold = 35  # S6 threshold
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

            # Créer M pour ce K (si modèle adjacent)
            if model_name == "adjacent":
                M = create_channel_interference_matrix(K)
            else:
                M = None

            # Exécuter BD-CeNN sur W_clean pour obtenir référence
            x_ref, _, _, _, _ = method_func(
                N, K, W_clean, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=seed,
                verbose=False
            )
            ref_x[seed] = x_ref
            # Calculer le coût de référence selon le modèle
            if M is None:
                cost_ref = compute_cochannel_cost(x_ref, W_clean)
            else:
                cost_ref = compute_adjacent_cost(x_ref, W_clean, M)
            ref_cost[seed] = cost_ref

        # Maintenant, pour chaque niveau de bruit, on génère W_bruitee et on évalue
        for b_noise, noise_label in zip(noise_levels, noise_labels):
            for seed in seeds:
                W_clean = ref_W[seed]
                # Générer W_bruitee en perturbant les coefficients non nuls
                # On utilise une graine déterministe pour chaque (seed, niveau de bruit)
                np.random.seed(seed + int(b_noise * 10000))
                W_noisy = W_clean.copy()
                # On ne perturbe que les coefficients > 0
                mask = W_clean > 0
                # Bruit = W * bruit relatif uniforme dans [-b_noise, b_noise]
                noise_factor = np.random.uniform(-b_noise, b_noise, size=W_clean.shape)
                W_noisy[mask] = W_clean[mask] * (1 + noise_factor[mask])
                # Arrondir et clip à 0, et symétriser
                W_noisy = np.round(W_noisy)
                W_noisy[W_noisy < 0] = 0
                # Symétriser
                for i in range(N):
                    for j in range(i+1, N):
                        W_noisy[j, i] = W_noisy[i, j]
                # Forcer diagonale nulle
                np.fill_diagonal(W_noisy, 0)

                # Créer M pour ce K (si modèle adjacent)
                if model_name == "adjacent":
                    M = create_channel_interference_matrix(K)
                else:
                    M = None

                # Exécuter BD-CeNN sur W_noisy
                # On utilise une seed différente pour le bruit pour ne pas avoir les mêmes initialisations
                seed_noisy = seed + int(b_noise * 1000) + 100
                x_noisy, _, _, _, _ = method_func(
                    N, K, W_noisy, M=M,
                    num_restarts=config.NUM_RESTARTS,
                    max_iter=config.MAX_ITER_BD,
                    random_order=True,
                    seed=seed_noisy,
                    verbose=False
                )
                if M is None:
                    cost_noisy = compute_cochannel_cost(x_noisy, W_noisy)
                else:
                    cost_noisy = compute_adjacent_cost(x_noisy, W_noisy, M)

                # Calcul des métriques
                # Coût relatif (dégradation) en pourcentage
                if ref_cost[seed] > 0:
                    cost_relatif = ((cost_noisy - ref_cost[seed]) / ref_cost[seed]) * 100
                else:
                    cost_relatif = 0.0  # si coût de référence nul, on met 0

                # Taux de changement de canaux
                changes = np.sum(x_noisy != ref_x[seed])
                change_rate = (changes / N) * 100

                raw_data.append({
                    "noise_level": b_noise * 100,  # en pourcentage
                    "seed": seed,
                    "cost_ref": ref_cost[seed],
                    "cost_noisy": cost_noisy,
                    "cost_relatif": cost_relatif,
                    "change_rate": change_rate
                })

        # Convertir en DataFrame
        df_raw = pd.DataFrame(raw_data)

        # Sauvegarder les données brutes (format CSV standard, sans interprétation de date)
        raw_csv_path = model_csv_dir / "raw_metrics.csv"
        df_raw.to_csv(raw_csv_path, index=False, float_format='%.6f')
        print(f"✅ CSV E7/{folder}/raw_metrics.csv sauvegardé.")

        # Calcul du résumé par niveau de bruit
        summary = df_raw.groupby("noise_level").agg({
            "cost_relatif": ["mean", "std", "min", "max", "median"],
            "change_rate": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        # Aplatir les colonnes
        summary.columns = ['noise_level',
                           'cost_relatif_mean', 'cost_relatif_std', 'cost_relatif_min', 'cost_relatif_max', 'cost_relatif_median',
                           'change_rate_mean', 'change_rate_std', 'change_rate_min', 'change_rate_max', 'change_rate_median']
        # Arrondir à 1 décimale (les valeurs sont en pourcentage)
        for col in summary.columns:
            if col != 'noise_level':
                summary[col] = summary[col].round(1)

        # Sauvegarder le résumé (format CSV standard)
        summary_csv_path = model_csv_dir / "summary_metrics.csv"
        summary.to_csv(summary_csv_path, index=False, float_format='%.1f')
        print(f"✅ CSV E7/{folder}/summary_metrics.csv sauvegardé.")

        # --- Figure : deux courbes (coût relatif et taux de changement) vs bruit ---
        fig, ax = plt.subplots(figsize=(10, 6))
        # On ajoute le point (0,0) pour le niveau 0% (référence)
        noise_vals = [0] + sorted(summary['noise_level'].unique())
        cost_means = [0] + summary['cost_relatif_mean'].tolist()
        cost_stds = [0] + summary['cost_relatif_std'].tolist()
        change_means = [0] + summary['change_rate_mean'].tolist()
        change_stds = [0] + summary['change_rate_std'].tolist()

        # Couleurs et marqueurs
        ax.errorbar(noise_vals, cost_means, yerr=cost_stds, fmt='o-', capsize=6,
                    color='red', label='Coût relatif moyen', linewidth=2, markersize=8,
                    elinewidth=2, markeredgewidth=2)
        ax.errorbar(noise_vals, change_means, yerr=change_stds, fmt='s-', capsize=6,
                    color='blue', label='Taux de changement de canaux', linewidth=2, markersize=8,
                    elinewidth=2, markeredgewidth=2)

        ax.set_xlabel("Niveau de bruit (%)", fontsize=12)
        ax.set_ylabel("Pourcentage (%)", fontsize=12)
        ax.set_title(f"E7 - Robustesse au bruit - {model_name}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        # Forcer l'échelle de l'axe Y pour une meilleure visibilité des barres d'erreur
        ax.set_ylim(-50, 100)   # de -50% à +100%

        plt.tight_layout()
        fname = f"robustness_{model_name}.png"
        plt.savefig(model_fig_dir / fname, dpi=300)
        plt.close(fig)
        print(f"✅ Figure E7/{folder}/{fname} sauvegardée.")

    print("✅ E7 terminée.")


# ============================================================================
# E8 – RÉSEAU DYNAMIQUE (Warm Start vs Cold Start)
# ============================================================================
def run_experiment_E8(verbose=False):
    """
    E8 - Réseau dynamique : adaptation du BD-CeNN face à des modifications du réseau.
    Utilise le scénario S7 (N=45, K=5) avec 30 topologies de référence.
    Pour chaque topologie, on exécute BD-CeNN sur W original pour obtenir une solution
    stable (x_init). Ensuite, pour 3 niveaux de modification (5%, 10%, 20%), on perturbe
    W en modifiant aléatoirement des arêtes/poids. On compare deux modes de réoptimisation :
        - Warm start : départ de x_init
        - Cold start : départ aléatoire
    Métriques : temps d'adaptation, nombre de réaffectations, coût final.
    Les résultats sont séparés pour co-canal (sans M) et adjacent (avec M).
    Produit :
        - CSV bruts et résumés par modèle
        - Figure : nombre moyen de canaux réaffectés vs taux de modification,
          avec barres d'erreur, pour warm start et cold start.
    """
    import networkx as nx
    import time

    # Paramètres du scénario S7
    scenario_name = "S7"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"❌ Scénario {scenario_name} introuvable dans scenarios_data.json")
        return

    first_inst = instances.get("1") or list(instances.values())[0]
    N = first_inst["N"]          # 45
    K = first_inst["K"]          # 5
    area = 200                   # issu de config
    threshold = 30               # S7 threshold
    seeds = [int(s) for s in instances.keys() if s.isdigit()]
    seeds = sorted(seeds)  # 1..30

    # Niveaux de modification (en fraction)
    mod_levels = [0.05, 0.10, 0.20]
    mod_labels = ["5%", "10%", "20%"]

    # Dossiers de sortie
    e8_fig_dir = config.FIGURES_DIR / "E8"
    e8_csv_dir = config.CSV_DIR / "E8"
    os.makedirs(e8_fig_dir, exist_ok=True)
    os.makedirs(e8_csv_dir, exist_ok=True)

    # Modèles d'interférence
    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}  # M défini dans la boucle
    ]

    # Méthodes : on n'utilise que BD-CeNN
    method_func = bdcenn_allocation

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e8_fig_dir / folder
        model_csv_dir = e8_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []  # pour stocker les résultats bruts

        # Pour chaque seed, on va d'abord obtenir la solution stable initiale (0% modification)
        # On stocke les références pour chaque seed
        ref_x = {}
        ref_W = {}

        for seed in seeds:
            # Générer la topologie propre (W original)
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

            # Créer M pour ce K (si modèle adjacent)
            if model_name == "adjacent":
                M = create_channel_interference_matrix(K)
            else:
                M = None

            # Exécuter BD-CeNN sur W_orig pour obtenir solution stable
            x_init, _, _, _, _ = method_func(
                N, K, W_orig, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=seed,
                verbose=False
            )
            ref_x[seed] = x_init

        # Maintenant, pour chaque niveau de modification, on génère W_dyn et on évalue
        for b_mod, mod_label in zip(mod_levels, mod_labels):
            for seed in seeds:
                W_orig = ref_W[seed]
                # Générer W_dyn en modifiant aléatoirement des arêtes/poids
                # On utilise une graine déterministe pour chaque (seed, niveau de modification)
                np.random.seed(seed + int(b_mod * 10000) + 200)
                W_dyn = W_orig.copy()
                # Lister les arêtes existantes (i<j, W>0)
                edges = [(i, j) for i in range(N) for j in range(i+1, N) if W_orig[i, j] > 0]
                if len(edges) == 0:
                    # Si pas d'arêtes, on passe
                    continue
                num_mod = int(len(edges) * b_mod)
                # Choix aléatoire des arêtes à modifier
                indices = np.random.choice(len(edges), num_mod, replace=False)
                for idx in indices:
                    i, j = edges[idx]
                    # Soit on supprime l'arête, soit on modifie son poids
                    if np.random.random() > 0.5:
                        # Suppression
                        W_dyn[i, j] = 0
                        W_dyn[j, i] = 0
                    else:
                        # Modification du poids : +1 ou -1 (si possible)
                        current = W_dyn[i, j]
                        if current > 1:
                            delta = np.random.choice([-1, 1])
                        elif current == 1:
                            delta = 1  # seulement +1 pour ne pas tomber à 0
                        else:
                            delta = 0
                        new_w = current + delta
                        new_w = max(1, new_w)  # on ne descend pas en dessous de 1
                        new_w = min(4, new_w)  # on ne dépasse pas 4
                        W_dyn[i, j] = new_w
                        W_dyn[j, i] = new_w
                # S'assurer de la symétrie et diagonale nulle
                for i in range(N):
                    for j in range(i+1, N):
                        W_dyn[j, i] = W_dyn[i, j]
                np.fill_diagonal(W_dyn, 0)

                # Créer M pour ce K (si modèle adjacent)
                if model_name == "adjacent":
                    M = create_channel_interference_matrix(K)
                else:
                    M = None

                # --- Mode 1 : Warm Start (départ de x_init) ---
                start = time.perf_counter()
                x_warm, _, _, _, _ = method_func(
                    N, K, W_dyn, M=M,
                    num_restarts=1,          # un seul départ pour warm start
                    max_iter=config.MAX_ITER_BD,
                    random_order=True,
                    seed=seed + int(b_mod * 1000) + 300,  # seed différente pour reproductibilité
                    verbose=False
                )
                time_warm = time.perf_counter() - start

                # --- Mode 2 : Cold Start (départ aléatoire) ---
                start = time.perf_counter()
                x_cold, _, _, _, _ = method_func(
                    N, K, W_dyn, M=M,
                    num_restarts=config.NUM_RESTARTS,  # redémarrages multiples comme d'habitude
                    max_iter=config.MAX_ITER_BD,
                    random_order=True,
                    seed=seed + int(b_mod * 1000) + 400,
                    verbose=False
                )
                time_cold = time.perf_counter() - start

                # Calcul des métriques
                # Nombre de réaffectations (changements par rapport à x_init)
                changes_warm = np.sum(x_warm != ref_x[seed])
                changes_cold = np.sum(x_cold != ref_x[seed])

                # Coûts finaux
                if M is None:
                    cost_warm = compute_cochannel_cost(x_warm, W_dyn)
                    cost_cold = compute_cochannel_cost(x_cold, W_dyn)
                else:
                    cost_warm = compute_adjacent_cost(x_warm, W_dyn, M)
                    cost_cold = compute_adjacent_cost(x_cold, W_dyn, M)

                raw_data.append({
                    "mod_level": b_mod * 100,  # en pourcentage
                    "seed": seed,
                    "mode": "warm",
                    "time": time_warm,
                    "changes": changes_warm,
                    "cost": cost_warm
                })
                raw_data.append({
                    "mod_level": b_mod * 100,
                    "seed": seed,
                    "mode": "cold",
                    "time": time_cold,
                    "changes": changes_cold,
                    "cost": cost_cold
                })

        # Convertir en DataFrame
        df_raw = pd.DataFrame(raw_data)

        # Sauvegarder les données brutes
        raw_csv_path = model_csv_dir / "raw_metrics.csv"
        df_raw.to_csv(raw_csv_path, index=False, float_format='%.6f')
        print(f"✅ CSV E8/{folder}/raw_metrics.csv sauvegardé.")

        # Calcul du résumé par niveau de modification et mode
        summary = df_raw.groupby(["mod_level", "mode"]).agg({
            "time": ["mean", "std", "min", "max", "median"],
            "changes": ["mean", "std", "min", "max", "median"],
            "cost": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        # Aplatir les colonnes
        summary.columns = ['mod_level', 'mode',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median',
                           'changes_mean', 'changes_std', 'changes_min', 'changes_max', 'changes_median',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median']
        # Arrondir les valeurs
        for col in summary.columns:
            if col not in ['mod_level', 'mode']:
                if 'changes' in col or 'cost' in col:
                    summary[col] = summary[col].round(1)
                else:  # time
                    summary[col] = summary[col].round(6)

        # Sauvegarder le résumé
        summary_csv_path = model_csv_dir / "summary_metrics.csv"
        summary.to_csv(summary_csv_path, index=False, float_format='%.6f')
        print(f"✅ CSV E8/{folder}/summary_metrics.csv sauvegardé.")

        # --- Figure : nombre moyen de canaux réaffectés vs niveau de modification ---
        fig, ax = plt.subplots(figsize=(10, 6))
        # Filtrer les données pour les deux modes
        warm_data = summary[summary['mode'] == 'warm']
        cold_data = summary[summary['mode'] == 'cold']

        # Ajouter le point (0,0) pour référence (aucune modification)
        mod_vals_warm = [0] + sorted(warm_data['mod_level'].unique())
        changes_warm_means = [0] + warm_data['changes_mean'].tolist()
        changes_warm_stds = [0] + warm_data['changes_std'].tolist()

        mod_vals_cold = [0] + sorted(cold_data['mod_level'].unique())
        changes_cold_means = [0] + cold_data['changes_mean'].tolist()
        changes_cold_stds = [0] + cold_data['changes_std'].tolist()

        ax.errorbar(mod_vals_warm, changes_warm_means, yerr=changes_warm_stds, fmt='o-', capsize=6,
                    color='green', label='Warm Start', linewidth=2, markersize=8,
                    elinewidth=2, markeredgewidth=2)
        ax.errorbar(mod_vals_cold, changes_cold_means, yerr=changes_cold_stds, fmt='s-', capsize=6,
                    color='purple', label='Cold Start', linewidth=2, markersize=8,
                    elinewidth=2, markeredgewidth=2)

        ax.set_xlabel("Taux de modification du réseau (%)", fontsize=12)
        ax.set_ylabel("Nombre moyen de canaux réaffectés", fontsize=12)
        ax.set_title(f"E8 - Adaptation dynamique - {model_name}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        # Ajuster l'axe Y pour bien voir les barres d'erreur
        y_min = min(changes_warm_means + changes_cold_means) - 2
        y_max = max(changes_warm_means + changes_cold_means) + 2
        ax.set_ylim([max(y_min, 0), y_max + 5])
        plt.tight_layout()
        fname = f"adaptation_{model_name}.png"
        plt.savefig(model_fig_dir / fname, dpi=300)
        plt.close(fig)
        print(f"✅ Figure E8/{folder}/{fname} sauvegardée.")

    print("✅ E8 terminée.")


# ============================================================================
# E9 – MINIMA LOCAUX (effet des redémarrages vs ordres de parcours)
# ============================================================================
def run_experiment_E9(verbose=False):
    """
    E9 - Minima locaux : effet du nombre de redémarrages pour BD-CeNN et 
    du nombre d'ordres de parcours pour Greedy.
    Utilise le scénario S3 (N=50, K=6) avec 30 topologies de référence.
    Pour chaque topologie, on exécute BD-CeNN avec 1, 5, 10, 20 redémarrages.
    Pour Greedy, on exécute avec 1, 5, 10, 20 ordres de parcours différents
    et on conserve le meilleur coût.
    Métriques : coût, conflits, temps d'exécution.
    Les résultats sont séparés pour co-canal (sans M) et adjacent (avec M).
    Produit :
        - CSV bruts et résumés par modèle
        - Figure : coût moyen vs nombre d'essais (redémarrages/ordres) avec barres d'erreur,
          pour BD-CeNN et Greedy.
    """
    import networkx as nx
    import time

    # Paramètres du scénario S3
    scenario_name = "S3"
    instances = all_instances.get(scenario_name, {})
    if not instances:
        print(f"❌ Scénario {scenario_name} introuvable dans scenarios_data.json")
        return

    first_inst = instances.get("1") or list(instances.values())[0]
    N = first_inst["N"]          # 50
    K = first_inst["K"]          # 6
    area = 200                   # issu de config
    threshold = 60               # S3 threshold
    seeds = [int(s) for s in instances.keys() if s.isdigit()]
    seeds = sorted(seeds)  # 1..30

    # Valeurs à tester
    values = [1, 5, 10, 20]
    value_labels = [str(v) for v in values]

    # Dossiers de sortie
    e9_fig_dir = config.FIGURES_DIR / "E9"
    e9_csv_dir = config.CSV_DIR / "E9"
    os.makedirs(e9_fig_dir, exist_ok=True)
    os.makedirs(e9_csv_dir, exist_ok=True)

    # Modèles d'interférence
    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"}  # M défini dans la boucle
    ]

    # Méthodes
    method_bd = bdcenn_allocation
    method_greedy = greedy_allocation

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_fig_dir = e9_fig_dir / folder
        model_csv_dir = e9_csv_dir / folder
        os.makedirs(model_fig_dir, exist_ok=True)
        os.makedirs(model_csv_dir, exist_ok=True)

        raw_data = []  # pour stocker les résultats bruts

        # Générer les topologies pour chaque seed
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

        # Boucle sur les topologies
        for seed, W in topologies.items():
            # Créer M pour ce K (si modèle adjacent)
            if model_name == "adjacent":
                M = create_channel_interference_matrix(K)
            else:
                M = None

            # --- BD-CeNN avec différents nombres de redémarrages ---
            for num_restarts in values:
                start = time.perf_counter()
                x_bd, _, _, _, _ = method_bd(
                    N, K, W, M=M,
                    num_restarts=num_restarts,
                    max_iter=config.MAX_ITER_BD,
                    random_order=True,
                    seed=seed + num_restarts * 1000,  # seed différente par configuration
                    verbose=False
                )
                elapsed = time.perf_counter() - start

                if M is None:
                    cost = compute_cochannel_cost(x_bd, W)
                    conflicts = count_cochannel_conflicts(x_bd, W)
                else:
                    cost = compute_adjacent_cost(x_bd, W, M)
                    conflicts = count_adjacent_conflicts(x_bd, W, M)

                raw_data.append({
                    "seed": seed,
                    "method": "BD-CeNN",
                    "value": num_restarts,
                    "cost": cost,
                    "conflicts": conflicts,
                    "time": elapsed
                })

            # --- Greedy avec différents nombres d'ordres de parcours ---
            for num_orders in values:
                best_cost = float('inf')
                best_conflicts = None
                best_time = 0.0
                best_x = None

                start_total = time.perf_counter()
                # Générer plusieurs ordres et garder le meilleur
                for order_idx in range(num_orders):
                    np.random.seed(seed + order_idx * 100 + num_orders * 1000)
                    order = np.random.permutation(N).tolist()
                    start = time.perf_counter()
                    x_g = method_greedy(N, K, W, order=order, M=M)
                    elapsed = time.perf_counter() - start
                    best_time += elapsed

                    if M is None:
                        cost_g = compute_cochannel_cost(x_g, W)
                        conflicts_g = count_cochannel_conflicts(x_g, W)
                    else:
                        cost_g = compute_adjacent_cost(x_g, W, M)
                        conflicts_g = count_adjacent_conflicts(x_g, W, M)

                    if cost_g < best_cost:
                        best_cost = cost_g
                        best_conflicts = conflicts_g
                        best_x = x_g

                total_elapsed = time.perf_counter() - start_total

                raw_data.append({
                    "seed": seed,
                    "method": "Greedy",
                    "value": num_orders,
                    "cost": best_cost,
                    "conflicts": best_conflicts,
                    "time": total_elapsed
                })

        # Convertir en DataFrame
        df_raw = pd.DataFrame(raw_data)

        # Sauvegarder les données brutes
        raw_csv_path = model_csv_dir / "raw_metrics.csv"
        df_raw.to_csv(raw_csv_path, index=False, float_format='%.6f')
        print(f"✅ CSV E9/{folder}/raw_metrics.csv sauvegardé.")

        # Calcul du résumé par (value, method)
        summary = df_raw.groupby(["value", "method"]).agg({
            "cost": ["mean", "std", "min", "max", "median"],
            "conflicts": ["mean", "std", "min", "max", "median"],
            "time": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        # Aplatir les colonnes
        summary.columns = ['value', 'method',
                           'cost_mean', 'cost_std', 'cost_min', 'cost_max', 'cost_median',
                           'conflicts_mean', 'conflicts_std', 'conflicts_min', 'conflicts_max', 'conflicts_median',
                           'time_mean', 'time_std', 'time_min', 'time_max', 'time_median']

        # Arrondir les valeurs
        for col in summary.columns:
            if col not in ['value', 'method']:
                if 'cost' in col or 'conflicts' in col:
                    summary[col] = summary[col].round(1)
                else:  # time
                    summary[col] = summary[col].round(6)

        # Sauvegarder le résumé
        summary_csv_path = model_csv_dir / "summary_metrics.csv"
        summary.to_csv(summary_csv_path, index=False, float_format='%.6f')
        print(f"✅ CSV E9/{folder}/summary_metrics.csv sauvegardé.")

        # --- Figure : coût moyen vs nombre d'essais (redémarrages/ordres) ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filtrer les données par méthode
        bd_data = summary[summary['method'] == 'BD-CeNN'].sort_values('value')
        greedy_data = summary[summary['method'] == 'Greedy'].sort_values('value')

        x_vals = values

        # BD-CeNN
        ax.errorbar(x_vals, bd_data['cost_mean'], yerr=bd_data['cost_std'], 
                    fmt='o-', capsize=6, color='red', label='BD-CeNN',
                    linewidth=2, markersize=8, elinewidth=2, markeredgewidth=2)

        # Greedy
        ax.errorbar(x_vals, greedy_data['cost_mean'], yerr=greedy_data['cost_std'],
                    fmt='s-', capsize=6, color='blue', label='Greedy',
                    linewidth=2, markersize=8, elinewidth=2, markeredgewidth=2)

        ax.set_xlabel("Nombre d'essais (redémarrages pour BD-CeNN, ordres de parcours pour Greedy)", fontsize=12)
        ax.set_ylabel("Coût global moyen J(x)", fontsize=12)
        ax.set_title(f"E9 - Minima locaux - {model_name}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        # Ajuster les ticks pour qu'ils correspondent aux valeurs
        ax.set_xticks(values)
        ax.set_xticklabels([str(v) for v in values])

        plt.tight_layout()
        fname = f"minima_locaux_{model_name}.png"
        plt.savefig(model_fig_dir / fname, dpi=300)
        plt.close(fig)
        print(f"✅ Figure E9/{folder}/{fname} sauvegardée.")

    print("✅ E9 terminée.")





# ============================================================================
# E10 – AUDIT LLM SUR 20 CAS REPRÉSENTATIFS
# ============================================================================
# ============================================================================
# E10 – AUDIT LLM SUR 20 CAS REPRÉSENTATIFS
# ============================================================================
def run_experiment_E10(verbose=False):
    """
    E10 - Évaluation de la fidélité du LLM sur 20 cas représentatifs.

    Pour chaque cas :
        - Construction des données de simulation à partir du fichier JSON
          (scenarios_data.json) et des résultats déjà calculés (validation_results.xlsx,
          E7/E8/E9 CSV) ou recalculés si nécessaire.
        - Soumission au module `llm_assistant` (prompt → LLM → contrôleur → PDF).

    DEUX MODES D'INTERFÉRENCE (dossier séparé pour chacun) :
        - "CCI-only" (dossier cochannel) : coût et conflits = CCI uniquement.
        - "CCI+ACI"  (dossier adjacent) : coût et conflits = CCI ET ACI combinés
                                         (UN SEUL chiffre, pas deux séparés).

    NOTE : l'évaluation académique automatique (grille 5 critères × 2 pts)
    et les figures radar associées ont été RETIRÉES. Les cotes sont attribuées
    MANUELLEMENT par l'auteur en se basant sur les PDF et le CSV récapitulatif.

    Produit :
        - Rapports PDF individuels dans
          results/logs/llm_logs/reports_pdf/{cochannel,adjacent}/
        - Fichier CSV récapitulatif :
          results/csv/E10/{cochannel,adjacent}/llm_fidelity_evaluation.csv

    NOTE (Solution 2 intégrée) :
        Les cas où l'appel LLM échoue définitivement (status == "LLM_FAILED")
        sont exclus du calcul de la moyenne. Ils restent listés à part dans le
        bilan global pour information.
    """
    import networkx as nx
    import time
    from llm_assistant import audit_case, LLM_REPORTS_DIR
    from bdcenn_solver import bdcenn_allocation
    from baselines import greedy_allocation, dsatur_allocation, random_allocation
    from metrics import (
        create_channel_interference_matrix,
        compute_cochannel_cost, count_cochannel_conflicts,
        compute_adjacent_cost, count_adjacent_conflicts,
    )

    # Dossiers de sortie (uniquement le CSV — plus de dossier de figures)
    e10_csv_dir = config.CSV_DIR / "E10"
    os.makedirs(e10_csv_dir, exist_ok=True)

    # --- Définition des 20 cas (voir image jointe) ---
    cases_def = [
        {"case_id": "#1",  "scenario": "S1", "N": 8,   "K": 3, "seed": 1, "context": "Validation visuelle simple"},
        {"case_id": "#2",  "scenario": "S1", "N": 8,   "K": 3, "seed": 2, "context": "Deuxième topologie simple"},
        {"case_id": "#3",  "scenario": "S2", "N": 30,  "K": 4, "seed": 1, "context": "Cas nominal standard"},
        {"case_id": "#4",  "scenario": "S2", "N": 30,  "K": 4, "seed": 2, "context": "Deuxième topologie moyenne"},
        {"case_id": "#5",  "scenario": "S2", "N": 30,  "K": 6, "seed": 1, "context": "Plus de canaux"},
        {"case_id": "#6",  "scenario": "S3", "N": 50,  "K": 6, "seed": 1, "context": "Forte densité (beaucoup de contraintes)"},
        {"case_id": "#7",  "scenario": "S3", "N": 50,  "K": 6, "seed": 2, "context": "Deuxième topologie dense"},
        {"case_id": "#8",  "scenario": "S3", "N": 50,  "K": 6, "seed": 3, "context": "Troisième topologie dense"},
        {"case_id": "#9",  "scenario": "S4", "N": 50,  "K": 2, "seed": 1, "context": "Cas très difficile (beaucoup de conflits restants)"},
        {"case_id": "#10", "scenario": "S4", "N": 50,  "K": 2, "seed": 2, "context": "Deuxième cas de pénurie sévère"},
        {"case_id": "#11", "scenario": "S4", "N": 50,  "K": 3, "seed": 1, "context": "Pénurie modérée"},
        {"case_id": "#12", "scenario": "S5", "N": 100, "K": 8, "seed": 1, "context": "Grand réseau (100 cellules)"},
        {"case_id": "#13", "scenario": "S5", "N": 200, "K": 8, "seed": 1, "context": "Très grand réseau (200 cellules)"},
        {"case_id": "#14", "scenario": "S6", "N": 50,  "K": 4, "seed": 1, "context": "Faible bruit de mesure sur la matrice W (bruit 5%)"},
        {"case_id": "#15", "scenario": "S6", "N": 50,  "K": 4, "seed": 1, "context": "Bruit moyen sur W (bruit 10%)"},
        {"case_id": "#16", "scenario": "S6", "N": 50,  "K": 4, "seed": 1, "context": "Fort bruit (dégradation de coût élevée, bruit 20%)"},
        {"case_id": "#17", "scenario": "S7", "N": 45,  "K": 5, "seed": 1, "context": "Perturbation dynamique faible (réoptimisation 'à chaud', modif 5%)"},
        {"case_id": "#18", "scenario": "S7", "N": 45,  "K": 5, "seed": 1, "context": "Perturbation dynamique moyenne (modif 10%)"},
        {"case_id": "#19", "scenario": "S7", "N": 45,  "K": 5, "seed": 1, "context": "Forte perturbation dynamique (modif 20%)"},
        {"case_id": "#20", "scenario": "S4", "N": 50,  "K": 2, "seed": 1, "context": "Cas où le BD-CeNN s'est bloqué dans un minimum local (E9, 1 redémarrage)"},
    ]

    # --- Boucle sur les deux modèles (co-canal / adjacent) ---
    models = [
        {"name": "cochannel", "M": None, "folder": "cochannel"},
        {"name": "adjacent", "M": None, "folder": "adjacent"},
    ]

    # Dictionnaire global pour stocker les résultats de tous les cas (par modèle)
    all_results = {}

    for model in models:
        model_name = model["name"]
        folder = model["folder"]
        model_csv_dir = e10_csv_dir / folder
        os.makedirs(model_csv_dir, exist_ok=True)

        results_for_model = []

        # --- Boucle sur les 20 cas ---
        for case in cases_def:
            print(f"\n{'='*70}")
            print(f"🔎 Cas {case['case_id']} - {case['scenario']} ({model_name})")
            print(f"{'='*70}")

            sc_name = case["scenario"]
            N = case["N"]
            K = case["K"]
            seed = case["seed"]

            # Charger la topologie correspondante
            instances = all_instances.get(sc_name, {})
            if not instances:
                print(f"❌ Scénario {sc_name} introuvable.")
                continue

            # Pour S5, les tailles N=200 ne sont pas dans le JSON (N=100 max)
            # -> on génère une topologie synthétique avec la seed
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

            # Adapter W au cas N/K spécifique (S2 avec K=6, S4 avec K=3)
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

            # Gérer les cas spéciaux S6 (bruit) et S7 (modification dynamique)
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
                            new_w = current + np.random.choice([-1, 1]) * min(current, 1)
                            new_w = max(1, min(4, new_w))
                            W[i, j] = new_w
                            W[j, i] = new_w

            # Créer la matrice M pour ce K (si modèle adjacent)
            if model_name == "adjacent":
                M = create_channel_interference_matrix(K)
            else:
                M = None

            # --- Exécuter BD-CeNN ---
            # Cas #20 : 1 seul redémarrage (minimum local)
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

            # Coût initial (première allocation de l'historique)
            if history_bd:
                alloc_init = history_bd[0][2]
                if M is None:
                    cost_init = compute_cochannel_cost(alloc_init, W)
                else:
                    cost_init = compute_adjacent_cost(alloc_init, W, M)
            else:
                cost_init = 0.0

            # ---------------------------------------------------------------
            # Métriques finales BD-CeNN — UNE SEULE valeur de "conflicts"
            # ---------------------------------------------------------------
            if M is None:
                cost_final = compute_cochannel_cost(x_bd, W)
                conflicts_final = count_cochannel_conflicts(x_bd, W)
            else:
                cost_final = compute_adjacent_cost(x_bd, W, M)
                conflicts_final = count_adjacent_conflicts(x_bd, W, M)

            used_channels = len(set(x_bd))

            # --- Exécuter les baselines ---
            np.random.seed(seed)
            x_rand = random_allocation(N, K)
            np.random.seed(seed)
            order_greedy = np.random.permutation(N).tolist()
            x_greedy = greedy_allocation(N, K, W, order=order_greedy, M=M)
            x_dsatur = dsatur_allocation(N, K, W, M=M)

            baselines = {}
            for name, x_b, t_b in [("Random", x_rand, 0.0), ("Greedy", x_greedy, 0.0), ("DSATUR", x_dsatur, 0.0)]:
                if M is None:
                    c = compute_cochannel_cost(x_b, W)
                    cf = count_cochannel_conflicts(x_b, W)
                else:
                    c = compute_adjacent_cost(x_b, W, M)
                    cf = count_adjacent_conflicts(x_b, W, M)
                baselines[name] = {"cost": float(c), "conflicts": int(cf), "time": float(t_b)}

            # --- Cellules en conflit ---
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

            # --- Étiquette du mode d'interférence ---
            model_label = "CCI-only" if model_name == "cochannel" else "CCI+ACI"

            # --- Construire le case_data complet ---
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
            }

            # --- Appeler le module LLM ---
            try:
                audit_result = audit_case(
                    case_data,
                    max_correction_attempts=2,
                    model_folder=model_name,
                )
                results_for_model.append(audit_result)
            except Exception as e:
                print(f"❌ Erreur lors de l'audit LLM du cas {case['case_id']} : {e}")
                continue

        all_results[model_name] = results_for_model

        # ----------------------------------------------------------------
        # Filtrage des cas réussis vs échoués
        # ----------------------------------------------------------------
        valid_results = [r for r in results_for_model if r.get("status") != "LLM_FAILED"]
        failed_results = [r for r in results_for_model if r.get("status") == "LLM_FAILED"]

        # --- Sauvegarde du CSV récapitulatif ---
        # NOTE : les colonnes de score (Score /10, Exactitude, Cohérence,
        # Clarté, Utilité, Traçabilité) ont été RETIRÉES. Les cotes sont
        # attribuées manuellement par l'auteur.
        rows = []
        for r in results_for_model:
            status = r.get("status", "OK")
            if status == "LLM_FAILED":
                rows.append({
                    "Cas": r["case_id"],
                    "Scénario (N, K, seed)": f"{r['scenario']} (N={r['N']}, K={r['K']}, seed={r['seed']})",
                    "Contexte": r["context"],
                    "Statut": "LLM_FAILED",
                    "Invention détectée (Oui/Non)": "N/A",
                    "Type d'erreur / Correction appliquée": "Échec LLM (503 répété) — cas non évalué",
                    "Taux exactitude initial": 0.0,
                    "Taux exactitude final": 0.0,
                })
            else:
                inv_detected = "Oui" if r["verification_initial"]["has_hallucination"] else "Non"
                if r["verification_initial"]["has_hallucination"]:
                    invented_vals = ", ".join(raw for _, raw in r["verification_initial"]["invented_numbers"])
                    type_err = f"Hallucination : {invented_vals}"
                    if r["verification_final"]["has_hallucination"]:
                        type_err += " (non corrigée)"
                    else:
                        type_err += f" (corrigée en {r['correction_attempts']} tentative(s))"
                else:
                    type_err = "Aucune erreur"
                rows.append({
                    "Cas": r["case_id"],
                    "Scénario (N, K, seed)": f"{r['scenario']} (N={r['N']}, K={r['K']}, seed={r['seed']})",
                    "Contexte": r["context"],
                    "Statut": "OK",
                    "Invention détectée (Oui/Non)": inv_detected,
                    "Type d'erreur / Correction appliquée": type_err,
                    "Taux exactitude initial": r["verification_initial"]["accuracy_rate"],
                    "Taux exactitude final": r["verification_final"]["accuracy_rate"],
                })
        df_eval = pd.DataFrame(rows)
        csv_path = model_csv_dir / "llm_fidelity_evaluation.csv"
        df_eval.to_csv(csv_path, index=False, float_format="%.2f")
        print(f" CSV E10/{folder}/llm_fidelity_evaluation.csv sauvegardé.")

        # NOTE : la génération des figures radar individuelles et de la
        # figure radar groupée a été RETIRÉE. Le dossier figures/E10/ n'est
        # plus créé.

    # --- Bilan global (filtrage des cas LLM_FAILED) ---
    print("\n" + "=" * 80)
    print("📊 BILAN E10 - Évaluation de la fidélité du LLM")
    print("=" * 80)
    for model_name, results in all_results.items():
        if not results:
            continue

        valid_results = [r for r in results if r.get("status") != "LLM_FAILED"]
        failed_results = [r for r in results if r.get("status") == "LLM_FAILED"]

        print(f"\n[{model_name}]")
        print(f"  Nombre de cas évalués         : {len(results)}")
        print(f"  Cas réussis                   : {len(valid_results)}")
        print(f"  Cas échoués (LLM indisponible): {len(failed_results)}")

        if valid_results:
            hallu_count = sum(1 for r in valid_results if r["verification_initial"]["has_hallucination"])
            corrected_count = sum(
                1 for r in valid_results
                if r["verification_initial"]["has_hallucination"] and not r["verification_final"]["has_hallucination"]
            )
            print(f"  Cas avec hallucination (init) : {hallu_count}/{len(valid_results)}")
            print(f"  Cas corrigés automatiquement  : {corrected_count}/{hallu_count if hallu_count > 0 else 1}")

            if failed_results:
                failed_ids = ", ".join(r["case_id"] for r in failed_results)
                print(f"  Cas LLM_FAILED                : {failed_ids}")
        else:
            print("  ⚠️ Aucun cas réussi pour ce modèle.")

    print("\n✅ E10 terminée.")






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
    run_experiment_E2(verbose)
    run_experiment_E3(verbose)
    run_experiment_E4(verbose)
    run_experiment_E6(verbose)
    run_experiment_E7(verbose)
    run_experiment_E8(verbose)
    run_experiment_E9(verbose)
    run_experiment_E10(verbose)

    print("\n" + "=" * 80)
    print("✅ TOUTES LES EXPÉRIENCES SONT TERMINÉES.")
    print("=" * 80)


if __name__ == "__main__":
    # Par défaut, exécute E9 pour une visualisation rapide.
    # Pour lancer toutes les expériences, utiliser main.py ou décommenter la ligne ci-dessous.
    run_experiment_E10()
    # run_all_experiments()