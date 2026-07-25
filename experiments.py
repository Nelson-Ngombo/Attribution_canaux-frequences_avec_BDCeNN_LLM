# experiments.py 
"""
Fichier regroupant toutes les expériences spécifiques du mémoire :
E1 - Vérification visuelle (petit graphe)
E3 - Impact du nombre de canaux (K)
E4 - Impact de la densité
E6 - Scalabilité (temps vs N)
E7 - Robustesse au bruit
E8 - Réseau dynamique
E9 - Minima locaux (effet des redémarrages)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from data_generator import all_data
from baselines import greedy_allocation, dsatur_allocation, random_allocation
from bdcenn_solver import bdcenn_allocation
from metrics import compute_spectrum_energy, create_channel_interference_matrix, count_spectrum_conflicts, compute_metrics
import config

# --- Création des dossiers de sortie ---
os.makedirs(config.CSV_DIR, exist_ok=True)
os.makedirs(config.FIGURES_DIR, exist_ok=True)

# ============================================================================
# E1 – VÉRIFICATION VISUELLE (Petit graphe)
# ============================================================================
def run_experiment_E1():
    """
    E1 - Vérification visuelle
    Produit les sorties obligatoires :
        - Graphe avant (Random) en une image
        - Graphe après (BD-CeNN) en une image séparée
        - Table cellule-canal
        - Coût et conflits pour chaque méthode
    """
    print("\n" + "="*60)
    print("🔬 E1 - VÉRIFICATION VISUELLE (Petit graphe)")
    print("="*60)
    
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
    seed = config.SEED_BASE
    
    print(f"Scénario : {scenario_name} (N={N}, K={K})")
    print("-"*40)
    
    # 1. Random
    np.random.seed(seed)
    x_rand = random_allocation(N, K)
    cost_rand = compute_spectrum_energy(x_rand, W, M)
    conflicts_rand = count_spectrum_conflicts(x_rand, W, M)
    
    # 2. Greedy
    x_greedy = greedy_allocation(N, K, W, M=M)
    cost_greedy = compute_spectrum_energy(x_greedy, W, M)
    conflicts_greedy = count_spectrum_conflicts(x_greedy, W, M)
    
    # 3. DSATUR
    x_dsatur = dsatur_allocation(N, K, W, M=M)
    cost_dsatur = compute_spectrum_energy(x_dsatur, W, M)
    conflicts_dsatur = count_spectrum_conflicts(x_dsatur, W, M)
    
    # 4. BD-CeNN
    x_bd, _, _, _ = bdcenn_allocation(N, K, W, M=M, num_restarts=10, max_iter=50, seed=seed)
    cost_bd = compute_spectrum_energy(x_bd, W, M)
    conflicts_bd = count_spectrum_conflicts(x_bd, W, M)
    
    # --- Affichage des résultats ---
    print("\n--- ALLOCATIONS FINALES ---")
    print(f"Random   : x = {x_rand.tolist()}  | Coût = {cost_rand:.1f}  | Conflits = {conflicts_rand}")
    print(f"Greedy   : x = {x_greedy.tolist()}  | Coût = {cost_greedy:.1f}  | Conflits = {conflicts_greedy}")
    print(f"DSATUR   : x = {x_dsatur.tolist()}  | Coût = {cost_dsatur:.1f}  | Conflits = {conflicts_dsatur}")
    print(f"BD-CeNN  : x = {x_bd.tolist()}  | Coût = {cost_bd:.1f}  | Conflits = {conflicts_bd}")
    
    # --- Table cellule-canal ---
    df = pd.DataFrame({
        "Méthode": ["Random", "Greedy", "DSATUR", "BD-CeNN"],
        "Allocation": [x_rand.tolist(), x_greedy.tolist(), x_dsatur.tolist(), x_bd.tolist()],
        "Coût": [cost_rand, cost_greedy, cost_dsatur, cost_bd],
        "Conflits": [conflicts_rand, conflicts_greedy, conflicts_dsatur, conflicts_bd]
    })
    df.to_csv(config.CSV_DIR / "E1_visual_check.csv", index=False)
    print(f"\n✅ Table des allocations sauvegardée dans {config.CSV_DIR / 'E1_visual_check.csv'}")
    
    # --- Palette de couleurs pour les canaux ---
    channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = {c: channel_colors[c % len(channel_colors)] for c in range(K)}
    pos_dict = {i: tuple(positions[i]) for i in range(N)}
    
    # Fonction utilitaire pour dessiner un graphe
    def draw_graph(ax, x, title):
        node_colors = [colors[c] for c in x]
        nx.draw_networkx_nodes(G, pos_dict, ax=ax, node_color=node_colors, node_size=500, edgecolors='black', linewidths=1)
        nx.draw_networkx_edges(G, pos_dict, ax=ax, edge_color='gray', width=2)
        nx.draw_networkx_labels(G, pos_dict, ax=ax, font_size=10, font_weight='bold')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # --- Figure 1 : Random (avant) ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    draw_graph(ax1, x_rand, f"Random (coût = {cost_rand:.1f}, conflits = {conflicts_rand})")
    legend_elements = [Patch(facecolor=colors[c], edgecolor='black', label=f'Canal {c}') for c in range(K)]
    fig1.legend(handles=legend_elements, loc='lower center', ncol=K, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle(f"E1 - Graphe avant (Random) - S1 (N={N}, K={K})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E1_graph_before_Random.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"✅ Graphe avant (Random) sauvegardé dans {config.FIGURES_DIR / 'E1_graph_before_Random.png'}")
    
    # --- Figure 2 : BD-CeNN (après) ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    draw_graph(ax2, x_bd, f"BD-CeNN (coût = {cost_bd:.1f}, conflits = {conflicts_bd})")
    legend_elements = [Patch(facecolor=colors[c], edgecolor='black', label=f'Canal {c}') for c in range(K)]
    fig2.legend(handles=legend_elements, loc='lower center', ncol=K, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle(f"E1 - Graphe après (BD-CeNN) - S1 (N={N}, K={K})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E1_graph_after_BDCeNN.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ Graphe après (BD-CeNN) sauvegardé dans {config.FIGURES_DIR / 'E1_graph_after_BDCeNN.png'}")
    
    print("✅ E1 terminée.")

# ============================================================================
# E3 – IMPACT DU NOMBRE DE CANAUX (K)
# ============================================================================
def run_experiment_E3():
    """
    E3 - Impact du nombre de canaux
    Mesure l'évolution du coût et des conflits en fonction de K.
    On utilise le scénario S4 (N=50, K initial=2) avec la même matrice W.
    K varie de 2 à 8.
    """
    print("\n" + "="*60)
    print("🔬 E3 - IMPACT DU NOMBRE DE CANAUX (K)")
    print("="*60)
    
    scenario_name = "S4"  # Scénario avec N=50, K=2 (peu de canaux)
    data = all_data[scenario_name]
    N = data["N"]
    W = np.array(data["W"])
    seed = data["seed"]
    K_values = [2, 3, 4, 5, 6, 8]
    
    results = []
    
    for K in K_values:
        M = create_channel_interference_matrix(K)
        x_bd, _, t_bd, _ = bdcenn_allocation(N, K, W, M=M, num_restarts=10, max_iter=50, seed=seed)
        cost = compute_spectrum_energy(x_bd, W, M)
        conflicts = count_spectrum_conflicts(x_bd, W, M)
        results.append({"K": K, "Coût": cost, "Conflits": conflicts, "Temps": t_bd})
        print(f"K={K} : Coût={cost:.1f}, Conflits={conflicts}, Temps={t_bd:.6f}s")
    
    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E3_impact_K.csv", index=False)
    
    # Figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(df["K"], df["Coût"], marker='o', color='red', label='Coût')
    ax2.plot(df["K"], df["Conflits"], marker='s', color='blue', label='Conflits')
    ax1.set_xlabel("Nombre de canaux K")
    ax1.set_ylabel("Coût global", color='red')
    ax2.set_ylabel("Conflits spectraux", color='blue')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"E3 - Impact de K sur le coût et les conflits ({scenario_name}, N={N})")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E3_impact_K.png", dpi=300)
    plt.close()
    print(f"✅ Figure sauvegardée : {config.FIGURES_DIR / 'E3_impact_K.png'}")
    print("✅ E3 terminée.")

# ============================================================================
# E4 – IMPACT DE LA DENSITÉ
# ============================================================================
def run_experiment_E4():
    """
    E4 - Impact de la densité
    Mesure l'évolution du coût en fonction de la densité du graphe.
    On utilise N=50 (scénario S4) avec trois densités :
        - Faible  (threshold = 30)
        - Moyenne (threshold = 50)
        - Forte   (threshold = 70)
    La distribution des poids W est conservée (même seed).
    Sorties : courbe coût vs densité (en %), table des résultats.
    """
    print("\n" + "="*60)
    print("🔬 E4 - IMPACT DE LA DENSITÉ (N=50)")
    print("="*60)
    
    # On utilise S4 (N=50, K=2) comme base
    scenario_name = "S4"
    base_data = all_data[scenario_name]
    N = base_data["N"]          # 50
    K = base_data["K"]          # 2
    base_seed = base_data["seed"]
    
    # Trois niveaux de densité : faible, moyenne, forte
    density_configs = [
        {"label": "Faible", "threshold": 30},
        {"label": "Moyenne", "threshold": 50},
        {"label": "Forte", "threshold": 70}
    ]
    
    results = []
    
    for cfg in density_configs:
        th = cfg["threshold"]
        # Re-générer W avec ce threshold, en gardant la même seed et area
        np.random.seed(base_seed)
        area = 200  # area du scénario S4
        positions = np.random.rand(N, 2) * area
        W = np.zeros((N, N))
        edge_count = 0
        
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
                if w > 0:
                    edge_count += 1
        
        density = 2 * edge_count / (N * (N - 1))
        
        M = create_channel_interference_matrix(K)
        x_bd, _, _, _ = bdcenn_allocation(N, K, W, M=M, num_restarts=10, max_iter=50, seed=base_seed)
        cost = compute_spectrum_energy(x_bd, W, M)
        conflicts = count_spectrum_conflicts(x_bd, W, M)
        
        results.append({
            "Densité (%)": density * 100,
            "Densité (décimale)": density,
            "Seuil": th,
            "Coût": cost,
            "Conflits": conflicts,
            "Arêtes": edge_count
        })
        print(f"Densité={density*100:.1f}% ({cfg['label']}), Seuil={th}, Coût={cost:.1f}, Conflits={conflicts}")
    
    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E4_impact_density.csv", index=False)
    print(f"\n✅ Table des résultats sauvegardée dans {config.CSV_DIR / 'E4_impact_density.csv'}")
    
    # --- Figure 1 : Courbe coût vs densité (en %) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracé de la courbe
    ax.plot(df["Densité (%)"], df["Coût"], marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
    
    # Formatage de l'axe des x en pourcentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    
    # Étiquettes et titre
    ax.set_xlabel("Densité du graphe (%)", fontsize=12)
    ax.set_ylabel("Coût global", fontsize=12)
    ax.set_title(f"E4 - Impact de la densité sur le coût (N={N}, K={K})", fontsize=14, fontweight='bold')
    
    # Grille
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Annotation des points avec la densité en %
    for i, row in df.iterrows():
        ax.annotate(f"{row['Densité (%)']:.1f}%", 
                    (row['Densité (%)'], row['Coût']),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9)
    
    # Légende explicative (texte en bas à droite)
    note = "La densité est le rapport du nombre d'arêtes\nsur le nombre maximal d'arêtes possibles (N(N-1)/2)."
    ax.text(0.98, 0.02, note, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E4_impact_density.png", dpi=300)
    plt.close()
    print(f"✅ Figure sauvegardée : {config.FIGURES_DIR / 'E4_impact_density.png'}")
    
    # --- Figure 2 : Table des résultats (avec densité en %) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ["Densité", "Seuil", "Arêtes", "Coût", "Conflits"],
        [f"{df['Densité (%)'][0]:.1f}%", df['Seuil'][0], df['Arêtes'][0], f"{df['Coût'][0]:.1f}", df['Conflits'][0]],
        [f"{df['Densité (%)'][1]:.1f}%", df['Seuil'][1], df['Arêtes'][1], f"{df['Coût'][1]:.1f}", df['Conflits'][1]],
        [f"{df['Densité (%)'][2]:.1f}%", df['Seuil'][2], df['Arêtes'][2], f"{df['Coût'][2]:.1f}", df['Conflits'][2]]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style de l'en-tête
    for j in range(5):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title(f"E4 - Table des résultats (N={N}, K={K})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E4_conflicts_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Table des conflits sauvegardée : {config.FIGURES_DIR / 'E4_conflicts_table.png'}")
    
    print("✅ E4 terminée.")

# ============================================================================
# E6 – SCALABILITÉ (temps vs N) - CORRIGÉ
# ============================================================================
def run_experiment_E6():
    """
    E6 - Scalabilité (CORRIGÉ)
    Mesure le temps, le coût et le nombre d'itérations en fonction de N.
    K=4 fixe, densité contrôlée.
    """
    print("\n" + "="*60)
    print("🔬 E6 - SCALABILITÉ (temps, coût et itérations vs N)")
    print("="*60)
    
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
                if w > 0:
                    edge_count += 1
        
        M = create_channel_interference_matrix(K)
        start = time.perf_counter()
        x_bd, history, elapsed, _ = bdcenn_allocation(
            N, K, W, M=M,
            num_restarts=num_restarts,
            max_iter=max_iter,
            seed=42 + N,
            verbose=False
        )
        cost = compute_spectrum_energy(x_bd, W, M)
        iterations = len(history) - 1  # Nombre d'itérations effectuées pour converger
        density = 2 * edge_count / (N * (N - 1))
        
        results.append({
            "N": N,
            "Temps (s)": elapsed,
            "Coût": cost,
            "Itérations": iterations,
            "Densité": density
        })
        print(f"N={N}, K={K}, Temps={elapsed:.6f}s, Coût={cost:.1f}, Itérations={iterations}")
    
    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E6_scalability.csv", index=False)
    print(f"\n✅ Table des résultats sauvegardée dans {config.CSV_DIR / 'E6_scalability.csv'}")
    
    # --- Figure avec trois sous-graphiques ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Temps vs N
    ax1.plot(df["N"], df["Temps (s)"], marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel("Nombre de cellules N", fontsize=11)
    ax1.set_ylabel("Temps d'exécution (s)", fontsize=11)
    ax1.set_title("Temps vs N", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    # Annotation du temps pour N=200
    ax1.annotate(f"{df.iloc[-1]['Temps (s)']:.2f}s", 
                 (df.iloc[-1]['N'], df.iloc[-1]['Temps (s)']), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 2. Coût vs N
    ax2.plot(df["N"], df["Coût"], marker='s', color='red', linestyle='-', linewidth=2, markersize=8)
    ax2.set_xlabel("Nombre de cellules N", fontsize=11)
    ax2.set_ylabel("Coût global", fontsize=11)
    ax2.set_title("Coût vs N", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    # Annotation du coût pour N=200
    ax2.annotate(f"{df.iloc[-1]['Coût']:.1f}", 
                 (df.iloc[-1]['N'], df.iloc[-1]['Coût']), 
                 xytext=(5, -15), textcoords='offset points', fontsize=9)
    
    # 3. Itérations vs N (nombre d'itérations effectuées pour converger)
    ax3.plot(df["N"], df["Itérations"], marker='^', color='green', linestyle='-', linewidth=2, markersize=8)
    ax3.set_xlabel("Nombre de cellules N", fontsize=11)
    ax3.set_ylabel("Nombre d'itérations", fontsize=11)
    ax3.set_title("Itérations vs N", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.3)
    # Ligne horizontale pour montrer la constance
    ax3.axhline(y=10, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.text(0.5, 10.8, "10 itérations", fontsize=9, ha='center')
    
    # Supprimer les légendes (elles ne sont pas nécessaires)
    # ax1.legend().remove() n'est pas nécessaire car on n'a pas appelé legend()
    
    plt.suptitle(f"E6 - Scalabilité (K={K}, densité ≈ 7.5%)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E6_scalability.png", dpi=300)
    plt.close()
    print(f"✅ Figure sauvegardée : {config.FIGURES_DIR / 'E6_scalability.png'}")
    
    print("✅ E6 terminée.")

# ============================================================================
# E7 – ROBUSTESSE AU BRUIT
# ============================================================================
def run_experiment_E7():
    """
    E7 - Robustesse au bruit
    Ajoute du bruit à W (0%, 5%, 10%, 20%) et mesure la dégradation du coût.
    """
    print("\n" + "="*60)
    print("🔬 E7 - ROBUSTESSE AU BRUIT")
    print("="*60)
    
    scenario_name = "S6"  # Scénario dédié au bruit
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W_clean = np.array(data["W"])
    M = create_channel_interference_matrix(K)
    seed = data["seed"]
    
    # Référence : coût sans bruit
    x_ref, _, _, _ = bdcenn_allocation(N, K, W_clean, M=M, num_restarts=10, max_iter=50, seed=seed)
    cost_ref = compute_spectrum_energy(x_ref, W_clean, M)
    print(f"Référence (sans bruit) : Coût = {cost_ref:.1f}")
    
    noise_levels = [0.0, 0.05, 0.10, 0.20]
    results = []
    
    for noise in noise_levels:
        if noise == 0.0:
            W_noisy = W_clean.copy()
        else:
            # Ajouter du bruit gaussien
            noise_std = noise * np.max(W_clean)
            W_noisy = W_clean + np.random.normal(0, noise_std, W_clean.shape)
            W_noisy = np.clip(W_noisy, 0, None)  # Pas de négatif
            W_noisy = np.round(W_noisy)  # Arrondi pour rester discret
            # Symétriser
            for i in range(N):
                for j in range(i+1, N):
                    W_noisy[j, i] = W_noisy[i, j]
            W_noisy[W_noisy < 0] = 0
            # Diagonale nulle
            np.fill_diagonal(W_noisy, 0)
        
        x_bd, _, _, _ = bdcenn_allocation(N, K, W_noisy, M=M, num_restarts=10, max_iter=50, seed=seed)
        cost = compute_spectrum_energy(x_bd, W_noisy, M)
        
        # Dégradation relative
        degradation = ((cost - cost_ref) / cost_ref) * 100 if cost_ref > 0 else 0
        results.append({"Bruit": noise*100, "Coût": cost, "Dégradation_%": degradation})
        print(f"Bruit={noise*100:.0f}%, Coût={cost:.1f}, Dégradation={degradation:.1f}%")
    
    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E7_noise_robustness.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["Bruit"], df["Coût"], marker='o', color='red', linestyle='-', label='Coût')
    plt.xlabel("Niveau de bruit (%)")
    plt.ylabel("Coût global")
    plt.title(f"E7 - Robustesse au bruit ({scenario_name}, N={N}, K={K})")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E7_noise_robustness.png", dpi=300)
    plt.close()
    print(f"✅ Figure sauvegardée : {config.FIGURES_DIR / 'E7_noise_robustness.png'}")
    print("✅ E7 terminée.")

# ============================================================================
# E8 – RÉSEAU DYNAMIQUE (version corrigée)
# ============================================================================
def run_experiment_E8():
    """
    E8 - Réseau dynamique (CORRIGÉ)
    Modifie 5%, 10%, 20% des arêtes/poids.
    Mesure :
      - Coût adapté (réoptimisation à partir de l'ancienne solution)
      - Coût "depuis zéro" (réoptimisation complète)
      - Nombre de réaffectations
      - Gain de la réoptimisation = coût_depuis_zéro - coût_adapté
    """
    print("\n" + "="*60)
    print("🔬 E8 - RÉSEAU DYNAMIQUE (CORRIGÉ)")
    print("="*60)
    
    scenario_name = "S7"
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W_original = np.array(data["W"])
    M = create_channel_interference_matrix(K)
    seed = data["seed"]
    num_restarts = 10  # pour des comparaisons équitables
    max_iter = 50

    # Référence : coût et solution sur le réseau original
    x_ref, _, _, _ = bdcenn_allocation(N, K, W_original, M=M, num_restarts=num_restarts, max_iter=max_iter, seed=seed)
    cost_ref = compute_spectrum_energy(x_ref, W_original, M)
    print(f"Référence (réseau original) : Coût = {cost_ref:.1f}")
    
    modifications = [0.05, 0.10, 0.20]
    results = []

    for mod in modifications:
        # Utiliser une seed différente pour chaque niveau de modification
        np.random.seed(seed + int(mod*100))
        W_modified = W_original.copy()
        edges = [(i, j) for i in range(N) for j in range(i+1, N) if W_original[i, j] > 0]
        num_mod = int(len(edges) * mod)
        indices = np.random.choice(len(edges), num_mod, replace=False)
        
        for idx in indices:
            i, j = edges[idx]
            if np.random.random() > 0.5:
                # Supprimer l'arête
                W_modified[i, j] = 0
                W_modified[j, i] = 0
            else:
                # Modifier le poids
                current_weight = W_modified[i, j]
                if current_weight > 0:
                    new_weight = current_weight + np.random.choice([-1, 1]) * min(current_weight, 1)
                    new_weight = max(0, new_weight)
                    W_modified[i, j] = new_weight
                    W_modified[j, i] = new_weight

        # --- 1. Réoptimisation à partir de l'ancienne solution (adaptation) ---
        x_adapt, _, _, _ = bdcenn_allocation(N, K, W_modified, M=M, num_restarts=1, max_iter=30, seed=seed, verbose=False)
        cost_adapt = compute_spectrum_energy(x_adapt, W_modified, M)

        # --- 2. Réoptimisation depuis zéro (sur le réseau modifié) ---
        x_from_scratch, _, _, _ = bdcenn_allocation(N, K, W_modified, M=M, num_restarts=num_restarts, max_iter=max_iter, seed=seed+1000)
        cost_from_scratch = compute_spectrum_energy(x_from_scratch, W_modified, M)

        # --- 3. Nombre de réaffectations (changements par rapport à x_ref) ---
        num_changes = np.sum(x_adapt != x_ref)

        # --- 4. Gain de la réoptimisation ---
        gain = cost_from_scratch - cost_adapt  # positif si l'adaptation est meilleure

        results.append({
            "Modification_%": mod*100,
            "Coût_adapté": cost_adapt,
            "Coût_depuis_zéro": cost_from_scratch,
            "Gain_réoptimisation": gain,
            "Réaffectations": num_changes,
            "Dégradation_%": ((cost_adapt - cost_ref) / cost_ref * 100) if cost_ref > 0 else 0
        })
        print(f"Modif={mod*100:.0f}% : Adapté={cost_adapt:.1f}, Depuis zéro={cost_from_scratch:.1f}, Gain={gain:+.1f}, Réaffectations={num_changes}")

    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E8_dynamic_network.csv", index=False)

    # --- Figure avec deux sous-graphiques ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sous-graphique 1 : Coûts (adapté vs depuis zéro)
    ax1.plot(df["Modification_%"], df["Coût_adapté"], marker='o', color='red', label='Coût adapté')
    ax1.plot(df["Modification_%"], df["Coût_depuis_zéro"], marker='s', color='blue', linestyle='--', label='Coût depuis zéro')
    ax1.set_xlabel("Taux de modification du réseau (%)")
    ax1.set_ylabel("Coût global")
    ax1.set_title("E8 - Coût adapté vs depuis zéro")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Sous-graphique 2 : Gain et réaffectations
    ax2.bar(df["Modification_%"] - 1, df["Gain_réoptimisation"], width=2, color='green', alpha=0.7, label='Gain (coût depuis zéro - adapté)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df["Modification_%"], df["Réaffectations"], marker='o', color='purple', linestyle='-', label='Réaffectations')
    ax2.set_xlabel("Taux de modification du réseau (%)")
    ax2.set_ylabel("Gain de la réoptimisation", color='green')
    ax2_twin.set_ylabel("Nombre de réaffectations", color='purple')
    ax2.set_title("E8 - Gain et réaffectations")
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E8_dynamic_network.png", dpi=300)
    plt.close()
    print(f"✅ Figure sauvegardée : {config.FIGURES_DIR / 'E8_dynamic_network.png'}")
    print("✅ E8 terminée.")

# ============================================================================
# E9 – MINIMA LOCAUX (effet des redémarrages)
# ============================================================================
def run_experiment_E9():
    """
    E9 - Minima locaux (effet des redémarrages)
    Mesure l'évolution du meilleur coût et du temps total en fonction du nombre de redémarrages.
    Utilise le scénario S4 (N=50, K=2) : réseau dense et faible nombre de canaux.
    Compare 1, 5, 10 et 20 redémarrages.
    Sortie : Meilleur coût et temps total selon redémarrages.
    """
    print("\n" + "="*60)
    print("🔬 E9 - MINIMA LOCAUX (effet des redémarrages)")
    print("="*60)
    
    # On utilise S4 : réseau dense (threshold=40) et K=2 (pénurie de canaux)
    scenario_name = "S4"
    data = all_data[scenario_name]
    N = data["N"]          # 50
    K = data["K"]          # 2
    W = np.array(data["W"])
    M = create_channel_interference_matrix(K)
    seed = data["seed"]
    
    # Nombres de redémarrages à tester
    restart_values = [1, 5, 10, 20]
    results = []
    
    for num_restarts in restart_values:
        # Mesurer le temps total de toutes les exécutions
        start_time = time.perf_counter()
        x_bd, _, _, _ = bdcenn_allocation(
            N, K, W, M=M,
            num_restarts=num_restarts,
            max_iter=50,
            seed=seed,
            verbose=False
        )
        elapsed = time.perf_counter() - start_time  # Temps total
        cost = compute_spectrum_energy(x_bd, W, M)
        conflicts = count_spectrum_conflicts(x_bd, W, M)
        results.append({
            "Redémarrages": num_restarts,
            "Coût": cost,
            "Conflits": conflicts,
            "Temps total (s)": elapsed
        })
        print(f"Redémarrages={num_restarts} : Coût={cost:.1f}, Conflits={conflicts}, Temps total={elapsed:.6f}s")
    
    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E9_restart_effect.csv", index=False)
    print(f"\n✅ Table des résultats sauvegardée dans {config.CSV_DIR / 'E9_restart_effect.csv'}")
    
    # --- Figure 1 : Évolution du coût et du temps avec les redémarrages ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Tracé du coût
    ax1.plot(df["Redémarrages"], df["Coût"], marker='o', color='red', linestyle='-', linewidth=2, markersize=8, label='Coût')
    ax1.set_xlabel("Nombre de redémarrages", fontsize=12)
    ax1.set_ylabel("Meilleur coût global", color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Deuxième axe pour le temps total
    ax2 = ax1.twinx()
    ax2.plot(df["Redémarrages"], df["Temps total (s)"], marker='s', color='blue', linestyle='--', linewidth=2, markersize=8, label='Temps total')
    ax2.set_ylabel("Temps total d'exécution (s)", color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Annotation des points
    for i, row in df.iterrows():
        ax1.annotate(f"{row['Coût']:.1f}", (row['Redémarrages'], row['Coût']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        ax2.annotate(f"{row['Temps total (s)']:.3f}s", (row['Redémarrages'], row['Temps total (s)']),
                     textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
    
    # Légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_title(f"E9 - Effet des redémarrages sur le coût et le temps (N={N}, K={K})", fontsize=14, fontweight='bold')
    
    # Note explicative
    note = "Le temps total est la somme du temps de tous les redémarrages."
    ax1.text(0.98, 0.02, note, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E9_restart_effect.png", dpi=300)
    plt.close()
    print(f"✅ Figure sauvegardée : {config.FIGURES_DIR / 'E9_restart_effect.png'}")
    
    # --- Figure 2 : Table des résultats ---
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ["Redémarrages", "Coût", "Conflits", "Temps total (s)"],
        [df["Redémarrages"][0], f"{df['Coût'][0]:.1f}", df["Conflits"][0], f"{df['Temps total (s)'][0]:.6f}"],
        [df["Redémarrages"][1], f"{df['Coût'][1]:.1f}", df["Conflits"][1], f"{df['Temps total (s)'][1]:.6f}"],
        [df["Redémarrages"][2], f"{df['Coût'][2]:.1f}", df["Conflits"][2], f"{df['Temps total (s)'][2]:.6f}"],
        [df["Redémarrages"][3], f"{df['Coût'][3]:.1f}", df["Conflits"][3], f"{df['Temps total (s)'][3]:.6f}"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style de l'en-tête
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title(f"E9 - Table des résultats (N={N}, K={K})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E9_restart_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Table des résultats sauvegardée : {config.FIGURES_DIR / 'E9_restart_table.png'}")
    
    print("✅ E9 terminée.")

# ============================================================================
# ORCHESTRATEUR PRINCIPAL
# ============================================================================
def run_all_experiments():
    """
    Lance toutes les expériences (E1, E3, E4, E6, E7, E8, E9).
    """
    print("\n" + "🚀 LANCEMENT DES EXPÉRIENCES SPÉCIFIQUES")
    print("="*80)
    
    run_experiment_E1()
    run_experiment_E3()
    run_experiment_E4()
    run_experiment_E6()
    run_experiment_E7()
    run_experiment_E8()
    run_experiment_E9()
    
    print("\n" + "="*80)
    print("✅ TOUTES LES EXPÉRIENCES SONT TERMINÉES.")
    print("="*80)

if __name__ == "__main__":
    run_all_experiments()