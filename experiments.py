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
    Vérifie que la chaîne fonctionne sur un cas contrôlable (S1).
    Affiche l'affectation des canaux pour Random, Greedy, DSATUR, BD-CeNN.
    """
    print("\n" + "="*60)
    print("🔬 E1 - VÉRIFICATION VISUELLE (Petit graphe)")
    print("="*60)
    
    scenario_name = "S1"
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W = np.array(data["W"])
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
    
    # Affichage des résultats
    print("\n--- ALLOCATIONS FINALES ---")
    print(f"Random   : x = {x_rand.tolist()}  | Coût = {cost_rand:.1f}  | Conflits = {conflicts_rand}")
    print(f"Greedy   : x = {x_greedy.tolist()}  | Coût = {cost_greedy:.1f}  | Conflits = {conflicts_greedy}")
    print(f"DSATUR   : x = {x_dsatur.tolist()}  | Coût = {cost_dsatur:.1f}  | Conflits = {conflicts_dsatur}")
    print(f"BD-CeNN  : x = {x_bd.tolist()}  | Coût = {cost_bd:.1f}  | Conflits = {conflicts_bd}")
    
    # Sauvegarde des résultats
    df = pd.DataFrame({
        "Méthode": ["Random", "Greedy", "DSATUR", "BD-CeNN"],
        "Allocation": [x_rand.tolist(), x_greedy.tolist(), x_dsatur.tolist(), x_bd.tolist()],
        "Coût": [cost_rand, cost_greedy, cost_dsatur, cost_bd],
        "Conflits": [conflicts_rand, conflicts_greedy, conflicts_dsatur, conflicts_bd]
    })
    df.to_csv(config.CSV_DIR / "E1_visual_check.csv", index=False)
    print(f"\n✅ Résultats sauvegardés dans {config.CSV_DIR / 'E1_visual_check.csv'}")
    print("✅ E1 terminée.")

# ============================================================================
# E3 – IMPACT DU NOMBRE DE CANAUX (K)
# ============================================================================
def run_experiment_E3():
    """
    E3 - Impact du nombre de canaux
    Mesure l'évolution du coût et des conflits en fonction de K.
    """
    print("\n" + "="*60)
    print("🔬 E3 - IMPACT DU NOMBRE DE CANAUX (K)")
    print("="*60)
    
    scenario_name = "S2"  # Réseau moyen
    data = all_data[scenario_name]
    N = data["N"]
    W = np.array(data["W"])
    seed = config.SEED_BASE
    K_values = [2, 3, 4, 5, 6, 8]
    
    results = []
    
    for K in K_values:
        M = create_channel_interference_matrix(K)
        # Exécuter BD-CeNN pour ce K
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
    Mesure l'évolution du coût en fonction de la densité du graphe (threshold).
    """
    print("\n" + "="*60)
    print("🔬 E4 - IMPACT DE LA DENSITÉ")
    print("="*60)
    
    # On utilise S2 (N=30) avec différents thresholds
    scenario_name = "S2"
    base_data = all_data[scenario_name]
    N = base_data["N"]
    K = base_data["K"]
    seed = config.SEED_BASE
    
    thresholds = [20, 30, 40, 50, 60, 80]
    densities = []
    costs = []
    
    for th in thresholds:
        # Re-générer W avec ce threshold
        np.random.seed(base_data["seed"])
        positions = np.random.rand(N, 2) * 150  # area fixe
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
        densities.append(density)
        
        M = create_channel_interference_matrix(K)
        x_bd, _, _, _ = bdcenn_allocation(N, K, W, M=M, num_restarts=10, max_iter=50, seed=seed)
        cost = compute_spectrum_energy(x_bd, W, M)
        costs.append(cost)
        print(f"Threshold={th}, Densité={density:.3f}, Coût={cost:.1f}")
    
    df = pd.DataFrame({"Threshold": thresholds, "Densité": densities, "Coût": costs})
    df.to_csv(config.CSV_DIR / "E4_impact_density.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["Densité"], df["Coût"], marker='o', color='red', linestyle='-')
    plt.xlabel("Densité du graphe")
    plt.ylabel("Coût global")
    plt.title(f"E4 - Impact de la densité sur le coût ({scenario_name}, N={N}, K={K})")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E4_impact_density.png", dpi=300)
    plt.close()
    print(f"✅ Figure sauvegardée : {config.FIGURES_DIR / 'E4_impact_density.png'}")
    print("✅ E4 terminée.")

# ============================================================================
# E6 – SCALABILITÉ (temps vs N)
# ============================================================================
def run_experiment_E6():
    """
    E6 - Scalabilité
    Mesure le temps de calcul en fonction du nombre de cellules N.
    """
    print("\n" + "="*60)
    print("🔬 E6 - SCALABILITÉ (temps vs N)")
    print("="*60)
    
    # Utiliser S2, S3, S5 et créer des cas plus grands
    scenario_names = ["S2", "S3", "S5"]
    N_values = []
    times = []
    costs = []
    
    for name in scenario_names:
        data = all_data[name]
        N = data["N"]
        K = data["K"]
        W = np.array(data["W"])
        M = create_channel_interference_matrix(K)
        seed = data["seed"]
        
        # Exécuter BD-CeNN et mesurer le temps
        start = time.perf_counter()
        x_bd, _, _, _ = bdcenn_allocation(N, K, W, M=M, num_restarts=10, max_iter=50, seed=seed)
        elapsed = time.perf_counter() - start
        cost = compute_spectrum_energy(x_bd, W, M)
        
        N_values.append(N)
        times.append(elapsed)
        costs.append(cost)
        print(f"N={N}, K={K}, Temps={elapsed:.6f}s, Coût={cost:.1f}")
    
    # Ajouter N=150 et N=200 (création ad-hoc)
    for N in [150, 200]:
        K = 8
        area = 400 if N > 150 else 300
        np.random.seed(42 + N)
        positions = np.random.rand(N, 2) * area
        W = np.zeros((N, N))
        threshold = 50
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
        M = create_channel_interference_matrix(K)
        start = time.perf_counter()
        x_bd, _, _, _ = bdcenn_allocation(N, K, W, M=M, num_restarts=10, max_iter=50, seed=42)
        elapsed = time.perf_counter() - start
        cost = compute_spectrum_energy(x_bd, W, M)
        N_values.append(N)
        times.append(elapsed)
        costs.append(cost)
        print(f"N={N}, K={K}, Temps={elapsed:.6f}s, Coût={cost:.1f}")
    
    df = pd.DataFrame({"N": N_values, "Temps": times, "Coût": costs})
    df = df.sort_values("N")
    df.to_csv(config.CSV_DIR / "E6_scalability.csv", index=False)
    
    # Figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["N"], df["Temps"], marker='o', color='blue', label='Temps')
    ax1.set_xlabel("Nombre de cellules N")
    ax1.set_ylabel("Temps d'exécution (s)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["N"], df["Coût"], marker='s', color='red', linestyle='--', label='Coût')
    ax2.set_ylabel("Coût global", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title("E6 - Scalabilité : temps et coût en fonction de N")
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
    E9 - Minima locaux
    Compare le coût final pour 1, 5, 10, 20 redémarrages.
    """
    print("\n" + "="*60)
    print("🔬 E9 - EFFET DES REDÉMARRAGES (minima locaux)")
    print("="*60)
    
    scenario_name = "S3"  # Réseau dense (difficile)
    data = all_data[scenario_name]
    N = data["N"]
    K = data["K"]
    W = np.array(data["W"])
    M = create_channel_interference_matrix(K)
    seed = data["seed"]
    
    restart_values = config.RESTART_EXPERIMENT_VALUES
    results = []
    
    for num_restarts in restart_values:
        # Exécuter BD-CeNN avec ce nombre de redémarrages
        start_time = time.perf_counter()
        x_bd, _, elapsed, _ = bdcenn_allocation(
            N, K, W, M=M,
            num_restarts=num_restarts,
            max_iter=50,
            seed=seed,
            verbose=False
        )
        cost = compute_spectrum_energy(x_bd, W, M)
        conflicts = count_spectrum_conflicts(x_bd, W, M)
        results.append({
            "Redémarrages": num_restarts,
            "Coût": cost,
            "Conflits": conflicts,
            "Temps": elapsed
        })
        print(f"Redémarrages={num_restarts} : Coût={cost:.1f}, Conflits={conflicts}, Temps={elapsed:.6f}s")
    
    df = pd.DataFrame(results)
    df.to_csv(config.CSV_DIR / "E9_restart_effect.csv", index=False)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["Redémarrages"], df["Coût"], marker='o', color='red', label='Coût')
    ax1.set_xlabel("Nombre de redémarrages")
    ax1.set_ylabel("Coût global", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["Redémarrages"], df["Conflits"], marker='s', color='blue', linestyle='--', label='Conflits')
    ax2.set_ylabel("Conflits spectraux", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title(f"E9 - Effet des redémarrages ({scenario_name}, N={N}, K={K})")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "E9_restart_effect.png", dpi=300)
    plt.close()
    print(f"✅ Figure sauvegardée : {config.FIGURES_DIR / 'E9_restart_effect.png'}")
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