# validation.py
import numpy as np
import pandas as pd
import time
import os
import csv
from datetime import datetime
from data_generator import all_data
from baselines import greedy_allocation, dsatur_allocation
from bdcenn_solver import bdcenn_allocation
from metrics import compute_metrics, create_channel_interference_matrix, compute_spectrum_energy, count_spectrum_conflicts
import config

def run_validation():
    print("="*80)
    print("🚀 LANCEMENT DE LA VALIDATION APPROFONDIE")
    print(f"   - {len(all_data)} scénarios")
    print(f"   - {config.NUM_RUNS} répétitions par scénario")
    print("   - Méthodes : Random, Greedy (ordres variés), DSATUR, BD-CeNN (redémarrages)")
    print("   - Métriques : Conflits (spectraux), Canaux utilisés, Coût global (avec M)")
    print("="*80)

    raw_rows = []
    scenario_list = list(all_data.items())
    total_runs = len(scenario_list) * config.NUM_RUNS
    current_run = 0

    convergence_file = config.CONVERGENCE_HISTORY_FILE
    excel_file = config.VALIDATION_EXCEL_FILE

    if os.path.exists(convergence_file):
        os.remove(convergence_file)

    for run_seed in range(1, config.NUM_RUNS + 1):
        print(f"\n--- Run {run_seed}/{config.NUM_RUNS} ---")
        np.random.seed(run_seed)

        for name, data in scenario_list:
            current_run += 1
            if current_run % 50 == 0:
                print(f"  Progression : {current_run}/{total_runs} runs effectués")

            N = data["N"]
            K = data["K"]
            W = np.array(data["W"])
            seed_scenario = data["seed"]
            M = create_channel_interference_matrix(K)

            greedy_order = np.random.permutation(N).tolist()

            # ---------- 1. Random ----------
            start = time.perf_counter()
            x_rand = np.random.randint(0, K, size=N)
            time_rand = time.perf_counter() - start
            m_rand = compute_metrics(x_rand, W)
            global_cost_rand = compute_spectrum_energy(x_rand, W, M)
            spectrum_conflicts_rand = count_spectrum_conflicts(x_rand, W, M)
            # Coût initial : avant optimisation (identique pour toutes les méthodes car random)
            initial_cost_rand = global_cost_rand

            raw_rows.append({
                "run_seed": run_seed,
                "scenario": name,
                "N": N,
                "K": K,
                "seed_scenario": seed_scenario,
                "method": "Random",
                "spectrum_conflicts": spectrum_conflicts_rand,
                "used_channels": m_rand["used_channels"],
                "time": time_rand,
                "global_cost": global_cost_rand,
                "initial_cost": initial_cost_rand
            })

            # ---------- 2. Greedy (ordre aléatoire) ----------
            start = time.perf_counter()
            x_greedy = greedy_allocation(N, K, W, order=greedy_order, M=M)
            time_greedy = time.perf_counter() - start
            m_greedy = compute_metrics(x_greedy, W)
            global_cost_greedy = compute_spectrum_energy(x_greedy, W, M)
            spectrum_conflicts_greedy = count_spectrum_conflicts(x_greedy, W, M)
            # Coût initial : même initialisation que Random (car on part de la même allocation)
            initial_cost_greedy = global_cost_rand

            raw_rows.append({
                "run_seed": run_seed,
                "scenario": name,
                "N": N,
                "K": K,
                "seed_scenario": seed_scenario,
                "method": "Greedy",
                "spectrum_conflicts": spectrum_conflicts_greedy,
                "used_channels": m_greedy["used_channels"],
                "time": time_greedy,
                "global_cost": global_cost_greedy,
                "initial_cost": initial_cost_greedy
            })

            # ---------- 3. DSATUR ----------
            start = time.perf_counter()
            x_dsatur = dsatur_allocation(N, K, W, M=M)
            time_dsatur = time.perf_counter() - start
            m_dsatur = compute_metrics(x_dsatur, W)
            global_cost_dsatur = compute_spectrum_energy(x_dsatur, W, M)
            spectrum_conflicts_dsatur = count_spectrum_conflicts(x_dsatur, W, M)
            # DSATUR n'utilise pas d'initialisation aléatoire, coût initial = coût final
            initial_cost_dsatur = global_cost_dsatur

            raw_rows.append({
                "run_seed": run_seed,
                "scenario": name,
                "N": N,
                "K": K,
                "seed_scenario": seed_scenario,
                "method": "DSATUR",
                "spectrum_conflicts": spectrum_conflicts_dsatur,
                "used_channels": m_dsatur["used_channels"],
                "time": time_dsatur,
                "global_cost": global_cost_dsatur,
                "initial_cost": initial_cost_dsatur
            })

            # ---------- 4. BD-CeNN (redémarrages multiples) ----------
            x_bd, history_bd, t_bd, conf_bd = bdcenn_allocation(
                N, K, W, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=run_seed,
                verbose=False
            )
            m_bd = compute_metrics(x_bd, W)
            global_cost_bd = compute_spectrum_energy(x_bd, W, M)
            spectrum_conflicts_bd = count_spectrum_conflicts(x_bd, W, M)
            # Coût initial : calculé sur la première initialisation (avant redémarrages)
            # On refait une initialisation avec la même seed pour obtenir le coût initial
            np.random.seed(run_seed)
            x_init = np.random.randint(0, K, size=N)
            initial_cost_bd = compute_spectrum_energy(x_init, W, M)

            raw_rows.append({
                "run_seed": run_seed,
                "scenario": name,
                "N": N,
                "K": K,
                "seed_scenario": seed_scenario,
                "method": "BD-CeNN",
                "spectrum_conflicts": spectrum_conflicts_bd,
                "used_channels": m_bd["used_channels"],
                "time": t_bd,
                "global_cost": global_cost_bd,
                "initial_cost": initial_cost_bd
            })

            # ---------- Sauvegarde de l'historique (TOUTES les runs) ----------
            # On ajoute une colonne run_seed pour distinguer les runs
            file_exists = os.path.isfile(convergence_file)
            with open(convergence_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['scenario', 'run_seed', 'iteration', 'global_cost', 'seed_init'])
                for it, cost, alloc in history_bd:
                    writer.writerow([name, run_seed, it, cost, run_seed])

    print("\n✅ Toutes les exécutions sont terminées.")
    print("📊 Construction du DataFrame brut...")

    df_raw = pd.DataFrame(raw_rows)

    # Agrégation avec statistiques (incluant initial_cost)
    grouped = df_raw.groupby(["scenario", "N", "K", "seed_scenario", "method"])
    df_agg = grouped.agg({
        "spectrum_conflicts": ["mean", "std", "min", "max", "median"],
        "used_channels": ["mean", "std", "min", "max", "median"],
        "time": ["mean", "std", "min", "max", "median"],
        "global_cost": ["mean", "std", "min", "max", "median"],
        "initial_cost": ["mean", "std", "min", "max", "median"]
    }).reset_index()

    # Aplatir les noms de colonnes
    df_agg.columns = [
        'scenario', 'N', 'K', 'seed_scenario', 'method',
        'spectrum_conflicts_mean', 'spectrum_conflicts_std',
        'spectrum_conflicts_min', 'spectrum_conflicts_max', 'spectrum_conflicts_median',
        'used_channels_mean', 'used_channels_std',
        'used_channels_min', 'used_channels_max', 'used_channels_median',
        'time_mean', 'time_std',
        'time_min', 'time_max', 'time_median',
        'global_cost_mean', 'global_cost_std',
        'global_cost_min', 'global_cost_max', 'global_cost_median',
        'initial_cost_mean', 'initial_cost_std',
        'initial_cost_min', 'initial_cost_max', 'initial_cost_median'
    ]

    # Arrondi des canaux
    for col in ['used_channels_mean', 'used_channels_std', 'used_channels_min', 'used_channels_max', 'used_channels_median']:
        df_agg[col] = df_agg[col].round(0)

    # Pivot
    pivot = df_agg.pivot_table(
        index=['scenario', 'N', 'K', 'seed_scenario'],
        columns='method',
        values=[
            'spectrum_conflicts_mean', 'spectrum_conflicts_std',
            'spectrum_conflicts_min', 'spectrum_conflicts_max', 'spectrum_conflicts_median',
            'used_channels_mean', 'used_channels_std',
            'used_channels_min', 'used_channels_max', 'used_channels_median',
            'time_mean', 'time_std',
            'time_min', 'time_max', 'time_median',
            'global_cost_mean', 'global_cost_std',
            'global_cost_min', 'global_cost_max', 'global_cost_median',
            'initial_cost_mean', 'initial_cost_std',
            'initial_cost_min', 'initial_cost_max', 'initial_cost_median'
        ]
    )

    pivot.columns = [f"{method}_{metric}" for metric, method in pivot.columns]
    df_summary = pivot.reset_index()

    # Ordre des colonnes
    ordered_cols = ['scenario', 'N', 'K', 'seed_scenario']
    methods = ['Random', 'Greedy', 'DSATUR', 'BD-CeNN']
    metrics = [
        'spectrum_conflicts_mean', 'spectrum_conflicts_std',
        'spectrum_conflicts_min', 'spectrum_conflicts_max', 'spectrum_conflicts_median',
        'used_channels_mean', 'used_channels_std',
        'used_channels_min', 'used_channels_max', 'used_channels_median',
        'time_mean', 'time_std',
        'time_min', 'time_max', 'time_median',
        'global_cost_mean', 'global_cost_std',
        'global_cost_min', 'global_cost_max', 'global_cost_median',
        'initial_cost_mean', 'initial_cost_std',
        'initial_cost_min', 'initial_cost_max', 'initial_cost_median'
    ]
    for method in methods:
        for metric in metrics:
            col = f"{method}_{metric}"
            if col in df_summary.columns:
                ordered_cols.append(col)
    df_summary = df_summary[ordered_cols]

    print("💾 Sauvegarde du fichier Excel...")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n✅ Validation terminée !")
    print(f"📁 Fichier Excel : {excel_file}")
    print(f"📁 Historique convergence (toutes les runs) : {convergence_file}")

if __name__ == "__main__":
    print(f"📂 Chargement des scénarios : {len(all_data)} scénarios trouvés.")
    if len(all_data) < 50:
        print("⚠️  Attention : moins de 50 scénarios détectés.")
    run_validation()