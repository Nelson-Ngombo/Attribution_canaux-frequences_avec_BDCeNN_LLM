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
from metrics import (
    create_channel_interference_matrix,
    compute_cochannel_cost,
    count_cochannel_conflicts,
    compute_adjacent_cost,
    count_adjacent_conflicts
)
import config


def run_validation():
    print("=" * 80)
    print("🚀 LANCEMENT DE LA VALIDATION APPROFONDIE")
    print(f"   - {len(all_data)} scénarios")
    print(f"   - {config.NUM_RUNS} répétitions par scénario")
    print("   - Méthodes : Random, Greedy (ordres variés), DSATUR, BD-CeNN (redémarrages)")
    print("   - Métriques : Co-canal (coût+conflits) et Adjacent (coût+conflits)")
    print("=" * 80)

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

            # ------------------------------------------------------------
            # 1. Random
            # ------------------------------------------------------------
            start = time.perf_counter()
            x_rand = np.random.randint(0, K, size=N)
            time_rand = time.perf_counter() - start

            cochannel_cost_rand = compute_cochannel_cost(x_rand, W)
            cochannel_conf_rand = count_cochannel_conflicts(x_rand, W)
            adjacent_cost_rand = compute_adjacent_cost(x_rand, W, M)
            adjacent_conf_rand = count_adjacent_conflicts(x_rand, W, M)
            used_rand = len(set(x_rand))

            raw_rows.append({
                "run_seed": run_seed,
                "scenario": name,
                "N": N,
                "K": K,
                "seed_scenario": seed_scenario,
                "method": "Random",
                "cochannel_cost": cochannel_cost_rand,
                "cochannel_conflicts": cochannel_conf_rand,
                "adjacent_cost": adjacent_cost_rand,
                "adjacent_conflicts": adjacent_conf_rand,
                "used_channels": used_rand,
                "time": time_rand,
                "initial_adjacent_cost": adjacent_cost_rand,   # Random initial = final
                "initial_cochannel_cost": cochannel_cost_rand
            })

            # ------------------------------------------------------------
            # 2. Greedy
            # ------------------------------------------------------------
            start = time.perf_counter()
            x_greedy = greedy_allocation(N, K, W, order=greedy_order, M=M)
            time_greedy = time.perf_counter() - start

            cochannel_cost_greedy = compute_cochannel_cost(x_greedy, W)
            cochannel_conf_greedy = count_cochannel_conflicts(x_greedy, W)
            adjacent_cost_greedy = compute_adjacent_cost(x_greedy, W, M)
            adjacent_conf_greedy = count_adjacent_conflicts(x_greedy, W, M)
            used_greedy = len(set(x_greedy))

            # Greedy part de la même initialisation que Random (même seed)
            initial_adj_greedy = adjacent_cost_rand
            initial_co_greedy = cochannel_cost_rand

            raw_rows.append({
                "run_seed": run_seed,
                "scenario": name,
                "N": N,
                "K": K,
                "seed_scenario": seed_scenario,
                "method": "Greedy",
                "cochannel_cost": cochannel_cost_greedy,
                "cochannel_conflicts": cochannel_conf_greedy,
                "adjacent_cost": adjacent_cost_greedy,
                "adjacent_conflicts": adjacent_conf_greedy,
                "used_channels": used_greedy,
                "time": time_greedy,
                "initial_adjacent_cost": initial_adj_greedy,
                "initial_cochannel_cost": initial_co_greedy
            })

            # ------------------------------------------------------------
            # 3. DSATUR
            # ------------------------------------------------------------
            start = time.perf_counter()
            x_dsatur = dsatur_allocation(N, K, W, M=M)
            time_dsatur = time.perf_counter() - start

            cochannel_cost_dsatur = compute_cochannel_cost(x_dsatur, W)
            cochannel_conf_dsatur = count_cochannel_conflicts(x_dsatur, W)
            adjacent_cost_dsatur = compute_adjacent_cost(x_dsatur, W, M)
            adjacent_conf_dsatur = count_adjacent_conflicts(x_dsatur, W, M)
            used_dsatur = len(set(x_dsatur))

            raw_rows.append({
                "run_seed": run_seed,
                "scenario": name,
                "N": N,
                "K": K,
                "seed_scenario": seed_scenario,
                "method": "DSATUR",
                "cochannel_cost": cochannel_cost_dsatur,
                "cochannel_conflicts": cochannel_conf_dsatur,
                "adjacent_cost": adjacent_cost_dsatur,
                "adjacent_conflicts": adjacent_conf_dsatur,
                "used_channels": used_dsatur,
                "time": time_dsatur,
                "initial_adjacent_cost": adjacent_cost_dsatur,
                "initial_cochannel_cost": cochannel_cost_dsatur
            })

            # ------------------------------------------------------------
            # 4. BD-CeNN
            # ------------------------------------------------------------
            x_bd, history_bd, t_bd, conf_bd = bdcenn_allocation(
                N, K, W, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=run_seed,
                verbose=False
            )

            cochannel_cost_bd = compute_cochannel_cost(x_bd, W)
            cochannel_conf_bd = count_cochannel_conflicts(x_bd, W)
            adjacent_cost_bd = compute_adjacent_cost(x_bd, W, M)
            adjacent_conf_bd = count_adjacent_conflicts(x_bd, W, M)
            used_bd = len(set(x_bd))

            # Coût initial : on refait une initialisation avec la même seed
            np.random.seed(run_seed)
            x_init = np.random.randint(0, K, size=N)
            initial_adj_bd = compute_adjacent_cost(x_init, W, M)
            initial_co_bd = compute_cochannel_cost(x_init, W)

            raw_rows.append({
                "run_seed": run_seed,
                "scenario": name,
                "N": N,
                "K": K,
                "seed_scenario": seed_scenario,
                "method": "BD-CeNN",
                "cochannel_cost": cochannel_cost_bd,
                "cochannel_conflicts": cochannel_conf_bd,
                "adjacent_cost": adjacent_cost_bd,
                "adjacent_conflicts": adjacent_conf_bd,
                "used_channels": used_bd,
                "time": t_bd,
                "initial_adjacent_cost": initial_adj_bd,
                "initial_cochannel_cost": initial_co_bd
            })

            # --- Sauvegarde de l'historique de convergence (adjacent cost) ---
            file_exists = os.path.isfile(convergence_file)
            with open(convergence_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['scenario', 'run_seed', 'iteration', 'adjacent_cost', 'seed_init'])
                for it, cost, alloc in history_bd:
                    writer.writerow([name, run_seed, it, cost, run_seed])

    print("\n✅ Toutes les exécutions sont terminées.")
    print("📊 Construction du DataFrame brut...")

    df_raw = pd.DataFrame(raw_rows)

    # --- Agrégation par (scenario, N, K, seed_scenario, method) ---
    grouped = df_raw.groupby(["scenario", "N", "K", "seed_scenario", "method"])
    df_agg = grouped.agg({
        "cochannel_cost": ["mean", "std", "min", "max", "median"],
        "cochannel_conflicts": ["mean", "std", "min", "max", "median"],
        "adjacent_cost": ["mean", "std", "min", "max", "median"],
        "adjacent_conflicts": ["mean", "std", "min", "max", "median"],
        "used_channels": ["mean", "std", "min", "max", "median"],
        "time": ["mean", "std", "min", "max", "median"],
        "initial_adjacent_cost": ["mean", "std", "min", "max", "median"],
        "initial_cochannel_cost": ["mean", "std", "min", "max", "median"]
    }).reset_index()

    # Aplatir les noms de colonnes
    df_agg.columns = [
        'scenario', 'N', 'K', 'seed_scenario', 'method',
        'cochannel_cost_mean', 'cochannel_cost_std',
        'cochannel_cost_min', 'cochannel_cost_max', 'cochannel_cost_median',
        'cochannel_conflicts_mean', 'cochannel_conflicts_std',
        'cochannel_conflicts_min', 'cochannel_conflicts_max', 'cochannel_conflicts_median',
        'adjacent_cost_mean', 'adjacent_cost_std',
        'adjacent_cost_min', 'adjacent_cost_max', 'adjacent_cost_median',
        'adjacent_conflicts_mean', 'adjacent_conflicts_std',
        'adjacent_conflicts_min', 'adjacent_conflicts_max', 'adjacent_conflicts_median',
        'used_channels_mean', 'used_channels_std',
        'used_channels_min', 'used_channels_max', 'used_channels_median',
        'time_mean', 'time_std',
        'time_min', 'time_max', 'time_median',
        'initial_adjacent_cost_mean', 'initial_adjacent_cost_std',
        'initial_adjacent_cost_min', 'initial_adjacent_cost_max', 'initial_adjacent_cost_median',
        'initial_cochannel_cost_mean', 'initial_cochannel_cost_std',
        'initial_cochannel_cost_min', 'initial_cochannel_cost_max', 'initial_cochannel_cost_median'
    ]

    # Arrondi des canaux utilisés
    for col in ['used_channels_mean', 'used_channels_std', 'used_channels_min',
                'used_channels_max', 'used_channels_median']:
        df_agg[col] = df_agg[col].round(0)

    # --- Pivot : une colonne par (méthode, métrique) ---
    pivot = df_agg.pivot_table(
        index=['scenario', 'N', 'K', 'seed_scenario'],
        columns='method',
        values=[
            'cochannel_cost_mean', 'cochannel_cost_std',
            'cochannel_cost_min', 'cochannel_cost_max', 'cochannel_cost_median',
            'cochannel_conflicts_mean', 'cochannel_conflicts_std',
            'cochannel_conflicts_min', 'cochannel_conflicts_max', 'cochannel_conflicts_median',
            'adjacent_cost_mean', 'adjacent_cost_std',
            'adjacent_cost_min', 'adjacent_cost_max', 'adjacent_cost_median',
            'adjacent_conflicts_mean', 'adjacent_conflicts_std',
            'adjacent_conflicts_min', 'adjacent_conflicts_max', 'adjacent_conflicts_median',
            'used_channels_mean', 'used_channels_std',
            'used_channels_min', 'used_channels_max', 'used_channels_median',
            'time_mean', 'time_std',
            'time_min', 'time_max', 'time_median',
            'initial_adjacent_cost_mean', 'initial_adjacent_cost_std',
            'initial_adjacent_cost_min', 'initial_adjacent_cost_max', 'initial_adjacent_cost_median',
            'initial_cochannel_cost_mean', 'initial_cochannel_cost_std',
            'initial_cochannel_cost_min', 'initial_cochannel_cost_max', 'initial_cochannel_cost_median'
        ]
    )

    pivot.columns = [f"{method}_{metric}" for metric, method in pivot.columns]
    df_summary = pivot.reset_index()

    # --- Ordre des colonnes ---
    ordered_cols = ['scenario', 'N', 'K', 'seed_scenario']
    methods = ['Random', 'Greedy', 'DSATUR', 'BD-CeNN']
    metric_suffixes = [
        'cochannel_cost_mean', 'cochannel_cost_std',
        'cochannel_cost_min', 'cochannel_cost_max', 'cochannel_cost_median',
        'cochannel_conflicts_mean', 'cochannel_conflicts_std',
        'cochannel_conflicts_min', 'cochannel_conflicts_max', 'cochannel_conflicts_median',
        'adjacent_cost_mean', 'adjacent_cost_std',
        'adjacent_cost_min', 'adjacent_cost_max', 'adjacent_cost_median',
        'adjacent_conflicts_mean', 'adjacent_conflicts_std',
        'adjacent_conflicts_min', 'adjacent_conflicts_max', 'adjacent_conflicts_median',
        'used_channels_mean', 'used_channels_std',
        'used_channels_min', 'used_channels_max', 'used_channels_median',
        'time_mean', 'time_std',
        'time_min', 'time_max', 'time_median',
        'initial_adjacent_cost_mean', 'initial_adjacent_cost_std',
        'initial_adjacent_cost_min', 'initial_adjacent_cost_max', 'initial_adjacent_cost_median',
        'initial_cochannel_cost_mean', 'initial_cochannel_cost_std',
        'initial_cochannel_cost_min', 'initial_cochannel_cost_max', 'initial_cochannel_cost_median'
    ]
    for method in methods:
        for suf in metric_suffixes:
            col = f"{method}_{suf}"
            if col in df_summary.columns:
                ordered_cols.append(col)
    df_summary = df_summary[ordered_cols]

    print("💾 Sauvegarde du fichier Excel...")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n✅ Validation terminée !")
    print(f"📁 Fichier Excel : {excel_file}")
    print(f"📁 Historique convergence (adjacent cost) : {convergence_file}")


if __name__ == "__main__":
    print(f"📂 Chargement des scénarios : {len(all_data)} scénarios trouvés.")
    if len(all_data) < 50:
        print("⚠️  Attention : moins de 50 scénarios détectés.")
    run_validation()