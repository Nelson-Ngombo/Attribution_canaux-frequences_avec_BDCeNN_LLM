# validation.py
import numpy as np
import pandas as pd
import time
import os
import csv
import json
from datetime import datetime
from baselines import greedy_allocation, dsatur_allocation
from bdcenn_solver import bdcenn_allocation
from metrics import (
    create_channel_interference_matrix,
    compute_cochannel_cost,
    count_cochannel_conflicts,
    compute_adjacent_cost,
    count_adjacent_conflicts,
    compute_confidence_interval
)
import config


def run_validation():
    print("=" * 80)
    print("🚀 LANCEMENT DE LA VALIDATION APPROFONDIE")
    print(f"   - {len(config.SCENARIOS)} scénarios")
    print(f"   - {config.NUM_RUNS} répétitions (chacune avec une topologie différente)")
    print("   - Méthodes : Random, Greedy (ordres variés), DSATUR, BD-CeNN (redémarrages)")
    print("   - Métriques : Co-canal (coût+conflits) et Adjacent (coût+conflits)")
    print("=" * 80)

    # --- Charger les données depuis le JSON ---
    try:
        with open(config.SCENARIOS_FILE, "r") as f:
            all_instances = json.load(f)
        print("✅ Données chargées depuis scenarios_data.json")
    except FileNotFoundError:
        print("❌ Fichier scenarios_data.json introuvable. Veuillez d'abord exécuter data_generator.py")
        return

    raw_rows = []
    scenario_list = list(config.SCENARIOS.items())
    total_runs = len(scenario_list) * config.NUM_RUNS
    current_run = 0

    convergence_file = config.CONVERGENCE_HISTORY_FILE
    convergence_cochannel_file = config.CONVERGENCE_HISTORY_COCHANNEL_FILE
    excel_file = config.VALIDATION_EXCEL_FILE

    # Pour stocker le nombre de balayages par run pour BD-CeNN
    iterations_data = []

    # Supprimer les anciens fichiers d'historique s'ils existent
    if os.path.exists(convergence_file):
        os.remove(convergence_file)
    if os.path.exists(convergence_cochannel_file):
        os.remove(convergence_cochannel_file)

    for run_seed in range(1, config.NUM_RUNS + 1):
        print(f"\n--- Run {run_seed}/{config.NUM_RUNS} ---")
        topology_seed = run_seed  # on utilise directement la seed 1..30
        np.random.seed(run_seed)  # pour les ordres et initialisations

        for name, params in scenario_list:
            current_run += 1
            if current_run % 50 == 0:
                print(f"  Progression : {current_run}/{total_runs} runs effectués")

            N = params["N"]
            K = params["K"]

            # Récupérer l'instance correspondant à topology_seed pour ce scénario
            instances = all_instances.get(name, {})
            inst_key = str(topology_seed)
            if inst_key not in instances:
                # Si la seed n'existe pas, on prend la première disponible
                inst_key = list(instances.keys())[0]
                print(f"    ⚠️ Seed {topology_seed} non trouvée pour {name}, utilisation de {inst_key}")
            inst = instances[inst_key]
            W = np.array(inst["W"])
            seed_scenario = inst["seed"]

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
                "initial_adjacent_cost": adjacent_cost_rand,
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
                "initial_adjacent_cost": adjacent_cost_rand,
                "initial_cochannel_cost": cochannel_cost_rand
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
            # On capture le nombre d'itérations effectuées
            x_bd, history_bd, t_bd, conf_bd = bdcenn_allocation(
                N, K, W, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=run_seed,
                verbose=False
            )
            # Nombre de balayages effectués = len(history_bd) - 1
            iterations_used = len(history_bd) - 1

            cochannel_cost_bd = compute_cochannel_cost(x_bd, W)
            cochannel_conf_bd = count_cochannel_conflicts(x_bd, W)
            adjacent_cost_bd = compute_adjacent_cost(x_bd, W, M)
            adjacent_conf_bd = count_adjacent_conflicts(x_bd, W, M)
            used_bd = len(set(x_bd))

            # Coût initial
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

            # Stocker le nombre d'itérations pour BD-CeNN
            iterations_data.append({
                "run_seed": run_seed,
                "scenario": name,
                "iterations_used": iterations_used,
                "max_iter": config.MAX_ITER_BD
            })

            # --- Sauvegarde de l'historique de convergence (adjacent cost) ---
            file_exists = os.path.isfile(convergence_file)
            with open(convergence_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['scenario', 'run_seed', 'iteration', 'adjacent_cost', 'seed_init'])
                for it, cost, alloc in history_bd:
                    writer.writerow([name, run_seed, it, cost, run_seed])

            # --- Sauvegarde de l'historique de convergence (co-canal cost) ---
            # On recalcule le coût co-canal pour chaque itération
            cochannel_history = []
            for it, cost_adj, alloc in history_bd:
                co_cost = compute_cochannel_cost(alloc, W)
                cochannel_history.append((it, co_cost))

            file_exists_co = os.path.isfile(convergence_cochannel_file)
            with open(convergence_cochannel_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists_co:
                    writer.writerow(['scenario', 'run_seed', 'iteration', 'cochannel_cost', 'seed_init'])
                for it, co_cost in cochannel_history:
                    writer.writerow([name, run_seed, it, co_cost, run_seed])

    print("\n✅ Toutes les exécutions sont terminées.")
    print("📊 Construction des DataFrames...")

    df_raw = pd.DataFrame(raw_rows)
    df_iter = pd.DataFrame(iterations_data)

    # --- Agrégation par (scenario, N, K, method) pour les statistiques ---
    grouped = df_raw.groupby(["scenario", "N", "K", "method"])
    # On agrège pour obtenir : mean, std, median, min, max, et on calcule IC
    agg_funcs = {
        "cochannel_cost": ["mean", "std", "median", "min", "max"],
        "cochannel_conflicts": ["mean", "std", "median", "min", "max"],
        "adjacent_cost": ["mean", "std", "median", "min", "max"],
        "adjacent_conflicts": ["mean", "std", "median", "min", "max"],
        "used_channels": ["mean", "std", "median", "min", "max"],
        "time": ["mean", "std", "median", "min", "max"],
        "initial_adjacent_cost": ["mean", "std", "median", "min", "max"],
        "initial_cochannel_cost": ["mean", "std", "median", "min", "max"]
    }
    df_agg = grouped.agg(agg_funcs).reset_index()
    # Aplatir les colonnes
    df_agg.columns = ['scenario', 'N', 'K', 'method'] + [f"{col}_{stat}" for col, stat in df_agg.columns[4:]]
    
    # Calcul de l'intervalle de confiance pour chaque métrique (sur les données brutes)
    ci_data = []
    for (scenario, N, K, method), group in df_raw.groupby(["scenario", "N", "K", "method"]):
        row = {"scenario": scenario, "N": N, "K": K, "method": method}
        for metric in ["cochannel_cost", "cochannel_conflicts", "adjacent_cost", 
                       "adjacent_conflicts", "used_channels", "time", 
                       "initial_adjacent_cost", "initial_cochannel_cost"]:
            values = group[metric].values
            ci_low, ci_high = compute_confidence_interval(values)
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        ci_data.append(row)
    df_ci = pd.DataFrame(ci_data)

    # --- Exporter les CSV de statistiques ---
    # 1. Tableau des médianes
    median_cols = ['scenario', 'N', 'K', 'method'] + [f"{metric}_median" for metric in 
                       ["cochannel_cost", "cochannel_conflicts", "adjacent_cost", 
                        "adjacent_conflicts", "used_channels", "time", 
                        "initial_adjacent_cost", "initial_cochannel_cost"]]
    df_median = df_agg[median_cols].copy()
    df_median.to_csv(config.STATS_MEDIAN_CSV, index=False)

    # 2. Tableau des minima
    min_cols = ['scenario', 'N', 'K', 'method'] + [f"{metric}_min" for metric in 
                       ["cochannel_cost", "cochannel_conflicts", "adjacent_cost", 
                        "adjacent_conflicts", "used_channels", "time", 
                        "initial_adjacent_cost", "initial_cochannel_cost"]]
    df_min = df_agg[min_cols].copy()
    df_min.to_csv(config.STATS_MIN_CSV, index=False)

    # 3. Tableau des maxima
    max_cols = ['scenario', 'N', 'K', 'method'] + [f"{metric}_max" for metric in 
                       ["cochannel_cost", "cochannel_conflicts", "adjacent_cost", 
                        "adjacent_conflicts", "used_channels", "time", 
                        "initial_adjacent_cost", "initial_cochannel_cost"]]
    df_max = df_agg[max_cols].copy()
    df_max.to_csv(config.STATS_MAX_CSV, index=False)

    # 4. Tableau des intervalles de confiance (on garde ci_low et ci_high)
    df_ci.to_csv(config.STATS_CI_CSV, index=False)

    # 5. Tableau du nombre de balayages (moyenne, écart-type, min, max, médiane)
    iter_stats = df_iter.groupby("scenario").agg({
        "iterations_used": ["mean", "std", "min", "max", "median"]
    }).reset_index()
    iter_stats.columns = ['scenario', 'iter_mean', 'iter_std', 'iter_min', 'iter_max', 'iter_median']
    iter_stats.to_csv(config.STATS_ITERATIONS_CSV, index=False)

    print("✅ Statistiques supplémentaires exportées en CSV.")

    # --- Sauvegarde du fichier Excel principal ---
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
        df_agg.to_excel(writer, sheet_name='Summary_Stats', index=False)
        df_ci.to_excel(writer, sheet_name='Confidence_Intervals', index=False)
        iter_stats.to_excel(writer, sheet_name='BD_Iterations', index=False)

    print(f"\n✅ Validation terminée !")
    print(f"📁 Fichier Excel : {excel_file}")
    print(f"📁 Statistiques CSV : {config.STATS_MEDIAN_CSV}, {config.STATS_MIN_CSV}, {config.STATS_MAX_CSV}, {config.STATS_CI_CSV}, {config.STATS_ITERATIONS_CSV}")
    print(f"📁 Historique convergence (adjacent) : {convergence_file}")
    print(f"📁 Historique convergence (co-canal) : {convergence_cochannel_file}")

if __name__ == "__main__":
    print(f"📂 Scénarios définis : {len(config.SCENARIOS)} scénarios.")
    run_validation()