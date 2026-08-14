# validation.py
import numpy as np
import pandas as pd
import time
import os
import csv
import json
from datetime import datetime
from baselines import greedy_allocation, dsatur_allocation, random_allocation
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
    print("   - Méthodes : Random, Greedy, DSATUR, BD-CeNN (redémarrages)")
    print("   - Deux optimisations : Co-canal (sans M) et Adjacent (avec M)")
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

    convergence_adj_file = config.CONVERGENCE_HISTORY_FILE
    convergence_co_file = config.CONVERGENCE_HISTORY_COCHANNEL_FILE
    excel_file = config.VALIDATION_EXCEL_FILE

    # Pour stocker le nombre de balayages par run pour BD-CeNN
    iterations_data_adj = []
    iterations_data_co = []

    # Supprimer les anciens fichiers d'historique s'ils existent
    if os.path.exists(convergence_adj_file):
        os.remove(convergence_adj_file)
    if os.path.exists(convergence_co_file):
        os.remove(convergence_co_file)

    for run_seed in range(1, config.NUM_RUNS + 1):
        print(f"\n--- Run {run_seed}/{config.NUM_RUNS} ---")
        topology_seed = run_seed
        np.random.seed(run_seed)

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
                inst_key = list(instances.keys())[0]
                print(f"    ⚠️ Seed {topology_seed} non trouvée pour {name}, utilisation de {inst_key}")
            inst = instances[inst_key]
            W = np.array(inst["W"])
            seed_scenario = inst["seed"]

            M = create_channel_interference_matrix(K)
            greedy_order = np.random.permutation(N).tolist()

            # ================================================================
            # 1. Random : générer UNE SEULE affectation aléatoire
            #    On évalue ensuite cette même affectation avec les deux fonctions de coût.
            # ================================================================
            start = time.perf_counter()
            x_rand = random_allocation(N, K)
            time_rand = time.perf_counter() - start

            # Métriques co-canal sur x_rand
            cochannel_cost_rand = compute_cochannel_cost(x_rand, W)
            cochannel_conf_rand = count_cochannel_conflicts(x_rand, W)

            # Métriques adjacentes sur la MÊME affectation x_rand
            adjacent_cost_rand = compute_adjacent_cost(x_rand, W, M)
            adjacent_conf_rand = count_adjacent_conflicts(x_rand, W, M)

            # Nombre de canaux utilisés (identique pour les deux)
            used_rand = len(set(x_rand))

            # ================================================================
            # 2. Greedy : même ordre de parcours pour les deux objectifs
            #    greedy_order est partagé (créé une fois avant).
            # ================================================================
            start = time.perf_counter()
            x_greedy_co = greedy_allocation(N, K, W, order=greedy_order, M=None)
            time_greedy_co = time.perf_counter() - start
            cochannel_cost_greedy = compute_cochannel_cost(x_greedy_co, W)
            cochannel_conf_greedy = count_cochannel_conflicts(x_greedy_co, W)
            used_greedy_co = len(set(x_greedy_co))

            start = time.perf_counter()
            x_greedy_adj = greedy_allocation(N, K, W, order=greedy_order, M=M)
            time_greedy_adj = time.perf_counter() - start
            adjacent_cost_greedy = compute_adjacent_cost(x_greedy_adj, W, M)
            adjacent_conf_greedy = count_adjacent_conflicts(x_greedy_adj, W, M)
            used_greedy_adj = len(set(x_greedy_adj))

            # ================================================================
            # 3. DSATUR : déterministe, pas de paramètre aléatoire à partager.
            # ================================================================
            start = time.perf_counter()
            x_dsatur_co = dsatur_allocation(N, K, W, M=None)
            time_dsatur_co = time.perf_counter() - start
            cochannel_cost_dsatur = compute_cochannel_cost(x_dsatur_co, W)
            cochannel_conf_dsatur = count_cochannel_conflicts(x_dsatur_co, W)
            used_dsatur_co = len(set(x_dsatur_co))

            start = time.perf_counter()
            x_dsatur_adj = dsatur_allocation(N, K, W, M=M)
            time_dsatur_adj = time.perf_counter() - start
            adjacent_cost_dsatur = compute_adjacent_cost(x_dsatur_adj, W, M)
            adjacent_conf_dsatur = count_adjacent_conflicts(x_dsatur_adj, W, M)
            used_dsatur_adj = len(set(x_dsatur_adj))

            # ================================================================
            # 4. BD-CeNN : utiliser la MÊME seed pour les deux objectifs
            #    pour que seules les fonctions de coût diffèrent.
            # ================================================================
            # 4.1 Co-canal
            x_bd_co, history_bd_co, t_bd_co, conf_bd_co = bdcenn_allocation(
                N, K, W, M=None,
                num_restarts=config.NUM_RESTARTS,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=run_seed,           # seed de base
                verbose=False
            )
            iterations_used_co = len(history_bd_co) - 1
            cochannel_cost_bd = compute_cochannel_cost(x_bd_co, W)
            cochannel_conf_bd = count_cochannel_conflicts(x_bd_co, W)
            used_bd_co = len(set(x_bd_co))

            # 4.2 Adjacent (MÊME seed que pour co-canal)
            x_bd_adj, history_bd_adj, t_bd_adj, conf_bd_adj = bdcenn_allocation(
                N, K, W, M=M,
                num_restarts=config.NUM_RESTARTS,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=run_seed,           # MÊME seed, pas run_seed+1000
                verbose=False
            )
            iterations_used_adj = len(history_bd_adj) - 1
            adjacent_cost_bd = compute_adjacent_cost(x_bd_adj, W, M)
            adjacent_conf_bd = count_adjacent_conflicts(x_bd_adj, W, M)
            used_bd_adj = len(set(x_bd_adj))

            # --- Stocker les itérations pour BD-CeNN ---
            iterations_data_co.append({
                "run_seed": run_seed,
                "scenario": name,
                "iterations_used": iterations_used_co,
                "max_iter": config.MAX_ITER_BD
            })
            iterations_data_adj.append({
                "run_seed": run_seed,
                "scenario": name,
                "iterations_used": iterations_used_adj,
                "max_iter": config.MAX_ITER_BD
            })

            # --- Sauvegarde des historiques de convergence ---
            # Co-canal
            file_exists_co = os.path.isfile(convergence_co_file)
            with open(convergence_co_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists_co:
                    writer.writerow(['scenario', 'run_seed', 'iteration', 'cochannel_cost', 'seed_init'])
                for it, cost, alloc in history_bd_co:
                    writer.writerow([name, run_seed, it, cost, run_seed])

            # Adjacent
            file_exists_adj = os.path.isfile(convergence_adj_file)
            with open(convergence_adj_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists_adj:
                    writer.writerow(['scenario', 'run_seed', 'iteration', 'adjacent_cost', 'seed_init'])
                for it, cost, alloc in history_bd_adj:
                    writer.writerow([name, run_seed, it, cost, run_seed])

            # ================================================================
            # 5. ENREGISTREMENT DES RÉSULTATS BRUTS
            # ================================================================
            raw_rows.append({
                "run_seed": run_seed,
                "scenario": name,
                "N": N,
                "K": K,
                "seed_scenario": seed_scenario,
                # Métriques co-canal
                "cochannel_cost_rand": cochannel_cost_rand,
                "cochannel_conflicts_rand": cochannel_conf_rand,
                "cochannel_time_rand": time_rand,          # temps unique pour Random
                "cochannel_used_channels_rand": used_rand,
                "cochannel_cost_greedy": cochannel_cost_greedy,
                "cochannel_conflicts_greedy": cochannel_conf_greedy,
                "cochannel_time_greedy": time_greedy_co,
                "cochannel_used_channels_greedy": used_greedy_co,
                "cochannel_cost_dsatur": cochannel_cost_dsatur,
                "cochannel_conflicts_dsatur": cochannel_conf_dsatur,
                "cochannel_time_dsatur": time_dsatur_co,
                "cochannel_used_channels_dsatur": used_dsatur_co,
                "cochannel_cost_bd": cochannel_cost_bd,
                "cochannel_conflicts_bd": cochannel_conf_bd,
                "cochannel_time_bd": t_bd_co,
                "cochannel_used_channels_bd": used_bd_co,
                # Métriques adjacentes
                "adjacent_cost_rand": adjacent_cost_rand,
                "adjacent_conflicts_rand": adjacent_conf_rand,
                "adjacent_time_rand": time_rand,          # même temps
                "adjacent_used_channels_rand": used_rand,
                "adjacent_cost_greedy": adjacent_cost_greedy,
                "adjacent_conflicts_greedy": adjacent_conf_greedy,
                "adjacent_time_greedy": time_greedy_adj,
                "adjacent_used_channels_greedy": used_greedy_adj,
                "adjacent_cost_dsatur": adjacent_cost_dsatur,
                "adjacent_conflicts_dsatur": adjacent_conf_dsatur,
                "adjacent_time_dsatur": time_dsatur_adj,
                "adjacent_used_channels_dsatur": used_dsatur_adj,
                "adjacent_cost_bd": adjacent_cost_bd,
                "adjacent_conflicts_bd": adjacent_conf_bd,
                "adjacent_time_bd": t_bd_adj,
                "adjacent_used_channels_bd": used_bd_adj
            })

    print("\n✅ Toutes les exécutions sont terminées.")
    print("📊 Construction des DataFrames...")

    df_raw = pd.DataFrame(raw_rows)

    # --- Melt des données pour faciliter les statistiques ---
    # Pour co-canal
    co_melted = []
    for method in ['rand', 'greedy', 'dsatur', 'bd']:
        for metric in ['cost', 'conflicts', 'time', 'used_channels']:
            col_name = f"cochannel_{metric}_{method}"
            if col_name in df_raw.columns:
                temp = df_raw[['scenario', 'N', 'K', 'seed_scenario', col_name]].copy()
                temp['method'] = {'rand': 'Random', 'greedy': 'Greedy', 'dsatur': 'DSATUR', 'bd': 'BD-CeNN'}[method]
                temp['metric'] = f"cochannel_{metric}"
                temp['value'] = temp[col_name]
                temp = temp.drop(columns=[col_name])
                co_melted.append(temp)

    df_co_melted = pd.concat(co_melted, ignore_index=True) if co_melted else pd.DataFrame()

    # Pour adjacent
    adj_melted = []
    for method in ['rand', 'greedy', 'dsatur', 'bd']:
        for metric in ['cost', 'conflicts', 'time', 'used_channels']:
            col_name = f"adjacent_{metric}_{method}"
            if col_name in df_raw.columns:
                temp = df_raw[['scenario', 'N', 'K', 'seed_scenario', col_name]].copy()
                temp['method'] = {'rand': 'Random', 'greedy': 'Greedy', 'dsatur': 'DSATUR', 'bd': 'BD-CeNN'}[method]
                temp['metric'] = f"adjacent_{metric}"
                temp['value'] = temp[col_name]
                temp = temp.drop(columns=[col_name])
                adj_melted.append(temp)

    df_adj_melted = pd.concat(adj_melted, ignore_index=True) if adj_melted else pd.DataFrame()

    # Combiner les deux
    df_melted = pd.concat([df_co_melted, df_adj_melted], ignore_index=True)

    # --- Calcul des statistiques par (scenario, method, metric) ---
    stats_list = []
    for (scenario, method, metric), group in df_melted.groupby(['scenario', 'method', 'metric']):
        N = group['N'].iloc[0]
        K = group['K'].iloc[0]
        seed_scenario = group['seed_scenario'].iloc[0]
        values = group['value'].values
        stats_list.append({
            'scenario': scenario,
            'N': N,
            'K': K,
            'seed_scenario': seed_scenario,
            'method': method,
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0,
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        })

    df_stats = pd.DataFrame(stats_list)

    # --- Pivot pour avoir une ligne par (scenario, method) avec toutes les métriques ---
    df_summary = df_stats.pivot_table(
        index=['scenario', 'N', 'K', 'seed_scenario', 'method'],
        columns='metric',
        values=['mean', 'std', 'median', 'min', 'max']
    ).reset_index()
    # Aplatir les colonnes
    df_summary.columns = ['scenario', 'N', 'K', 'seed_scenario', 'method'] + \
                         [f"{metric}_{stat}" for stat, metric in df_summary.columns[5:]]

    # --- Calcul des intervalles de confiance ---
    ci_data = []
    for (scenario, method, metric), group in df_melted.groupby(['scenario', 'method', 'metric']):
        N = group['N'].iloc[0]
        K = group['K'].iloc[0]
        seed_scenario = group['seed_scenario'].iloc[0]
        values = group['value'].values
        ci_low, ci_high = compute_confidence_interval(values)
        ci_data.append({
            'scenario': scenario,
            'N': N,
            'K': K,
            'seed_scenario': seed_scenario,
            'method': method,
            'metric': metric,
            'ci_low': ci_low,
            'ci_high': ci_high
        })
    df_ci = pd.DataFrame(ci_data)

    # --- Pivot des IC pour avoir une ligne par (scenario, method) ---
    df_ci_pivot = df_ci.pivot_table(
        index=['scenario', 'N', 'K', 'seed_scenario', 'method'],
        columns='metric',
        values=['ci_low', 'ci_high']
    ).reset_index()
    df_ci_pivot.columns = ['scenario', 'N', 'K', 'seed_scenario', 'method'] + \
                          [f"{metric}_{stat}" for stat, metric in df_ci_pivot.columns[5:]]

    # --- Exporter les CSV de statistiques ---
    # 1. Tableau des médianes
    df_median = df_summary[['scenario', 'N', 'K', 'seed_scenario', 'method'] + 
                           [col for col in df_summary.columns if '_median' in col]]
    df_median.to_csv(config.STATS_MEDIAN_CSV, index=False)

    # 2. Tableau des minima
    df_min = df_summary[['scenario', 'N', 'K', 'seed_scenario', 'method'] + 
                        [col for col in df_summary.columns if '_min' in col]]
    df_min.to_csv(config.STATS_MIN_CSV, index=False)

    # 3. Tableau des maxima
    df_max = df_summary[['scenario', 'N', 'K', 'seed_scenario', 'method'] + 
                        [col for col in df_summary.columns if '_max' in col]]
    df_max.to_csv(config.STATS_MAX_CSV, index=False)

    # 4. Tableau des intervalles de confiance
    df_ci_pivot.to_csv(config.STATS_CI_CSV, index=False)

    # 5. Tableau du nombre de balayages pour co-canal
    df_iter_co = pd.DataFrame(iterations_data_co)
    if not df_iter_co.empty:
        iter_stats_co = df_iter_co.groupby("scenario").agg({
            "iterations_used": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        iter_stats_co.columns = ['scenario', 'iter_mean_co', 'iter_std_co', 'iter_min_co', 'iter_max_co', 'iter_median_co']
        iter_stats_co.to_csv(config.STATS_ITERATIONS_CO_CSV, index=False)
    else:
        pd.DataFrame(columns=['scenario', 'iter_mean_co', 'iter_std_co', 'iter_min_co', 'iter_max_co', 'iter_median_co']).to_csv(config.STATS_ITERATIONS_CO_CSV, index=False)

    # 6. Tableau du nombre de balayages pour adjacent
    df_iter_adj = pd.DataFrame(iterations_data_adj)
    if not df_iter_adj.empty:
        iter_stats_adj = df_iter_adj.groupby("scenario").agg({
            "iterations_used": ["mean", "std", "min", "max", "median"]
        }).reset_index()
        iter_stats_adj.columns = ['scenario', 'iter_mean_adj', 'iter_std_adj', 'iter_min_adj', 'iter_max_adj', 'iter_median_adj']
        iter_stats_adj.to_csv(config.STATS_ITERATIONS_ADJ_CSV, index=False)
    else:
        pd.DataFrame(columns=['scenario', 'iter_mean_adj', 'iter_std_adj', 'iter_min_adj', 'iter_max_adj', 'iter_median_adj']).to_csv(config.STATS_ITERATIONS_ADJ_CSV, index=False)

    print("✅ Statistiques supplémentaires exportées en CSV.")

    # --- Sauvegarde du fichier Excel principal ---
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
        df_summary.to_excel(writer, sheet_name='Summary_Stats', index=False)
        df_ci_pivot.to_excel(writer, sheet_name='Confidence_Intervals', index=False)
        if not df_iter_co.empty:
            iter_stats_co.to_excel(writer, sheet_name='BD_Iterations_Co', index=False)
        if not df_iter_adj.empty:
            iter_stats_adj.to_excel(writer, sheet_name='BD_Iterations_Adj', index=False)

    print(f"\n✅ Validation terminée !")
    print(f"📁 Fichier Excel : {excel_file}")
    print(f"📁 Statistiques CSV : {config.STATS_MEDIAN_CSV}, {config.STATS_MIN_CSV}, {config.STATS_MAX_CSV}, {config.STATS_CI_CSV}")
    print(f"📁 Historique convergence (co-canal) : {convergence_co_file}")
    print(f"📁 Historique convergence (adjacent) : {convergence_adj_file}")

if __name__ == "__main__":
    print(f"📂 Scénarios définis : {len(config.SCENARIOS)} scénarios.")
    run_validation()