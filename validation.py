# validation.py
import numpy as np
import pandas as pd
import time
import os
import csv
from datetime import datetime
from data_generator import all_data
from baselines import greedy_allocation, dsatur_allocation, create_channel_interference_matrix
from bdcenn_solver import bdcenn_allocation
from bdcenn_spectrum import compute_spectrum_energy
from metrics import compute_metrics, count_spectrum_conflicts
import config  # <-- IMPORT

def run_validation():
    print("="*80)
    print("🚀 LANCEMENT DE LA VALIDATION APPROFONDIE")
    print(f"   - {len(all_data)} scénarios")
    print(f"   - {config.NUM_RUNS} répétitions par scénario")
    print("   - Méthodes : Random, Greedy, DSATUR, BD-CeNN (discret)")
    print("   - Métriques : Conflits (spectraux), Canaux utilisés, Coût global (avec M)")
    print("="*80)

    raw_rows = []
    scenario_list = list(all_data.items())
    total_runs = len(scenario_list) * config.NUM_RUNS
    current_run = 0

    # Utiliser les chemins de config
    convergence_file = config.CONVERGENCE_HISTORY_FILE
    excel_file = config.VALIDATION_EXCEL_FILE

    if os.path.exists(convergence_file):
        os.remove(convergence_file)

    for run_seed in range(1, config.NUM_RUNS + 1):
        print(f"\n--- Run {run_seed}/{config.NUM_RUNS} ---")
        for name, data in scenario_list:
            current_run += 1
            if current_run % 50 == 0:
                print(f"  Progression : {current_run}/{total_runs} runs effectués")

            N = data["N"]
            K = data["K"]
            W = np.array(data["W"])
            seed_scenario = data["seed"]
            M = create_channel_interference_matrix(K)

            # ---------- 1. Random ----------
            np.random.seed(run_seed)
            start = time.perf_counter()
            x_rand = np.random.randint(0, K, size=N)
            time_rand = time.perf_counter() - start
            m_rand = compute_metrics(x_rand, W)
            global_cost_rand = compute_spectrum_energy(x_rand, W, M)
            spectrum_conflicts_rand = count_spectrum_conflicts(x_rand, W, M)

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
                "global_cost": global_cost_rand
            })

            # ---------- 2. Greedy ----------
            start = time.perf_counter()
            x_greedy = greedy_allocation(N, K, W, M=M)
            time_greedy = time.perf_counter() - start
            m_greedy = compute_metrics(x_greedy, W)
            global_cost_greedy = compute_spectrum_energy(x_greedy, W, M)
            spectrum_conflicts_greedy = count_spectrum_conflicts(x_greedy, W, M)

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
                "global_cost": global_cost_greedy
            })

            # ---------- 3. DSATUR ----------
            start = time.perf_counter()
            x_dsatur = dsatur_allocation(N, K, W, M=M)
            time_dsatur = time.perf_counter() - start
            m_dsatur = compute_metrics(x_dsatur, W)
            global_cost_dsatur = compute_spectrum_energy(x_dsatur, W, M)
            spectrum_conflicts_dsatur = count_spectrum_conflicts(x_dsatur, W, M)

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
                "global_cost": global_cost_dsatur
            })

            # ---------- 4. BD-CeNN ----------
            x_bd, history, t_bd, conf_bd = bdcenn_allocation(
                N, K, W, M=M,
                max_iter=config.MAX_ITER_BD,
                random_order=True,
                seed=run_seed,
                verbose=False,
                use_sa=True,
                T_init=10.0,
                T_min=0.01,
                cooling_rate=0.99
            )
            m_bd = compute_metrics(x_bd, W)
            global_cost_bd = compute_spectrum_energy(x_bd, W, M)
            spectrum_conflicts_bd = count_spectrum_conflicts(x_bd, W, M)

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
                "global_cost": global_cost_bd
            })

            # ---------- Sauvegarde de l'historique pour la première run ----------
            if run_seed == 1:
                file_exists = os.path.isfile(convergence_file)
                with open(convergence_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['scenario', 'iteration', 'global_cost', 'seed_init'])
                    for it, cost, alloc in history:
                        writer.writerow([name, it, cost, 1])

    print("\n✅ Toutes les exécutions sont terminées.")
    print("📊 Construction du DataFrame brut...")

    df_raw = pd.DataFrame(raw_rows)

    grouped = df_raw.groupby(["scenario", "N", "K", "seed_scenario", "method"])
    df_agg = grouped.agg({
        "spectrum_conflicts": ["mean", "std"],
        "used_channels": ["mean", "std"],
        "time": ["mean", "std"],
        "global_cost": ["mean", "std"]
    }).reset_index()

    df_agg.columns = [
        'scenario', 'N', 'K', 'seed_scenario', 'method',
        'spectrum_conflicts_mean', 'spectrum_conflicts_std',
        'used_channels_mean', 'used_channels_std',
        'time_mean', 'time_std',
        'global_cost_mean', 'global_cost_std'
    ]

    # Arrondi des canaux
    df_agg['used_channels_mean'] = df_agg['used_channels_mean'].round(0)
    df_agg['used_channels_std'] = df_agg['used_channels_std'].round(2)

    pivot = df_agg.pivot_table(
        index=['scenario', 'N', 'K', 'seed_scenario'],
        columns='method',
        values=[
            'spectrum_conflicts_mean', 'spectrum_conflicts_std',
            'used_channels_mean', 'used_channels_std',
            'time_mean', 'time_std',
            'global_cost_mean', 'global_cost_std'
        ]
    )

    pivot.columns = [f"{method}_{metric}" for metric, method in pivot.columns]
    df_summary = pivot.reset_index()

    ordered_cols = ['scenario', 'N', 'K', 'seed_scenario']
    methods = ['Random', 'Greedy', 'DSATUR', 'BD-CeNN']
    metrics = [
        'spectrum_conflicts_mean', 'spectrum_conflicts_std',
        'used_channels_mean', 'used_channels_std',
        'time_mean', 'time_std',
        'global_cost_mean', 'global_cost_std'
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
    print(f"📁 Historique convergence : {convergence_file}")

if __name__ == "__main__":
    print(f"📂 Chargement des scénarios : {len(all_data)} scénarios trouvés.")
    if len(all_data) < 50:
        print("⚠️  Attention : moins de 50 scénarios détectés.")
    run_validation()