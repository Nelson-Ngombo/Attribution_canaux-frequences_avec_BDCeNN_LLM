# plots.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import validation
import config

def generate_full_experiment_plots():
    print("="*80)
    print("📊 GÉNÉRATION DU RAPPORT FINAL")
    print("="*80)

    excel_file = config.VALIDATION_EXCEL_FILE
    convergence_file = config.CONVERGENCE_HISTORY_FILE
    figures_dir = config.FIGURES_DIR
    summary_csv = config.SUMMARY_CSV_FILE

    if not os.path.exists(excel_file):
        print("🔁 Fichier validation_results.xlsx introuvable. Lancement de la validation (30 runs)...")
        validation.run_validation()
    else:
        print("✅ Fichier validation_results.xlsx trouvé.")

    df_summary = pd.read_excel(excel_file, sheet_name="Summary")
    scenarios = df_summary['scenario'].unique()
    methods = ['Random', 'Greedy', 'DSATUR', 'BD-CeNN']
    colors = ['royalblue', 'forestgreen', 'darkorange', 'firebrick']

    # ---------- 1. Graphiques comparatifs (inchangés) ----------
    metrics_list = [
        ('global_cost', 'Coût global'),
        ('spectrum_conflicts', 'Conflits spectraux'),
        ('used_channels', 'Canaux utilisés'),
        ('time', "Temps d'exécution (s)")
    ]

    for metric, label in metrics_list:
        if metric == 'time':
            mean_cols = [f"{m}_time_mean" for m in methods]
            std_cols = [f"{m}_time_std" for m in methods]
            factor = 1
        else:
            mean_cols = [f"{m}_{metric}_mean" for m in methods]
            std_cols = [f"{m}_{metric}_std" for m in methods]
            factor = 1

        if not all(col in df_summary.columns for col in mean_cols + std_cols):
            print(f"⚠️ Colonnes pour {metric} manquantes. Ignoré.")
            continue

        means = pd.DataFrame(index=scenarios)
        stds = pd.DataFrame(index=scenarios)
        for m in methods:
            means[m] = df_summary.set_index('scenario')[f"{m}_{metric}_mean"] * factor
            stds[m] = df_summary.set_index('scenario')[f"{m}_{metric}_std"] * factor

        x = np.arange(len(scenarios))
        width = 0.2
        fig, ax = plt.subplots(figsize=(18, 8))
        for i, m in enumerate(methods):
            offset = (i - 1.5) * width
            ax.bar(x + offset, means[m], width, yerr=stds[m],
                   capsize=3, label=m, color=colors[i], alpha=0.8)

        ax.set_xlabel('Scénario')
        ax.set_ylabel(label)
        ax.set_title(f'Comparaison des méthodes - {label}')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=90, fontsize=8)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        filename = f"comparison_{metric}.png" if metric != 'time' else "comparison_time.png"
        plt.savefig(figures_dir / filename, dpi=300)
        plt.show()

    # ---------- 2. Heatmap de classement (inchangée) ----------
    metric = 'global_cost'
    mean_cols = [f"{m}_{metric}_mean" for m in methods]
    if all(col in df_summary.columns for col in mean_cols):
        ranks = df_summary[mean_cols].rank(axis=1, method='dense', ascending=True).astype(int)
        ranks.columns = methods
        ranks.index = df_summary['scenario']

        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(ranks.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(methods))
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(ranks.index)))
        ax.set_xticklabels(methods, fontweight='bold')
        ax.set_yticklabels(ranks.index)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(ranks.index)):
            for j in range(len(methods)):
                ax.text(j, i, ranks.iloc[i, j],
                        ha="center", va="center",
                        color="black" if ranks.iloc[i, j] <= 2 else "white",
                        fontweight='bold', fontsize=9)
        ax.set_xlabel("Méthode", fontsize=12)
        ax.set_ylabel("Scénario", fontsize=12)
        ax.set_title("Classement des méthodes par scénario (1 = meilleur coût global)", fontsize=14)
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6, ticks=np.arange(1, len(methods)+1))
        cbar.set_label('Rang', fontsize=10)
        plt.tight_layout()
        plt.savefig(figures_dir / "comparison_ranking.png", dpi=300)
        plt.show()

    # ---------- 3. Courbe de convergence MOYENNE avec zone d'écart-type (UNIQUEMENT) ----------
    if os.path.exists(convergence_file):
        df_hist = pd.read_csv(convergence_file)
        scenarios_hist = df_hist['scenario'].unique()
        scenario_info = {}
        for idx, row in df_summary.iterrows():
            scenario_info[row['scenario']] = {
                'N': row['N'],
                'K': row['K'],
                'seed_scenario': row['seed_scenario']
            }

        for scenario in scenarios_hist:
            df_scenario = df_hist[df_hist['scenario'] == scenario]
            
            # Grouper par itération et calculer moyenne et écart-type
            grouped = df_scenario.groupby('iteration').agg({
                'global_cost': ['mean', 'std']
            }).reset_index()
            grouped.columns = ['iteration', 'mean', 'std']
            
            # Récupérer les infos du scénario
            info = scenario_info.get(scenario, {})
            N = info.get('N', '?')
            K = info.get('K', '?')
            seed_scenario = info.get('seed_scenario', '?')

            # Tracer la courbe avec zone d'écart-type
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Ligne de moyenne
            ax.plot(grouped['iteration'], grouped['mean'], 
                    marker='o', linestyle='-', markersize=3, color='red', linewidth=1.5,
                    label='Moyenne du coût global')
            
            # Zone d'écart-type (shaded area)
            ax.fill_between(grouped['iteration'], 
                            grouped['mean'] - grouped['std'], 
                            grouped['mean'] + grouped['std'],
                            alpha=0.25, color='red', label='Écart-type (±1σ)')
            
            ax.set_xlabel("Itérations")
            ax.set_ylabel("Coût global")
            ax.set_title(f"Convergence BD-CeNN - {scenario} (N={N}, K={K}, seed_scénario={seed_scenario})\nMoyenne sur {config.NUM_RUNS} runs")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"convergence_mean_{scenario}.png", dpi=300)
            plt.close(fig)
            
        print("✅ Courbes de convergence moyennes avec zone d'écart-type sauvegardées (convergence_mean_*.png) sans affichage.")
    else:
        print("⚠️ Fichier convergence_history.csv introuvable. Pas de courbes de convergence.")

    # ---------- 4. Exporter le tableau récapitulatif ----------
    df_summary.to_csv(summary_csv, index=False)
    print(f"✅ Tableau récapitulatif : {summary_csv}")

    # ---------- 5. Log d'exécution ----------
    import datetime
    log_file = config.LOGS_DIR / f"report_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("RAPPORT FINAL GÉNÉRÉ\n")
        f.write(f"Date : {datetime.datetime.now()}\n")
        f.write(f"Fichier Excel source : {excel_file}\n")
        f.write(f"Figures : {figures_dir}\n")
        f.write(f"CSV : {summary_csv}\n")
    print(f"✅ Log sauvegardé : {log_file}")

    print("\n🏁 Rapport final généré avec succès.")

if __name__ == "__main__":
    generate_full_experiment_plots()