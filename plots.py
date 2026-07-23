# plots.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import validation
import config  # <-- IMPORT

def generate_full_experiment_plots():
    """
    Génère le rapport final à partir des résultats de validation (30 runs).
    - Si validation_results.xlsx n'existe pas, lance la validation.
    - Produit les histogrammes comparatifs avec barres d'erreur :
        * Coût global
        * Conflits spectraux
        * Canaux utilisés
        * Temps d'exécution (en secondes)
    - Produit la heatmap de classement.
    - Sauvegarde les courbes de convergence (sans les afficher).
    - Exporte le tableau récapitulatif en CSV.
    """
    print("="*80)
    print("📊 GÉNÉRATION DU RAPPORT FINAL")
    print("="*80)

    # --- Utiliser les chemins de config ---
    excel_file = config.VALIDATION_EXCEL_FILE
    convergence_file = config.CONVERGENCE_HISTORY_FILE
    figures_dir = config.FIGURES_DIR
    summary_csv = config.SUMMARY_CSV_FILE

    # --- 1. Lancer la validation si le fichier Excel n'existe pas ---
    if not os.path.exists(excel_file):
        print("🔁 Fichier validation_results.xlsx introuvable. Lancement de la validation (30 runs)...")
        validation.run_validation()
    else:
        print("✅ Fichier validation_results.xlsx trouvé.")

    # --- 2. Lire le résumé ---
    df_summary = pd.read_excel(excel_file, sheet_name="Summary")
    scenarios = df_summary['scenario'].unique()
    methods = ['Random', 'Greedy', 'DSATUR', 'BD-CeNN']
    colors = ['royalblue', 'forestgreen', 'darkorange', 'firebrick']

    # --- 3. Graphiques comparatifs (avec barres d'erreur) ---
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

    # --- 4. Heatmap de classement (basée sur global_cost) ---
    metric = 'global_cost'
    mean_cols = [f"{m}_{metric}_mean" for m in methods]
    if all(col in df_summary.columns for col in mean_cols):
        ranks = df_summary[mean_cols].rank(axis=1, method='dense', ascending=True).astype(int)
        ranks.columns = methods
        ranks.index = df_summary['scenario']

        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(ranks.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=4)
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
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6, ticks=[1, 2, 3, 4])
        cbar.set_label('Rang (1 = meilleur, 4 = moins bon)', fontsize=10)
        plt.tight_layout()
        plt.savefig(figures_dir / "comparison_ranking.png", dpi=300)
        plt.show()

    # --- 5. Courbes de convergence (SAUVEGARDE UNIQUEMENT) ---
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
            df_scenario = df_hist[df_hist['scenario'] == scenario].sort_values('iteration')
            info = scenario_info.get(scenario, {})
            N = info.get('N', '?')
            K = info.get('K', '?')
            seed_scenario = info.get('seed_scenario', '?')

            fig = plt.figure(figsize=(10, 6))
            plt.plot(df_scenario['iteration'], df_scenario['global_cost'],
                     marker='o', linestyle='-', markersize=4, color='red')
            plt.xlabel("Itérations")
            plt.ylabel("Coût global")
            plt.title(f"Évolution du coût global - {scenario} (N={N}, K={K}, seed_scénario={seed_scenario}, seed_init=1)")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(figures_dir / f"convergence_{scenario}.png", dpi=300)
            plt.close(fig)

        print("✅ Courbes de convergence sauvegardées (convergence_*.png) sans affichage.")
    else:
        print("⚠️ Fichier convergence_history.csv introuvable. Pas de courbes de convergence.")

    # --- 6. Exporter le tableau récapitulatif en CSV ---
    df_summary.to_csv(summary_csv, index=False)
    print(f"✅ Tableau récapitulatif : {summary_csv}")

    # --- 7. Log d'exécution ---
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