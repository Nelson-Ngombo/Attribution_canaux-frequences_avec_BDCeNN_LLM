# plots.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import validation
import config


def generate_full_experiment_plots():
    print("=" * 80)
    print("📊 GÉNÉRATION DU RAPPORT FINAL")
    print("=" * 80)

    excel_file = config.VALIDATION_EXCEL_FILE
    convergence_adj_file = config.CONVERGENCE_HISTORY_FILE
    convergence_co_file = config.CONVERGENCE_HISTORY_COCHANNEL_FILE
    figures_dir = config.FIGURES_DIR
    summary_csv = config.SUMMARY_CSV_FILE

    if not os.path.exists(excel_file):
        print("🔁 Fichier validation_results.xlsx introuvable. Lancement de la validation (30 runs)...")
        validation.run_validation()
    else:
        print("✅ Fichier validation_results.xlsx trouvé.")

    # --- Lecture de la feuille "Summary_Stats" ---
    try:
        df_stats = pd.read_excel(excel_file, sheet_name="Summary_Stats")
    except ValueError:
        print("❌ Feuille 'Summary_Stats' non trouvée. Vérifie le fichier Excel.")
        return

    scenarios = df_stats['scenario'].unique()
    methods = ['Random', 'Greedy', 'DSATUR', 'BD-CeNN']
    colors = ['royalblue', 'forestgreen', 'darkorange', 'firebrick']

    def pivot_metric(metric_name, stat='mean'):
        col_name = f"{metric_name}_{stat}"
        if col_name not in df_stats.columns:
            return None
        pivot = df_stats.pivot(index='scenario', columns='method', values=col_name)
        pivot = pivot[methods]
        return pivot

    # ================================================================
    # 1. Graphiques de comparaison (moyennes avec barres d'erreur)
    # ================================================================
    def plot_bar_comparison(metric, label, filename):
        pivot_mean = pivot_metric(metric, 'mean')
        pivot_std = pivot_metric(metric, 'std')
        if pivot_mean is None or pivot_std is None:
            return

        pivot_mean = pivot_mean.reindex(scenarios)
        pivot_std = pivot_std.reindex(scenarios)

        # Arrondir pour used_channels
        if metric == 'used_channels':
            pivot_mean = pivot_mean.round(0)
            pivot_std = pivot_std.round(1)

        x = np.arange(len(scenarios))
        width = 0.2
        fig, ax = plt.subplots(figsize=(18, 8))
        for i, m in enumerate(methods):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, pivot_mean[m], width, yerr=pivot_std[m],
                          capsize=3, label=m, color=colors[i], alpha=0.8)
            if metric == 'used_channels':
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Scénario')
        ax.set_ylabel(label)
        title = f'Comparaison des méthodes - {label}'
        if metric == 'used_channels':
            title += ' (moyenne arrondie)'
        else:
            title += ' (moyenne ± écart-type)'
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=90, fontsize=8)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=300)
        plt.close(fig)
        print(f"✅ Figure {filename} sauvegardée.")

    plot_bar_comparison('cochannel_cost', 'Coût co-canal', 'comparison_cochannel_cost.png')
    plot_bar_comparison('cochannel_conflicts', 'Conflits co-canal', 'comparison_cochannel_conflicts.png')
    plot_bar_comparison('adjacent_cost', 'Coût avec interférences adjacentes', 'comparison_adjacent_cost.png')
    plot_bar_comparison('adjacent_conflicts', 'Conflits avec interférences adjacentes', 'comparison_adjacent_conflicts.png')
    plot_bar_comparison('used_channels', 'Canaux utilisés', 'comparison_used_channels.png')
    plot_bar_comparison('time', "Temps d'exécution (s)", 'comparison_time.png')

    # ================================================================
    # 2. Heatmaps de classement (6 figures)
    # ================================================================
    ranking_configs = [
        ('adjacent_cost', 'coût avec interférences adjacentes', 'comparison_ranking_adjacent_cost.png'),
        ('cochannel_cost', 'coût co-canal', 'comparison_ranking_cochannel_cost.png'),
        ('adjacent_conflicts', 'conflits avec interférences adjacentes', 'comparison_ranking_adjacent_conflicts.png'),
        ('cochannel_conflicts', 'conflits co-canal', 'comparison_ranking_cochannel_conflicts.png'),
        ('time', 'temps d\'exécution', 'comparison_ranking_time.png'),
        ('used_channels', 'nombre de canaux utilisés', 'comparison_ranking_used_channels.png')
    ]

    for metric, label, filename in ranking_configs:
        pivot_mean = pivot_metric(metric, 'mean')
        if pivot_mean is not None:
            if metric == 'used_channels':
                pivot_mean = pivot_mean.round(0)
            ranks = pivot_mean.rank(axis=1, method='dense', ascending=True).astype(int)
            
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
            ax.set_title(f"Classement des méthodes par scénario (1 = meilleur {label})", fontsize=14)
            cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6, ticks=np.arange(1, len(methods)+1))
            cbar.set_label('Rang', fontsize=10)
            plt.tight_layout()
            plt.savefig(figures_dir / filename, dpi=300)
            plt.close(fig)
            print(f"✅ Figure {filename} sauvegardée.")

    # ================================================================
    # 3. Courbes de convergence (adjacent ET co-canal)
    # ================================================================
    def plot_convergence(csv_file, cost_label, filename_prefix):
        if not os.path.exists(csv_file):
            print(f"⚠️ Fichier {csv_file} introuvable. Pas de courbes {cost_label}.")
            return

        df_hist = pd.read_csv(csv_file)
        scenarios_hist = df_hist['scenario'].unique()
        scenario_info = df_stats.groupby('scenario').first()[['N', 'K']].to_dict(orient='index')

        for scenario in scenarios_hist:
            df_scenario = df_hist[df_hist['scenario'] == scenario]
            grouped = df_scenario.groupby('iteration').agg({
                cost_label: ['mean', 'std']
            }).reset_index()
            grouped.columns = ['iteration', 'mean', 'std']

            info = scenario_info.get(scenario, {})
            N = info.get('N', '?')
            K = info.get('K', '?')

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(grouped['iteration'], grouped['mean'],
                    marker='o', linestyle='-', markersize=3, color='red', linewidth=1.5,
                    label=f'Moyenne du {cost_label}')
            ax.fill_between(grouped['iteration'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.25, color='red', label='Écart-type (±1σ)')
            ax.set_xlabel("Itérations")
            ylabel = "Coût co-canal" if "cochannel" in cost_label else "Coût avec interférences adjacentes"
            ax.set_ylabel(f"Coût global J(x) ({ylabel})")
            ax.set_title(f"Convergence BD-CeNN - {scenario} (N={N}, K={K})\n"
                         f"Moyenne sur {config.NUM_RUNS} runs")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"{filename_prefix}_{scenario}.png", dpi=300)
            plt.close(fig)
        print(f"✅ Courbes de convergence {cost_label} sauvegardées.")

    plot_convergence(convergence_adj_file, 'adjacent_cost', 'convergence_mean_adjacent')
    plot_convergence(convergence_co_file, 'cochannel_cost', 'convergence_mean_cochannel')

    # ================================================================
    # 4. FIGURES DES STATISTIQUES MANQUANTES (Médianes, Min, Max, IC, Itérations)
    # ================================================================
    try:
        df_median = pd.read_csv(config.STATS_MEDIAN_CSV)
        df_min = pd.read_csv(config.STATS_MIN_CSV)
        df_max = pd.read_csv(config.STATS_MAX_CSV)
        df_ci = pd.read_csv(config.STATS_CI_CSV)
        df_iter = pd.read_csv(config.STATS_ITERATIONS_CSV)
    except FileNotFoundError as e:
        print(f"⚠️ Fichier CSV manquant : {e}. Figures supplémentaires ignorées.")
    else:
        def plot_statistic(df_stat, stat_name, metric, label, filename_prefix):
            pivot = df_stat.pivot(index='scenario', columns='method', values=f"{metric}_{stat_name}")
            pivot = pivot.reindex(scenarios)
            if pivot.empty:
                return

            fig, ax = plt.subplots(figsize=(16, 8))
            x = np.arange(len(scenarios))
            width = 0.2
            for i, m in enumerate(methods):
                if m not in pivot.columns:
                    continue
                offset = (i - 1.5) * width
                ax.bar(x + offset, pivot[m], width, label=m, color=colors[i % len(colors)], alpha=0.8)

            ax.set_xlabel('Scénario')
            ax.set_ylabel(label)
            ax.set_title(f'{label} - {stat_name.capitalize()} par méthode')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=90, fontsize=8)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(figures_dir / f"{filename_prefix}_{metric}_{stat_name}.png", dpi=300)
            plt.close(fig)
            print(f"✅ Figure {filename_prefix}_{metric}_{stat_name}.png sauvegardée.")

        # Métriques pour les statistiques (médianes, min, max)
        stat_metrics = [
            ('adjacent_cost', 'Coût adjacent'),
            ('cochannel_cost', 'Coût co-canal'),
            ('adjacent_conflicts', 'Conflits adjacents'),
            ('cochannel_conflicts', 'Conflits co-canal'),
            ('time', "Temps d'exécution (s)"),
            ('used_channels', 'Canaux utilisés')
        ]

        for metric, label in stat_metrics:
            plot_statistic(df_median, 'median', metric, label, 'median')
            plot_statistic(df_min, 'min', metric, label, 'min')
            plot_statistic(df_max, 'max', metric, label, 'max')

        # --- 4.1 Intervalles de confiance (style professionnel) ---
        ci_metrics = [
            ('adjacent_cost', 'Coût adjacent'),
            ('cochannel_cost', 'Coût co-canal'),
            ('adjacent_conflicts', 'Conflits adjacents'),
            ('cochannel_conflicts', 'Conflits co-canal'),
            ('time', "Temps d'exécution (s)")
        ]

        for metric, label in ci_metrics:
            df_ci_metric = df_ci[['scenario', 'method', f"{metric}_ci_low", f"{metric}_ci_high"]]
            
            fig, ax = plt.subplots(figsize=(16, 8))
            x = np.arange(len(scenarios))
            width = 0.2
            
            for i, m in enumerate(methods):
                subset = df_ci_metric[df_ci_metric['method'] == m]
                lows = subset.set_index('scenario')[f"{metric}_ci_low"].reindex(scenarios).values
                highs = subset.set_index('scenario')[f"{metric}_ci_high"].reindex(scenarios).values
                
                means = (lows + highs) / 2
                errs = (highs - lows) / 2
                offset = (i - 1.5) * width
                
                ax.errorbar(x + offset, means, yerr=errs, fmt='o', capsize=5,
                           color=colors[i], label=m, markersize=6, elinewidth=2, capthick=2)
                ax.plot(x + offset, means, linestyle='-', color=colors[i], alpha=0.3, linewidth=1)

            ax.set_xlabel('Scénario')
            ax.set_ylabel(label)
            ax.set_title(f"Intervalle de confiance à 95% - {label}")
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=90, fontsize=8)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(figures_dir / f"confidence_interval_{metric}.png", dpi=300)
            plt.close(fig)
            print(f"✅ Figure confidence_interval_{metric}.png sauvegardée.")

        # --- 4.2 Figure du nombre de balayages (itérations) ---
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_iter['scenario']))
        ax.bar(x, df_iter['iter_mean'], yerr=df_iter['iter_std'], capsize=5,
               color='teal', alpha=0.7, label='Moyenne ± écart-type')
        ax.set_xticks(x)
        ax.set_xticklabels(df_iter['scenario'])
        ax.set_xlabel('Scénario')
        ax.set_ylabel("Nombre de balayages effectués")
        ax.set_title("BD-CeNN - Nombre d'itérations avant arrêt (moyenne ± σ)")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for i, row in df_iter.iterrows():
            ax.text(i, row['iter_mean'] + 0.5, f"{row['iter_mean']:.1f}", ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(figures_dir / "bd_iterations_histogram.png", dpi=300)
        plt.close(fig)
        print("✅ Figure bd_iterations_histogram.png sauvegardée.")

    # --- 5. Exporter le tableau récapitulatif au format CSV ---
    df_stats.to_csv(summary_csv, index=False)
    print(f"✅ Tableau récapitulatif : {summary_csv}")

    # --- 6. Log d'exécution ---
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