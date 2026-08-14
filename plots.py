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
        df_summary = pd.read_excel(excel_file, sheet_name="Summary_Stats")
    except ValueError:
        print("❌ Feuille 'Summary_Stats' non trouvée. Vérifie le fichier Excel.")
        return

    if df_summary.empty:
        print("⚠️ La feuille Summary_Stats est vide.")
        return

    scenarios = df_summary['scenario'].unique()
    methods = ['Random', 'Greedy', 'DSATUR', 'BD-CeNN']
    colors = ['royalblue', 'forestgreen', 'darkorange', 'firebrick']

    # --- Fonction pour pivoter une métrique ---
    def pivot_metric(metric_prefix, stat='mean'):
        """
        Retourne un DataFrame pivoté : index=scenario, columns=method, values = metric_stat.
        Si la colonne n'existe pas, retourne None.
        """
        col_name = f"{metric_prefix}_{stat}"
        if col_name not in df_summary.columns:
            return None
        pivot = df_summary.pivot(index='scenario', columns='method', values=col_name)
        # Réindexer sur toutes les méthodes (celles manquantes auront NaN)
        pivot = pivot.reindex(columns=methods)
        return pivot

    # ================================================================
    # 1. Graphiques de comparaison (moyennes avec barres d'erreur)
    # ================================================================
    def plot_bar_comparison(metric_prefix, label, filename, round_values=False):
        pivot_mean = pivot_metric(metric_prefix, 'mean')
        pivot_std = pivot_metric(metric_prefix, 'std')
        if pivot_mean is None or pivot_std is None:
            print(f"⚠️ Métrique {metric_prefix} manquante, figure ignorée.")
            return

        pivot_mean = pivot_mean.reindex(scenarios)
        pivot_std = pivot_std.reindex(scenarios)

        if round_values:
            pivot_mean = pivot_mean.round(0)
            pivot_std = pivot_std.round(1)

        x = np.arange(len(scenarios))
        width = 0.2
        fig, ax = plt.subplots(figsize=(18, 8))
        for i, m in enumerate(methods):
            if m not in pivot_mean.columns:
                continue
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, pivot_mean[m], width, yerr=pivot_std[m],
                          capsize=3, label=m, color=colors[i], alpha=0.8)
            if round_values:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Scénario')
        ax.set_ylabel(label)
        title = f'Comparaison des méthodes - {label}'
        if round_values:
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

    # Métriques co-canal
    plot_bar_comparison('cochannel_cost', 'Coût co-canal', 'comparison_cochannel_cost.png')
    plot_bar_comparison('cochannel_conflicts', 'Conflits co-canal', 'comparison_cochannel_conflicts.png')
    plot_bar_comparison('cochannel_used_channels', 'Canaux utilisés (co-canal)', 'comparison_cochannel_used_channels.png', round_values=True)
    plot_bar_comparison('cochannel_time', "Temps d'exécution (co-canal) (s)", 'comparison_cochannel_time.png')

    # Métriques adjacentes
    plot_bar_comparison('adjacent_cost', 'Coût avec interférences adjacentes', 'comparison_adjacent_cost.png')
    plot_bar_comparison('adjacent_conflicts', 'Conflits avec interférences adjacentes', 'comparison_adjacent_conflicts.png')
    plot_bar_comparison('adjacent_used_channels', 'Canaux utilisés (adjacent)', 'comparison_adjacent_used_channels.png', round_values=True)
    plot_bar_comparison('adjacent_time', "Temps d'exécution (adjacent) (s)", 'comparison_adjacent_time.png')

    # ================================================================
    # 2. Heatmaps de classement (pour chaque métrique)
    # ================================================================
    ranking_configs = [
        ('cochannel_cost', 'coût co-canal', 'comparison_ranking_cochannel_cost.png'),
        ('cochannel_conflicts', 'conflits co-canal', 'comparison_ranking_cochannel_conflicts.png'),
        ('cochannel_used_channels', 'canaux utilisés (co-canal)', 'comparison_ranking_cochannel_used_channels.png'),
        ('cochannel_time', 'temps d\'exécution (co-canal)', 'comparison_ranking_cochannel_time.png'),
        ('adjacent_cost', 'coût adjacent', 'comparison_ranking_adjacent_cost.png'),
        ('adjacent_conflicts', 'conflits adjacents', 'comparison_ranking_adjacent_conflicts.png'),
        ('adjacent_used_channels', 'canaux utilisés (adjacent)', 'comparison_ranking_adjacent_used_channels.png'),
        ('adjacent_time', 'temps d\'exécution (adjacent)', 'comparison_ranking_adjacent_time.png')
    ]

    for metric_prefix, label, filename in ranking_configs:
        pivot_mean = pivot_metric(metric_prefix, 'mean')
        if pivot_mean is None:
            continue
        pivot_mean = pivot_mean.reindex(scenarios)
        # Ne garder que les méthodes présentes (non toutes NaN)
        present_methods = [m for m in methods if m in pivot_mean.columns and not pivot_mean[m].isna().all()]
        if len(present_methods) < 2:
            continue
        pivot_sub = pivot_mean[present_methods]
        ranks = pivot_sub.rank(axis=1, method='dense', ascending=True).astype(int)

        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(ranks.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(present_methods))
        ax.set_xticks(np.arange(len(present_methods)))
        ax.set_yticks(np.arange(len(ranks.index)))
        ax.set_xticklabels(present_methods, fontweight='bold')
        ax.set_yticklabels(ranks.index)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(ranks.index)):
            for j in range(len(present_methods)):
                ax.text(j, i, ranks.iloc[i, j],
                        ha="center", va="center",
                        color="black" if ranks.iloc[i, j] <= 2 else "white",
                        fontweight='bold', fontsize=9)
        ax.set_xlabel("Méthode", fontsize=12)
        ax.set_ylabel("Scénario", fontsize=12)
        ax.set_title(f"Classement des méthodes par scénario (1 = meilleur {label})", fontsize=14)
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6, ticks=np.arange(1, len(present_methods)+1))
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
        # Récupérer N,K depuis df_summary
        scenario_info = {}
        for scenario in scenarios_hist:
            sub = df_summary[df_summary['scenario'] == scenario]
            if not sub.empty:
                scenario_info[scenario] = {'N': sub.iloc[0]['N'], 'K': sub.iloc[0]['K']}
            else:
                scenario_info[scenario] = {'N': '?', 'K': '?'}

        for scenario in scenarios_hist:
            # Filtrer pour ce scénario
            df_scenario = df_hist[df_hist['scenario'] == scenario]

            # --- Correction du biais du survivant avec forward fill ---
            # 1. Pivoter : lignes = run_seed, colonnes = iteration, valeurs = cost_label
            pivot_df = df_scenario.pivot_table(index='run_seed', columns='iteration', values=cost_label)

            # 2. Forward fill sur les colonnes (axis=1) pour propager la dernière valeur connue
            pivot_df = pivot_df.ffill(axis=1)

            # 3. Calculer la moyenne et l'écart-type sur les runs (axis=0) pour chaque itération
            mean_costs = pivot_df.mean(axis=0)
            std_costs = pivot_df.std(axis=0)

            # 4. Reconstruire un DataFrame pour le tracé
            grouped = pd.DataFrame({
                'iteration': mean_costs.index,
                'mean': mean_costs.values,
                'std': std_costs.values
            })

            # Récupérer N et K
            info = scenario_info.get(scenario, {'N': '?', 'K': '?'})
            N = info['N']
            K = info['K']

            # Tracer la courbe
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(grouped['iteration'], grouped['mean'],
                    marker='o', linestyle='-', markersize=3, color='red', linewidth=1.5,
                    label=f'Moyenne du {cost_label}')
            ax.fill_between(grouped['iteration'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.25, color='red', label='Écart-type (±1σ)')
            ax.set_xlabel("Itérations")
            ylabel = "coût co-canal" if "cochannel" in cost_label else "coût avec interférences adjacentes"
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
    # 4. FIGURES DES STATISTIQUES MANQUANTES (Médianes, Min, Max, IC)
    # ================================================================
    try:
        df_median = pd.read_csv(config.STATS_MEDIAN_CSV)
        df_min = pd.read_csv(config.STATS_MIN_CSV)
        df_max = pd.read_csv(config.STATS_MAX_CSV)
        df_ci = pd.read_csv(config.STATS_CI_CSV)
        df_iter_co = pd.read_csv(config.STATS_ITERATIONS_CO_CSV)
        df_iter_adj = pd.read_csv(config.STATS_ITERATIONS_ADJ_CSV)
    except FileNotFoundError as e:
        print(f"⚠️ Fichier CSV manquant : {e}. Figures supplémentaires ignorées.")
    else:
        def plot_statistic(df_stat, stat_name, metric_prefix, label, filename_prefix, round_values=False):
            col_name = f"{metric_prefix}_{stat_name}"
            if col_name not in df_stat.columns:
                print(f"⚠️ Colonne {col_name} manquante pour {metric_prefix}")
                return
            pivot = df_stat.pivot(index='scenario', columns='method', values=col_name)
            pivot = pivot.reindex(columns=methods)
            pivot = pivot.reindex(scenarios)
            if pivot.empty or pivot.isna().all().all():
                return

            if round_values:
                pivot = pivot.round(0)

            fig, ax = plt.subplots(figsize=(16, 8))
            x = np.arange(len(scenarios))
            width = 0.2
            for i, m in enumerate(methods):
                if m not in pivot.columns or pivot[m].isna().all():
                    continue
                offset = (i - 1.5) * width
                bars = ax.bar(x + offset, pivot[m], width, label=m, color=colors[i % len(colors)], alpha=0.8)
                if round_values:
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

            ax.set_xlabel('Scénario')
            ax.set_ylabel(label)
            ax.set_title(f'{label} - {stat_name.capitalize()} par méthode')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=90, fontsize=8)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(figures_dir / f"{filename_prefix}_{metric_prefix}_{stat_name}.png", dpi=300)
            plt.close(fig)
            print(f"✅ Figure {filename_prefix}_{metric_prefix}_{stat_name}.png sauvegardée.")

        # Métriques pour les statistiques (médianes, min, max)
        stat_metrics = [
            ('cochannel_cost', 'Coût co-canal'),
            ('cochannel_conflicts', 'Conflits co-canal'),
            ('cochannel_used_channels', 'Canaux utilisés (co-canal)'),
            ('cochannel_time', "Temps d'exécution (co-canal) (s)"),
            ('adjacent_cost', 'Coût adjacent'),
            ('adjacent_conflicts', 'Conflits adjacents'),
            ('adjacent_used_channels', 'Canaux utilisés (adjacent)'),
            ('adjacent_time', "Temps d'exécution (adjacent) (s)")
        ]

        for metric_prefix, label in stat_metrics:
            round_vals = 'used_channels' in metric_prefix
            plot_statistic(df_median, 'median', metric_prefix, label, 'median', round_vals)
            plot_statistic(df_min, 'min', metric_prefix, label, 'min', round_vals)
            plot_statistic(df_max, 'max', metric_prefix, label, 'max', round_vals)

        # --- 4.1 Intervalles de confiance (style professionnel) ---
        ci_metrics = [
            ('cochannel_cost', 'Coût co-canal'),
            ('cochannel_conflicts', 'Conflits co-canal'),
            ('cochannel_used_channels', 'Canaux utilisés (co-canal)'),
            ('cochannel_time', "Temps d'exécution (co-canal) (s)"),
            ('adjacent_cost', 'Coût adjacent'),
            ('adjacent_conflicts', 'Conflits adjacents'),
            ('adjacent_used_channels', 'Canaux utilisés (adjacent)'),
            ('adjacent_time', "Temps d'exécution (adjacent) (s)")
        ]

        for metric_prefix, label in ci_metrics:
            low_col = f"{metric_prefix}_ci_low"
            high_col = f"{metric_prefix}_ci_high"
            if low_col not in df_ci.columns or high_col not in df_ci.columns:
                print(f"⚠️ Colonnes {low_col} ou {high_col} manquantes")
                continue

            pivot_low = df_ci.pivot(index='scenario', columns='method', values=low_col)
            pivot_high = df_ci.pivot(index='scenario', columns='method', values=high_col)
            pivot_low = pivot_low.reindex(columns=methods).reindex(scenarios)
            pivot_high = pivot_high.reindex(columns=methods).reindex(scenarios)

            fig, ax = plt.subplots(figsize=(16, 8))
            x = np.arange(len(scenarios))
            width = 0.2

            for i, m in enumerate(methods):
                if m not in pivot_low.columns or pivot_low[m].isna().all():
                    continue
                lows = pivot_low[m].values
                highs = pivot_high[m].values
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
            plt.savefig(figures_dir / f"confidence_interval_{metric_prefix}.png", dpi=300)
            plt.close(fig)
            print(f"✅ Figure confidence_interval_{metric_prefix}.png sauvegardée.")

        # --- 4.2 Figure du nombre de balayages (itérations) pour co-canal et adjacent ---
        # Co-canal
        if not df_iter_co.empty and 'iter_mean_co' in df_iter_co.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(df_iter_co['scenario']))
            ax.bar(x, df_iter_co['iter_mean_co'], yerr=df_iter_co['iter_std_co'], capsize=5,
                   color='teal', alpha=0.7, label='Moyenne ± écart-type (co-canal)')
            ax.set_xticks(x)
            ax.set_xticklabels(df_iter_co['scenario'])
            ax.set_xlabel('Scénario')
            ax.set_ylabel("Nombre de balayages effectués")
            ax.set_title("BD-CeNN (co-canal) - Nombre d'itérations avant arrêt")
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            for i, row in df_iter_co.iterrows():
                ax.text(i, row['iter_mean_co'] + 0.5, f"{row['iter_mean_co']:.1f}", ha='center', fontsize=9)
            plt.tight_layout()
            plt.savefig(figures_dir / "bd_iterations_histogram_co.png", dpi=300)
            plt.close(fig)
            print("✅ Figure bd_iterations_histogram_co.png sauvegardée.")

        # Adjacent
        if not df_iter_adj.empty and 'iter_mean_adj' in df_iter_adj.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(df_iter_adj['scenario']))
            ax.bar(x, df_iter_adj['iter_mean_adj'], yerr=df_iter_adj['iter_std_adj'], capsize=5,
                   color='coral', alpha=0.7, label='Moyenne ± écart-type (adjacent)')
            ax.set_xticks(x)
            ax.set_xticklabels(df_iter_adj['scenario'])
            ax.set_xlabel('Scénario')
            ax.set_ylabel("Nombre de balayages effectués")
            ax.set_title("BD-CeNN (adjacent) - Nombre d'itérations avant arrêt")
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            for i, row in df_iter_adj.iterrows():
                ax.text(i, row['iter_mean_adj'] + 0.5, f"{row['iter_mean_adj']:.1f}", ha='center', fontsize=9)
            plt.tight_layout()
            plt.savefig(figures_dir / "bd_iterations_histogram_adj.png", dpi=300)
            plt.close(fig)
            print("✅ Figure bd_iterations_histogram_adj.png sauvegardée.")

    # --- 5. Exporter le tableau récapitulatif au format CSV ---
    df_summary.to_csv(summary_csv, index=False)
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