#config.py
"""
Central configuration module for paths, solver parameters, and scenario definitions.
All global variables and environment configurations are managed here.
"""

from pathlib import Path

# ============================================================================
# 1. RACINE DU PROJET
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent

# ============================================================================
# 2. DOSSIERS PRINCIPAUX
# ============================================================================
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# ============================================================================
# 3. SOUS-DOSSIERS DE RÉSULTATS
# ============================================================================
EXCEL_DIR = RESULTS_DIR / "excel"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
LLM_LOGS_DIR = RESULTS_DIR / "llm_logs"
VALIDATION_DIR = RESULTS_DIR / "validation"  # Chantier B+C : rapports d'audit

# ============================================================================
# 4. FICHIERS SPÉCIFIQUES
# ============================================================================
SCENARIOS_FILE = DATA_DIR / "scenarios_data.json"

VALIDATION_EXCEL_FILE = EXCEL_DIR / "validation_results.xlsx"
CONVERGENCE_HISTORY_FILE = CSV_DIR / "convergence_history.csv"
CONVERGENCE_HISTORY_COCHANNEL_FILE = CSV_DIR / "convergence_history_cochannel.csv"
SUMMARY_CSV_FILE = CSV_DIR / "validation_summary_table.csv"
FULL_TABLE_CSV_FILE = CSV_DIR / "comparison_full_table.csv"

# Fichiers CSV pour les statistiques supplémentaires
STATS_MEDIAN_CSV = CSV_DIR / "stats_median.csv"
STATS_MIN_CSV = CSV_DIR / "stats_min.csv"
STATS_MAX_CSV = CSV_DIR / "stats_max.csv"
STATS_CI_CSV = CSV_DIR / "stats_confidence_interval.csv"

# Fichiers CSV pour les statistiques de sweeps
STATS_SWEEPS_CSV = CSV_DIR / "stats_sweeps.csv"
STATS_SWEEPS_CO_CSV = CSV_DIR / "stats_sweeps_co.csv"
STATS_SWEEPS_ADJ_CSV = CSV_DIR / "stats_sweeps_adj.csv"
STATS_BEST_SWEEP_CO_CSV = CSV_DIR / "stats_best_sweep_co.csv"
STATS_BEST_SWEEP_ADJ_CSV = CSV_DIR / "stats_best_sweep_adj.csv"

# ============================================================================
# 5. HYPERPARAMÈTRES DU SOLVEUR BD-CeNN
# ============================================================================
# Convention : un "sweep" est un balayage complet des N cellules.
# Un "restart" est une exécution complète depuis une initialisation aléatoire.

NUM_RUNS = 30
MAX_SWEEPS_BD = 50        # Nombre maximal de sweeps par restart
NUM_RESTARTS = 10          # Nombre de restarts pour le multistart
PATIENCE_SWEEPS = 10       # Sweeps consécutifs sans variation avant arrêt

# --- Alias de rétrocompatibilité ---
# Ces alias permettent aux anciens scripts de continuer à fonctionner
# pendant la période de transition. Ils seront supprimés dans une version
# future. Tout nouveau code doit utiliser les noms ci-dessus.
MAX_ITER_BD = MAX_SWEEPS_BD      # Déprécié : utiliser MAX_SWEEPS_BD
PATIENCE = PATIENCE_SWEEPS        # Déprécié : utiliser PATIENCE_SWEEPS

# ============================================================================
# 6. SEEDS DE TOPOLOGIE POUR LES 30 INSTANCES
# ============================================================================
TOPOLOGY_SEEDS = list(range(1, NUM_RUNS + 1))  # [1, 2, ..., 30]

# ============================================================================
# 7. PARAMÈTRES POUR L'EXPÉRIENCE E9 (effet du nombre de restarts)
# ============================================================================
RESTART_EXPERIMENT_VALUES = [1, 5, 10, 20]

# ============================================================================
# 8. CRÉATION DES DOSSIERS
# ============================================================================
def ensure_dirs():
    """
    Crée tous les dossiers nécessaires s'ils n'existent pas.
    Doit être appelé explicitement au point d'entrée de l'application.
    """
    dirs = [
        DATA_DIR, RESULTS_DIR, EXCEL_DIR, CSV_DIR,
        FIGURES_DIR, LOGS_DIR, LLM_LOGS_DIR, VALIDATION_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

# ============================================================================
# 9. SCÉNARIOS (S1 à S7) — Spécification v1.0, Tableau récapitulatif
# ============================================================================
def build_scenarios():
    """
    Définit les 7 familles de scénarios conformes à la spécification :

    S1 - Petit graphe visuel (N=8, K=3)
    S2 - Réseau moyen (N=30, K=4)
    S3 - Réseau dense (N=50, K=6, seuil élevé)
    S4 - Spectre très limité (N=50, K=2)
    S5 - Scalabilité (N=100, K=8)
    S6 - Réseau bruité (N=50, K=4, seuil=35) — utilisé pour E7
    S7 - Réseau dynamique (N=45, K=5, seuil=30) — utilisé pour E8

    Returns
    -------
    dict
        Dictionnaire {nom_scenario: {N, K, area, threshold}}.
    """
    scenarios = {
        "S1": {"N": 8, "K": 3, "area": 100, "threshold": 30},
        "S2": {"N": 30, "K": 4, "area": 150, "threshold": 35},
        "S3": {"N": 50, "K": 6, "area": 200, "threshold": 60},
        "S4": {"N": 50, "K": 2, "area": 200, "threshold": 40},
        "S5": {"N": 100, "K": 8, "area": 300, "threshold": 50},
        "S6": {"N": 50, "K": 4, "area": 200, "threshold": 35},
        "S7": {"N": 45, "K": 5, "area": 200, "threshold": 30},
    }
    return scenarios


SCENARIOS = build_scenarios()