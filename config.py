#config.py
"""
Central configuration module for paths, solver parameters, and scenario definitions.
All global variables and environment configurations are managed here.
"""

from pathlib import Path

# --- 1. Project Root Directory ---
BASE_DIR = Path(__file__).resolve().parent

# --- 2. Main Data Directories ---
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# --- 3. Result Subdirectories ---
EXCEL_DIR = RESULTS_DIR / "excel"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
LLM_LOGS_DIR = RESULTS_DIR / "llm_logs"

# --- 4. Specific Data and Output Files ---
SCENARIOS_FILE = DATA_DIR / "scenarios_data.json"

VALIDATION_EXCEL_FILE = EXCEL_DIR / "validation_results.xlsx"
CONVERGENCE_HISTORY_FILE = CSV_DIR / "convergence_history.csv"
CONVERGENCE_HISTORY_COCHANNEL_FILE = CSV_DIR / "convergence_history_cochannel.csv"
SUMMARY_CSV_FILE = CSV_DIR / "validation_summary_table.csv"
FULL_TABLE_CSV_FILE = CSV_DIR / "comparison_full_table.csv"

# Extra CSV files for statistical analysis
STATS_MEDIAN_CSV = CSV_DIR / "stats_median.csv"
STATS_MIN_CSV = CSV_DIR / "stats_min.csv"
STATS_MAX_CSV = CSV_DIR / "stats_max.csv"
STATS_CI_CSV = CSV_DIR / "stats_confidence_interval.csv"

# Iterations statistics CSV files
STATS_ITERATIONS_CSV = CSV_DIR / "stats_iterations.csv"
STATS_ITERATIONS_CO_CSV = CSV_DIR / "stats_iterations_co.csv"
STATS_ITERATIONS_ADJ_CSV = CSV_DIR / "stats_iterations_adj.csv"
STATS_BEST_ITERATION_CO_CSV = CSV_DIR / "stats_best_iteration_co.csv"
STATS_BEST_ITERATION_ADJ_CSV = CSV_DIR / "stats_best_iteration_adj.csv"

# --- 5. General Hyperparameters ---
NUM_RUNS = 30
MAX_ITER_BD = 50          # Maximum iterations per cellular neural network restart
NUM_RESTARTS = 10         # Number of random restarts for BD-CeNN solver
PATIENCE = 10             # Iteration window for convergence stagnation detection

# --- 6. Topology Seeds for 30 Independent Runs ---
TOPOLOGY_SEEDS = list(range(1, NUM_RUNS + 1))  # [1, 2, ..., 30]

# --- 7. Parameters for Restart Experiments (E9) ---
RESTART_EXPERIMENT_VALUES = [1, 5, 10, 20]

# --- 8. Directory Verification Logic ---
def ensure_dirs():
    """
    Safely creates all required project directories if they do not exist.
    Must be called explicitly at the application entrypoint.
    """
    dirs = [DATA_DIR, RESULTS_DIR, EXCEL_DIR, CSV_DIR, FIGURES_DIR, LOGS_DIR, LLM_LOGS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# Run verification upon import
ensure_dirs()

# --- 9. Standardized Scenario Builds ---
def build_scenarios():
    """
    Defines 7 distinct scenario families according to practical network guidelines:
    S1 - Small visual graph (N=8, K=3) for mathematical verification
    S2 - Medium network (N=30, K=4)
    S3 - Dense network (N=50, K=6, high distance threshold)
    S4 - Severely constrained spectrum (N=50, K=2)
    S5 - Scalability testbed (N=100, K=8)
    S6 - Noisy measurement network (N=50, K=4, threshold=35) - used in E7
    S7 - Dynamically evolving network topology (N=45, K=5, threshold=30) - used in E8
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