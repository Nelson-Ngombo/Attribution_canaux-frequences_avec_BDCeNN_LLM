# config.py
from pathlib import Path

# --- 1. Racine du projet ---
BASE_DIR = Path(__file__).resolve().parent

# --- 2. Dossiers principaux ---
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# --- 3. Sous-dossiers de results ---
EXCEL_DIR = RESULTS_DIR / "excel"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
LLM_LOGS_DIR = RESULTS_DIR / "llm_logs"

# --- 4. Fichiers spécifiques ---
SCENARIOS_FILE = DATA_DIR / "scenarios_data.json"

VALIDATION_EXCEL_FILE = EXCEL_DIR / "validation_results.xlsx"
CONVERGENCE_HISTORY_FILE = CSV_DIR / "convergence_history.csv"
SUMMARY_CSV_FILE = CSV_DIR / "validation_summary_table.csv"
FULL_TABLE_CSV_FILE = CSV_DIR / "comparison_full_table.csv"

# --- 5. Paramètres généraux ---
SEED_BASE = 42
NUM_RUNS = 30
MAX_ITER_BD = 50          # Nombre d'itérations maximales par lancement
NUM_RESTARTS = 10         # Nombre de redémarrages pour le BD-CeNN

# --- 6. Fonction pour créer les dossiers ---
def ensure_dirs():
    """Crée tous les dossiers nécessaires s'ils n'existent pas."""
    dirs = [DATA_DIR, RESULTS_DIR, EXCEL_DIR, CSV_DIR, FIGURES_DIR, LOGS_DIR, LLM_LOGS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

ensure_dirs()

# --- 7. Scénarios selon le guide pratique (§4.2) ---
def build_scenarios():
    """
    7 familles de scénarios conformes au guide pratique :
    S1 - Petit graphe visuel (N=8, K=3)
    S2 - Réseau moyen (N=30, K=4)
    S3 - Réseau dense (N=50, K=6, threshold élevé)
    S4 - Peu de canaux (N=50, K=2)
    S5 - Scalabilité (N=100, K=8)
    S6 - Réseau bruité (N=50, K=4, threshold 35)  - à utiliser pour E7
    S7 - Réseau dynamique (N=50, K=4, threshold 35) - à utiliser pour E8
    """
    scenarios = {
        "S1": {"N": 8, "K": 3, "area": 100, "threshold": 30},      # Petit graphe visuel à faible densité
        "S2": {"N": 30, "K": 4, "area": 150, "threshold": 35},     # Réseau moyen
        "S3": {"N": 50, "K": 6, "area": 200, "threshold": 60},     # Réseau dense
        "S4": {"N": 50, "K": 2, "area": 200, "threshold": 40},     # Peu de canaux
        "S5": {"N": 100, "K": 8, "area": 300, "threshold": 50},    # Scalabilité
        "S6": {"N": 50, "K": 4, "area": 200, "threshold": 35},     # Bruité (E7)
        "S7": {"N": 50, "K": 4, "area": 200, "threshold": 35},     # Dynamique (E8)
    }
    return scenarios

SCENARIOS = build_scenarios()




