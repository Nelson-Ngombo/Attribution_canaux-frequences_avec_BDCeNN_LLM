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
MAX_ITER_BD = 10

# --- 6. Fonction pour créer les dossiers ---
def ensure_dirs():
    """Crée tous les dossiers nécessaires s'ils n'existent pas."""
    dirs = [DATA_DIR, RESULTS_DIR, EXCEL_DIR, CSV_DIR, FIGURES_DIR, LOGS_DIR, LLM_LOGS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# Création automatique des dossiers à l'import
ensure_dirs()

# Graine de base pour la reproductibilité (sera dérivée par scénario)
SEED_BASE = 42

# Fonction pour générer les scénarios
def build_scenarios():
    scenarios = {}

    # 1. Scénarios initiaux S1 à S4 (inchangés)
    scenarios["S1"] = {"N": 10, "K": 3, "area": 100, "threshold": 30}
    scenarios["S2"] = {"N": 30, "K": 5, "area": 150, "threshold": 35}
    scenarios["S3"] = {"N": 50, "K": 6, "area": 200, "threshold": 40}
    scenarios["S4"] = {"N": 50, "K": 2, "area": 200, "threshold": 40}

    # 2. Petit réseau (S5 à S14) – validation visuelle
    # S5-S10 : N variable 6..12, K variable 2..5, area=100, threshold=30
    small_sizes = [6, 8, 10, 12, 10, 10]
    small_ks    = [2, 3, 4, 5, 3, 4]
    for i, (n, k) in enumerate(zip(small_sizes, small_ks), start=5):
        scenarios[f"S{i}"] = {"N": n, "K": k, "area": 100, "threshold": 30}

    # S11-S14 : topologies particulières (on garde les mêmes paramètres mais avec des seeds différentes)
    # On laisse le générateur aléatoire, mais on utilisera des seeds décalées pour varier les formes.
    for i in range(11, 15):
        scenarios[f"S{i}"] = {"N": 10, "K": 3, "area": 100, "threshold": 30}

    # 3. Réseau moyen – diversité (S15 à S30)
    # S15-S20 : Profil rural (faible densité) – N=20, K=4, seuil bas
    for i in range(15, 21):
        scenarios[f"S{i}"] = {"N": 20, "K": 4, "area": 200, "threshold": 20}

    # S21-S25 : Profil périurbain – N=45, K variable 4,5,6
    for i, k in enumerate([4, 5, 6, 4, 5], start=21):
        scenarios[f"S{i}"] = {"N": 45, "K": k, "area": 200, "threshold": 35}

    # S26-S30 : Zones d'ombres (on simule en gardant les mêmes paramètres mais en jouant sur les seeds)
    for i in range(26, 31):
        scenarios[f"S{i}"] = {"N": 45, "K": 5, "area": 200, "threshold": 35}

    # 4. Stress-tests (S31 à S45)
    # S31-S35 : Haute densité – N=50, seuil élevé, K=6
    for i in range(31, 36):
        scenarios[f"S{i}"] = {"N": 50, "K": 6, "area": 200, "threshold": 60}

    # S36-S40 : Pénurie de canaux – N=50, K=2 ou 3
    for i, k in enumerate([2, 3, 2, 3, 2], start=36):
        scenarios[f"S{i}"] = {"N": 50, "K": k, "area": 200, "threshold": 40}

    # S41-S45 : Passage à l'échelle – N=100,150 avec K=8
    for i, n in enumerate([100, 100, 150, 150, 120], start=41):
        scenarios[f"S{i}"] = {"N": n, "K": 8, "area": 300, "threshold": 50}

    # 5. Scénarios avancés (S46 à S50)
    # S46-S48 : Priorités de trafic – on augmente artificiellement les poids pour certaines paires
    # On simule en utilisant les mêmes paramètres mais avec des seeds permettant des configurations denses
    for i in range(46, 49):
        scenarios[f"S{i}"] = {"N": 50, "K": 5, "area": 200, "threshold": 40}

    # S49-S50 : Incertitude – on ajoute du bruit en variant légèrement le seuil
    for i, th in enumerate([38, 42], start=49):
        scenarios[f"S{i}"] = {"N": 50, "K": 5, "area": 200, "threshold": th}

    return scenarios

SCENARIOS = build_scenarios()
OUTPUT_FILE = "scenarios_data.json"






