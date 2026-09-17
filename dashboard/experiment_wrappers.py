"""
Wrappers around the batch experiment functions defined in experiments.py.

Each wrapper:
    1. Lazy-imports experiments.py (avoids crashing when scenarios_data.json is missing)
    2. Captures stdout via ThreadSafeLogBuffer
    3. Reports success/failure via structured return dict
    4. Preserves all 30-seed logic and reproducibility

None of these wrappers modify the underlying run_experiment_EX() logic.
"""

from pathlib import Path
from typing import Optional

from dashboard.log_capture import ThreadSafeLogBuffer, capture_stdout


# ---------------------------------------------------------------------------
# Metadata for each experiment
# ---------------------------------------------------------------------------

EXPERIMENTS_METADATA = {
    "E1": {
        "code": "E1",
        "title": "Verification visuelle",
        "description": (
            "Genere les graphes colories pour chaque methode sur le scenario S1 "
            "(petit reseau N=8, K=3). Produit les tables de correspondance "
            "cellule -> canal et les metriques par methode. Sortie separee "
            "co-canal et adjacent."
        ),
        "scenario": "S1",
        "seeds_used": "1 (single seed)",
        "estimated_time": "< 1 minute",
        "outputs_dir": "results/figures/E1/ et results/csv/E1/",
        "supports_cci": True,
        "supports_aci": True,
    },
    "E2": {
        "code": "E2",
        "title": "Comparaison globale (campagne factorielle)",
        "description": (
            "Etudie l'impact de N (30, 50), K (4, 6) et densite (Faible, Moyenne, "
            "Forte) sur les 4 methodes. Utilise 30 topologies (seeds 1 a 30) par "
            "configuration. Produit 4 figures condensees par mode (cout, conflits, "
            "canaux, temps)."
        ),
        "scenario": "S2 et S3",
        "seeds_used": "30 seeds fixes (1 a 30)",
        "estimated_time": "5-10 minutes",
        "outputs_dir": "results/figures/E2/ et results/csv/E2/",
        "supports_cci": True,
        "supports_aci": True,
    },
    "E3": {
        "code": "E3",
        "title": "Impact du nombre de canaux K",
        "description": (
            "Mesure l'evolution du cout et des conflits pour K variant dans "
            "{2, 3, 4, 6, 8}. Utilise le scenario S4 (N=50) avec 30 topologies "
            "par valeur de K. Sortie moyenne + ecart-type."
        ),
        "scenario": "S4 (N=50)",
        "seeds_used": "30 seeds fixes (1 a 30)",
        "estimated_time": "5-10 minutes",
        "outputs_dir": "results/figures/E3/ et results/csv/E3/",
        "supports_cci": True,
        "supports_aci": True,
    },
    "E4": {
        "code": "E4",
        "title": "Impact de la densite",
        "description": (
            "Fait varier le threshold (30, 50, 70) sur le scenario S3 (N=50, K=6) "
            "pour tester trois niveaux de densite (Faible, Moyenne, Forte). "
            "30 seeds par niveau."
        ),
        "scenario": "S3 (N=50, K=6)",
        "seeds_used": "30 seeds fixes (1 a 30)",
        "estimated_time": "5-8 minutes",
        "outputs_dir": "results/figures/E4/ et results/csv/E4/",
        "supports_cci": True,
        "supports_aci": True,
    },
    "E6": {
        "code": "E6",
        "title": "Scalabilite (temps, cout, iterations vs N)",
        "description": (
            "Etudie N dans {20, 30, 50, 100, 200} avec K=8 fixe. Genere les "
            "topologies a la volee (area=300, threshold=50). 30 seeds par N. "
            "Attention : N=200 est le plus lourd."
        ),
        "scenario": "Genere en interne",
        "seeds_used": "30 seeds fixes (1 a 30)",
        "estimated_time": "15-30 minutes (N=200 dominant)",
        "outputs_dir": "results/figures/E6/ et results/csv/E6/",
        "supports_cci": True,
        "supports_aci": True,
    },
    "E7": {
        "code": "E7",
        "title": "Robustesse au bruit",
        "description": (
            "Perturbe la matrice W du scenario S6 (N=50, K=4) avec 3 niveaux "
            "de bruit (5%, 10%, 20%). Compare la solution BD-CeNN sur W_propre "
            "vs W_bruitee. Mesure cout relatif et taux de reaffectation."
        ),
        "scenario": "S6 (N=50, K=4)",
        "seeds_used": "30 seeds fixes (1 a 30)",
        "estimated_time": "8-12 minutes",
        "outputs_dir": "results/figures/E7/ et results/csv/E7/",
        "supports_cci": True,
        "supports_aci": True,
    },
    "E8": {
        "code": "E8",
        "title": "Reseau dynamique (Warm Start vs Cold Start)",
        "description": (
            "Perturbe le scenario S7 (N=45, K=5) avec 3 niveaux de modification "
            "(5%, 10%, 20%). Compare warm start (a partir de la solution avant "
            "perturbation) vs cold start (aleatoire). Mesure temps, cout, "
            "reaffectations."
        ),
        "scenario": "S7 (N=45, K=5)",
        "seeds_used": "30 seeds fixes (1 a 30)",
        "estimated_time": "10-15 minutes",
        "outputs_dir": "results/figures/E8/ et results/csv/E8/",
        "supports_cci": True,
        "supports_aci": True,
    },
    "E9": {
        "code": "E9",
        "title": "Minima locaux (effet des redemarrages)",
        "description": (
            "Compare BD-CeNN avec {1, 5, 10, 20} redemarrages face a Greedy avec "
            "{1, 5, 10, 20} ordres de parcours differents. Utilise S3 (N=50, K=6). "
            "Illustre l'importance du multistart."
        ),
        "scenario": "S3 (N=50, K=6)",
        "seeds_used": "30 seeds fixes (1 a 30)",
        "estimated_time": "12-20 minutes",
        "outputs_dir": "results/figures/E9/ et results/csv/E9/",
        "supports_cci": True,
        "supports_aci": True,
    },
    "E10": {
        "code": "E10",
        "title": "Audit LLM sur 20 cas representatifs",
        "description": (
            "Applique le pipeline LLM sur 20 cas types (couvrant S1-S7, bruit, "
            "dynamique, minima). Genere des PDF individuels avec badge de "
            "certification et un CSV recapitulatif. NECESSITE UNE CLE API "
            "GOOGLE_API_KEY. Duree variable selon la latence LLM."
        ),
        "scenario": "20 cas mixtes",
        "seeds_used": "Varie selon le cas",
        "estimated_time": "20-40 minutes (depend du LLM)",
        "outputs_dir": "results/csv/E10/ et results/llm_logs/",
        "supports_cci": True,
        "supports_aci": True,
    },
}


def get_experiment_metadata(code: str) -> dict:
    """Returns metadata dict for an experiment code (E1..E10)."""
    return EXPERIMENTS_METADATA.get(code, {})


def list_experiments() -> list:
    """Returns the ordered list of available experiment codes."""
    return list(EXPERIMENTS_METADATA.keys())


# ---------------------------------------------------------------------------
# Lazy import of experiments module (prevents crash when JSON is missing)
# ---------------------------------------------------------------------------

_experiments_module = None


def _load_experiments_module():
    """
    Lazily imports the experiments module.
    Raises RuntimeError with a clear message if the underlying data is missing.
    """
    global _experiments_module
    if _experiments_module is not None:
        return _experiments_module

    import config
    if not Path(config.SCENARIOS_FILE).exists():
        raise RuntimeError(
            f"Fichier {config.SCENARIOS_FILE.name} introuvable. "
            f"Regenerez les topologies via le bouton de la sidebar."
        )

    try:
        import experiments  # this line triggers the module-level JSON load
        _experiments_module = experiments
        return experiments
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger le module experiments : "
            f"{type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# Individual experiment runners with log capture
# ---------------------------------------------------------------------------

def run_experiment(code: str, log_buffer: ThreadSafeLogBuffer,
                    also_print: bool = True) -> dict:
    """
    Runs a single experiment by code (E1..E10) with stdout capture.

    Args:
        code: Experiment identifier (e.g., "E3")
        log_buffer: Buffer to capture stdout into
        also_print: If True, also prints to real stdout for debugging

    Returns:
        Result dict: {"code": str, "success": bool, "error": str|None,
                      "duration_seconds": float}
    """
    import time
    start = time.perf_counter()
    result = {
        "code": code,
        "success": False,
        "error": None,
        "duration_seconds": 0.0,
    }

    try:
        exp_mod = _load_experiments_module()
        func_name = f"run_experiment_{code}"
        func = getattr(exp_mod, func_name, None)
        if func is None:
            raise AttributeError(
                f"Fonction {func_name}() introuvable dans experiments.py"
            )

        with capture_stdout(log_buffer, also_print=also_print):
            print(f"===== Debut {code} =====")
            func(verbose=False)
            print(f"===== Fin {code} =====")

        result["success"] = True
    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        log_buffer.write(f"\n[ERREUR] {result['error']}\n{result['traceback']}\n")
    finally:
        result["duration_seconds"] = time.perf_counter() - start

    return result


def run_multiple_experiments(codes: list, log_buffer: ThreadSafeLogBuffer,
                              stop_flag=None, also_print: bool = True) -> dict:
    """
    Runs a sequence of experiments serially (never in parallel to preserve
    reproducibility of np.random.seed calls).

    Args:
        codes: List of experiment codes to run
        log_buffer: Shared log buffer
        stop_flag: Optional threading.Event; if set(), aborts before the next experiment
        also_print: If True, also prints to real stdout

    Returns:
        Summary dict with per-experiment results
    """
    import time
    summary = {
        "total": len(codes),
        "completed": 0,
        "succeeded": 0,
        "failed": 0,
        "aborted": False,
        "per_experiment": [],
        "total_duration_seconds": 0.0,
    }

    global_start = time.perf_counter()

    for i, code in enumerate(codes):
        if stop_flag is not None and stop_flag.is_set():
            log_buffer.write(f"\n[ANNULATION] Arret demande par l'utilisateur.\n")
            summary["aborted"] = True
            break

        log_buffer.write(f"\n{'='*70}\n")
        log_buffer.write(f"EXPERIENCE {i+1}/{len(codes)} : {code}\n")
        log_buffer.write(f"{'='*70}\n")

        res = run_experiment(code, log_buffer, also_print=also_print)
        summary["per_experiment"].append(res)
        summary["completed"] += 1
        if res["success"]:
            summary["succeeded"] += 1
        else:
            summary["failed"] += 1

    summary["total_duration_seconds"] = time.perf_counter() - global_start
    return summary