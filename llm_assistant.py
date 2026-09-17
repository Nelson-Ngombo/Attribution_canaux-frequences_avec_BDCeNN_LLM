# llm_assistant.py
"""
LLM Assistant module for BD-CeNN + LLM thesis project.

Contains:
    A. Prompt generator with Zero-Hallucination Directive (in French)
    B. LLM interface with Google Gemini API (lazy initialization)
    C. Independent mathematical audit controller (Regex parser)
    D. PDF report generation
    E. Constrained regeneration mechanism

Author: Nelson Ngombo
"""

import os
import re
import json
import time
import random
import threading
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# 0. SUPPRESSION OF COSMETIC SDK WARNINGS
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*automatic function calling.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*AFC.*", category=UserWarning)


# -----------------------------------------------------------------------------
# 1. GLOBAL CONFIGURATION
# -----------------------------------------------------------------------------
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# -----------------------------------------------------------------------------
# Gemini models to use (preserved exactly as provided by user)
# -----------------------------------------------------------------------------
GOOGLE_MODEL_PREFERRED = "gemini-3.6-flash"
GOOGLE_MODEL_FALLBACKS = [
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

# -----------------------------------------------------------------------------
# Timing and retry constants
# -----------------------------------------------------------------------------
INTER_REQUEST_DELAY = 5.0
LLM_TIMEOUT_SECONDS = 60
LLM_TIMEOUT_RETRIES = 3
LLM_TIMEOUT_DELAY = 60

# -----------------------------------------------------------------------------
# Directories (lazy-created)
# -----------------------------------------------------------------------------
from config import LLM_LOGS_DIR, FIGURES_DIR, CSV_DIR

LLM_LOGS_DIR = Path(LLM_LOGS_DIR)
LLM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
LLM_REPORTS_DIR = LLM_LOGS_DIR / "reports_pdf"
LLM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LLM_RAW_DIR = LLM_LOGS_DIR / "raw_responses"
LLM_RAW_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# LAZY CLIENT INITIALIZATION (to prevent dashboard crash on import)
# -----------------------------------------------------------------------------
_client = None
_active_model = None
_LAST_SUCCESSFUL_MODEL = None
_initialization_error = None


def _lazy_init_client():
    """
    Lazily initializes the Gemini client and probes for an available model.
    Returns (success: bool, error_message: str).
    Never raises: caller must check the return value.
    """
    global _client, _active_model, _initialization_error

    if _client is not None and _active_model is not None:
        return True, None

    if not GOOGLE_API_KEY:
        _initialization_error = (
            "GOOGLE_API_KEY absente. Verifiez votre fichier .env a la racine du projet."
        )
        return False, _initialization_error

    try:
        from google import genai
        from google.genai import types

        _client = genai.Client(api_key=GOOGLE_API_KEY)

        candidates = [GOOGLE_MODEL_PREFERRED] + [
            f for f in GOOGLE_MODEL_FALLBACKS if f != GOOGLE_MODEL_PREFERRED
        ]

        for c in candidates:
            try:
                _client.models.generate_content(
                    model=c,
                    contents="ping",
                    config=types.GenerateContentConfig(max_output_tokens=5),
                )
                _active_model = c
                _initialization_error = None
                return True, None
            except Exception:
                continue

        _initialization_error = (
            f"Aucun modele Gemini disponible parmi {candidates}. "
            f"Verifiez votre cle API et l'etat du service Gemini."
        )
        return False, _initialization_error

    except ImportError:
        _initialization_error = (
            "Package google-genai non installe. Executez: pip install google-genai"
        )
        return False, _initialization_error
    except Exception as e:
        _initialization_error = f"Erreur d'initialisation LLM: {type(e).__name__}: {e}"
        return False, _initialization_error


def get_active_model() -> str:
    """Returns the currently active model name, or empty string if not initialized."""
    return _active_model or ""


def is_llm_available() -> bool:
    """Non-blocking check of LLM availability."""
    return _client is not None and _active_model is not None


def get_initialization_error() -> str:
    """Returns the last initialization error message, if any."""
    return _initialization_error or ""


def _get_mode_folders(model_label: str):
    """
    Returns (raw_dir, reports_dir) for the given interference mode.
    """
    if not model_label:
        sub = "misc"
    elif "CCI+ACI" in model_label or "adjacent" in model_label.lower():
        sub = "adjacent"
    else:
        sub = "cochannel"
    raw_dir = LLM_RAW_DIR / sub
    rep_dir = LLM_REPORTS_DIR / sub
    raw_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, rep_dir


# -----------------------------------------------------------------------------
# 2. FORMATTING UTILITIES
# -----------------------------------------------------------------------------

def _fmt(value, ndigits=2):
    """Formats a numeric value for prompt inclusion and reporting."""
    try:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{value:.{ndigits}f}"
    except Exception:
        pass
    return str(value)


def _conflicts_label(model_label: str) -> str:
    """Returns the appropriate conflict line label for the interference mode."""
    if model_label == "CCI+ACI":
        return "Conflits totaux (CCI+ACI)"
    return "Conflits co-canal (CCI)"


# -----------------------------------------------------------------------------
# A. PROMPT BUILDER WITH ZERO-HALLUCINATION DIRECTIVE (IN FRENCH)
# -----------------------------------------------------------------------------

ZERO_HALLUCINATION_DIRECTIVE = """
=== DIRECTIVE SYSTEME (ZERO-HALLUCINATION) ===
Tu es un assistant d'ingenierie radio. Ton role est d'auditer et d'expliquer
les resultats d'une simulation d'attribution de canaux.

REGLES STRICTES :
1. Tu ne dois JAMAIS inventer, alterer ou extrapoler un chiffre.
2. Tu dois citer EXCLUSIVEMENT les valeurs numeriques presentes dans les
   donnees fournies.
3. Si une information n'est pas presente dans les donnees, ecris simplement :
   "Information non disponible dans les donnees fournies."
4. Toute valeur numerique que tu cites doit pouvoir etre retrouvee textuellement
   dans le bloc "DONNEES DE SIMULATION" ci-dessous.
5. Si tu detectes une incoherence dans les donnees, signale-le sans inventer.
6. Redige une analyse structuree et concise, comme un rapport d'ingenieur.

=== REGLE D'ENUMERATION ===
- Pour enumerer tes points, utilise OBLIGATOIREMENT des LETTRES MAJUSCULES
  suivies d'une parenthese fermante : A), B), C), D)...
- N'utilise JAMAIS de chiffres suivis d'un point (1., 2., 3...), car tout
  chiffre dans ta reponse est interprete comme une DONNEE NUMERIQUE de la
  simulation et sera verifie par le controleur automatique.

=== REGLE SUR LES NOMBRES (TRES IMPORTANTE) ===
Un controleur automatique independant verifie CHAQUE nombre ecrit en chiffres
dans ta reponse. Tout chiffre qui n'est pas un resultat de la simulation
fournie sera CONSIDERE COMME UNE HALLUCINATION NUMERIQUE, meme s'il s'agit
d'un simple numero dans une phrase.

  REGLE 1 - RESULTATS DE SIMULATION : EN CHIFFRES.
  Ecris en chiffres UNIQUEMENT les valeurs qui proviennent du bloc
  "DONNEES DE SIMULATION" ci-dessous (cout, conflits, temps, iterations,
  canaux utilises, N, K, seed, identifiant du cas, nom du scenario).

  REGLE 2 - NOMBRES HORS SIMULATION : EN TOUTES LETTRES.
  Tous les autres nombres que tu souhaites introduire dans ton texte
  (numeros d'ordre, bornes d'echelle, effectifs generiques, dates,
  numeros de version, etc.) DOIVENT etre ecrits EN TOUTES LETTRES.

  EXEMPLES CORRECTS :
    "sur les trois baselines..."
    "au cours des deux premieres iterations..."
    "la note de confiance est quatre sur cinq"
    "Le cout final s'eleve a 37.75"  (37.75 = resultat, donc en chiffres)

  EXEMPLES INTERDITS :
    "sur les 3 baselines..."
    "pendant les 2 premieres..."
    "la note de confiance est 4 sur 5"

POURQUOI CETTE REGLE ?
Le controleur automatique ne fait pas la difference entre un chiffre qui est
un RESULTAT de la simulation et un chiffre qui est un simple NUMERO dans une
phrase. Ecrire les numeros non-resultats en toutes lettres elimine toute
ambiguite et garantit qu'aucun faux positif d'hallucination n'est declenche.

=== REGLE DE SINCERITE ===
- Reste strictement factuel dans tes comparaisons. Si une baseline (Random,
  Greedy, DSATUR) est meilleure que BD-CeNN sur un critere (cout ou temps),
  dis-le explicitement sans enjoliver.
=== FIN DIRECTIVE ===
"""


def build_prompt(case_data: dict) -> str:
    """
    Builds a structured prompt with guardrails for a simulation case.
    """
    m = case_data.get("metrics", {})
    b = case_data.get("baselines", {})
    model_label = case_data.get("model_label", "CCI-only")

    baseline_lines = []
    for name in ["Random", "Greedy", "DSATUR"]:
        if name in b:
            baseline_lines.append(
                f"  - {name} : cout = {_fmt(b[name].get('cost'))}, "
                f"conflits = {_fmt(b[name].get('conflicts'))}, "
                f"temps = {_fmt(b[name].get('time'), 6)} s"
            )
    baseline_str = "\n".join(baseline_lines) if baseline_lines else "  (aucune baseline fournie)"

    conflicting_cells = case_data.get("conflicting_cells", [])
    conflicting_str = (
        ", ".join(str(c) for c in conflicting_cells)
        if conflicting_cells
        else "aucune (ou non disponible)"
    )

    if model_label == "CCI+ACI":
        conflicts_label = "Conflits totaux (CCI+ACI)"
        mode_note = (
            "Le cout et le nombre de conflits presentes ci-dessous integrent\n"
            "A LA FOIS les interferences co-canal (meme canal) ET les\n"
            "interferences entre canaux adjacents. C'est UN SEUL chiffre\n"
            "combine, pas deux chiffres separes."
        )
    else:
        conflicts_label = "Conflits co-canal (CCI)"
        mode_note = (
            "Le cout et le nombre de conflits presentes ci-dessous prennent en\n"
            "compte UNIQUEMENT les interferences co-canal (meme canal attribue\n"
            "a deux cellules interferentes)."
        )

    topology_meta = case_data.get("topology_meta", {})
    if topology_meta:
        meta_str = (
            f"  - Seed de la topologie (fichier JSON) : {topology_meta.get('seed', 'N/A')}\n"
            f"  - Threshold (seuil de portee radio)    : {topology_meta.get('threshold', 'N/A')}\n"
            f"  - Nombre de cellules N                 : {topology_meta.get('N', 'N/A')}\n"
            f"  - Nombre de canaux K                   : {topology_meta.get('K', 'N/A')}"
        )
    else:
        meta_str = "  (metadonnees non disponibles)"

    cell_positions = case_data.get("cell_positions", {})
    if cell_positions and len(cell_positions) <= 20:
        pos_lines = []
        for cid in sorted(cell_positions.keys()):
            x, y = cell_positions[cid]
            pos_lines.append(f"  - Cellule {cid:3d} : (x = {x:.2f}, y = {y:.2f})")
        positions_str = "\n".join(pos_lines)
    elif cell_positions:
        pos_lines = []
        for cid in sorted(cell_positions.keys())[:20]:
            x, y = cell_positions[cid]
            pos_lines.append(f"  - Cellule {cid:3d} : (x = {x:.2f}, y = {y:.2f})")
        pos_lines.append(f"  ... ({len(cell_positions) - 20} autres cellules non affichees)")
        positions_str = "\n".join(pos_lines)
    else:
        positions_str = "  (positions non disponibles)"

    topology_edges = case_data.get("topology_edges", [])
    total_edges = case_data.get("total_edges", 0)
    if topology_edges:
        topology_lines = []
        for (i, j, w) in topology_edges:
            topology_lines.append(f"  - Cellule {i} <-> Cellule {j} : poids W = {w}")
        topology_body = "\n".join(topology_lines)
        if total_edges > len(topology_edges):
            topology_header = (
                f"({len(topology_edges)} aretes affichees sur {total_edges} au total ; "
                f"triees par poids decroissant - les plus critiques en premier)"
            )
        else:
            topology_header = f"({total_edges} aretes au total)"
        topology_str = f"{topology_header}\n{topology_body}"
    else:
        topology_str = "  (topologie non disponible)"

    cell_summary = case_data.get("cell_summary", [])
    if cell_summary:
        cs_lines = []
        for entry in cell_summary:
            cs_lines.append(
                f"  - Cellule {entry['cell']:3d} : "
                f"{entry['degree']:2d} voisins, "
                f"force cumulee = {entry['strength']}"
            )
        cell_summary_str = "\n".join(cs_lines)
    else:
        cell_summary_str = "  (resume non disponible)"

    allocations = case_data.get("allocations", {})
    if allocations:
        alloc_lines = []
        for method_name in ["Random", "Greedy", "DSATUR", "BD-CeNN"]:
            if method_name in allocations:
                alloc_list = allocations[method_name]
                alloc_arr = "[" + ", ".join(str(c) for c in alloc_list) + "]"
                alloc_lines.append(f"  - {method_name:8s} : {alloc_arr}")
        allocations_str = "\n".join(alloc_lines)
        allocations_header = (
            "(La i-eme valeur est le canal attribue a la cellule i. "
            "Les indices commencent a zero.)"
        )
    else:
        allocations_str = "  (affectations non disponibles)"
        allocations_header = ""

    prompt = f"""
{ZERO_HALLUCINATION_DIRECTIVE}

=== CONTEXTE DU CAS ===
- Cas : {case_data.get('case_id')}
- Scenario : {case_data.get('scenario')}
- Configuration : N = {case_data.get('N')} cellules, K = {case_data.get('K')} canaux
- Seed de topologie : {case_data.get('seed')}
- Mode d'interference : {model_label}
- Contexte : {case_data.get('context')}

=== NOTE SUR LE MODE D'INTERFERENCE ===
{mode_note}

=== DONNEES DE SIMULATION (SOLVEUR BD-CeNN) ===
- Cout initial J_0 : {_fmt(m.get('cost_initial'))}
- Cout final J(x) : {_fmt(m.get('cost_final'))}
- {conflicts_label} : {_fmt(m.get('conflicts'))}
- Temps d'execution du BD-CeNN : {_fmt(m.get('time_seconds'), 6)} s
- Nombre d'iterations avant convergence : {_fmt(m.get('iterations'))}
- Canaux utilises : {_fmt(m.get('used_channels'))}

=== BASELINES (COMPARAISON) ===
{baseline_str}

=== METADONNEES DE LA TOPOLOGIE (fichier JSON) ===
{meta_str}

=== POSITIONS DES CELLULES (coordonnees x, y) ===
{positions_str}

=== TOPOLOGIE DU RESEAU (aretes d'interference) ===
{topology_str}

=== RESUME DES CELLULES LES PLUS CONTRAINTES (top 10) ===
{cell_summary_str}

=== AFFECTATIONS PAR METHODE (cellule -> canal) ===
{allocations_header}
{allocations_str}

=== CELLULES ENCORE EN CONFLIT ===
{conflicting_str}

=== TACHES DEMANDEES (a enumerer avec des LETTRES : A), B), C)...) ===
A) Resume la qualite de la solution BD-CeNN en comparant le COUT FINAL aux
   baselines. Appuie-toi sur les AFFECTATIONS fournies ci-dessus pour
   identifier les cellules ou BD-CeNN attribue un canal different des
   baselines et qui expliquent les ecarts observes.

B) Evalue la convergence en citant explicitement le nombre d'iterations
   avant convergence.

C) Compare les PERFORMANCES DE CALCUL (TEMPS D'EXECUTION) :
   - Cite le temps d'execution du BD-CeNN et celui de chaque baseline.
   - Si une baseline est PLUS RAPIDE que BD-CeNN, dis-le franchement.

D) Analyse la TOPOLOGIE DU RESEAU :
   - Identifie les cellules les plus contraintes a l'aide du RESUME fourni.
   - Explique comment ces contraintes locales influencent la difficulte du
     probleme et la qualite de la solution obtenue.
   - Si pertinent, cite explicitement certaines aretes critiques (poids 4)
     et les cellules concernees.

E) Dans le cas ou les resultats du BD-CeNN ne sont pas fameux, as-tu une
   recommandation a faire en tant qu'assistant de l'ingenieur radio pour
   ameliorer les resultats ? Si les resultats sont satisfaisants, tu peux
   dire "pas de recommandation a faire vu les bons resultats obtenus".

F) Termine par un avis de confiance : attribue une note de confiance
   sur une echelle a cinq niveaux (le niveau maximal etant le plus eleve),
   puis justifie-la brievement. Ecris la note EN TOUTES LETTRES
   (par exemple : "quatre sur cinq"), PAS en chiffres.

Redige ton rapport de facon detaillee, longue et argumentee en francais,
de maniere professionnelle et structuree.

Rappels :
- N'utilise JAMAIS de chiffres pour enumerer. Utilise A), B), C)...
- Tous les chiffres cites doivent provenir EXCLUSIVEMENT des donnees ci-dessus.
- Pour tout autre nombre (numeros d'ordre, bornes d'echelle), utilise des
  LETTRES (ex. "trois", "quatre sur cinq").
- Ne cite PAS de "conflits CCI" ni de "conflits ACI" separement dans ce mode.
  Utilise UNIQUEMENT le chiffre "{conflicts_label}" fourni ci-dessus.
- Tu peux citer des indices de cellules (par exemple "la cellule 17")
  et des canaux (par exemple "canal 2") car ils font partie des donnees
  de simulation fournies.
"""
    return prompt.strip()


# -----------------------------------------------------------------------------
# B. LLM INTERFACE WITH THREAD-BASED TIMEOUT
# -----------------------------------------------------------------------------

def _call_generate_content_with_timeout(model_to_try: str, prompt: str,
                                         max_tokens: int, timeout_seconds: float):
    """
    Calls the LLM in a daemon thread with strict timeout enforcement.
    Returns (response, error_type, error_message).
    """
    from google.genai import types

    result_container = {"response": None, "error": None, "done": False}

    def _worker():
        try:
            resp = _client.models.generate_content(
                model=model_to_try,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=max_tokens,
                    top_p=0.9,
                ),
            )
            result_container["response"] = resp
        except Exception as exc:
            result_container["error"] = exc
        finally:
            result_container["done"] = True

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)

    if not result_container["done"]:
        return None, "TIMEOUT", (
            f"Aucune reponse du modele '{model_to_try}' apres "
            f"{timeout_seconds:.0f}s."
        )

    if result_container["error"] is not None:
        e = result_container["error"]
        return None, "SDK_ERROR", f"{type(e).__name__} : {e}"

    return result_container["response"], None, None


def ask_llm(prompt: str, timeout: int = 180, max_retries: int = 8,
            inter_request_delay: float = INTER_REQUEST_DELAY) -> str:
    """
    Queries the LLM with layered retry and fallback mechanisms.
    Returns text response or an "[ERREUR LLM]" prefixed error string.
    """
    global _LAST_SUCCESSFUL_MODEL

    # Lazy initialization
    ok, err = _lazy_init_client()
    if not ok:
        return f"[ERREUR LLM] {err}"

    if inter_request_delay > 0:
        time.sleep(inter_request_delay)

    all_models = [_active_model] + [m for m in GOOGLE_MODEL_FALLBACKS if m != _active_model]
    if _LAST_SUCCESSFUL_MODEL and _LAST_SUCCESSFUL_MODEL in all_models:
        candidates = [_LAST_SUCCESSFUL_MODEL] + [m for m in all_models if m != _LAST_SUCCESSFUL_MODEL]
    else:
        candidates = all_models

    total_timeout_attempts = LLM_TIMEOUT_RETRIES + 1
    final_error = None

    for timeout_attempt in range(1, total_timeout_attempts + 1):
        if timeout_attempt > 1:
            time.sleep(LLM_TIMEOUT_DELAY)

        tried_models = set()
        last_error = None
        current_max_tokens = 8000
        timeout_triggered = False

        for attempt in range(1, max_retries + 1):
            model_to_try = None
            for c in candidates:
                if c not in tried_models:
                    model_to_try = c
                    break
            if model_to_try is None:
                tried_models.clear()
                model_to_try = candidates[0]

            response, error_type, error_msg = _call_generate_content_with_timeout(
                model_to_try=model_to_try,
                prompt=prompt,
                max_tokens=current_max_tokens,
                timeout_seconds=LLM_TIMEOUT_SECONDS,
            )

            if error_type == "TIMEOUT":
                timeout_triggered = True
                last_error = f"Timeout global ({LLM_TIMEOUT_SECONDS}s) sur '{model_to_try}'"
                break

            if error_type == "SDK_ERROR":
                last_error = error_msg
                error_str = error_msg
                is_503 = "503" in error_str or "UNAVAILABLE" in error_str
                is_429 = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                is_404 = "404" in error_str or "NOT_FOUND" in error_str
                needs_rotation = is_503 or is_429 or is_404

                tried_models.add(model_to_try)

                if attempt < max_retries:
                    if needs_rotation:
                        wait = 1 + random.uniform(0, 3)
                    else:
                        wait = min(60, (2 ** (attempt + 1)) + random.uniform(0, 3))
                    time.sleep(wait)
                continue

            # SUCCESS
            extracted_text = ""
            finish_reason_str = None

            if response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, "finish_reason"):
                    finish_reason_str = str(cand.finish_reason).upper()

                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        if getattr(part, "thought", False):
                            continue
                        if hasattr(part, "text") and part.text:
                            extracted_text += part.text

            if extracted_text.strip():
                _LAST_SUCCESSFUL_MODEL = model_to_try
                return extracted_text.strip()

            if hasattr(response, "text") and response.text:
                text = response.text.strip()
                if text:
                    _LAST_SUCCESSFUL_MODEL = model_to_try
                    return text

            return "[ERREUR LLM] Reponse vide renvoyee par le modele."

        final_error = last_error
        if not timeout_triggered:
            break

    return (
        f"[ERREUR LLM] Echec apres {total_timeout_attempts} tentative(s) globale(s) "
        f"de {LLM_TIMEOUT_SECONDS:.0f}s chacune. "
        f"Verifiez votre connexion internet, votre cle API et l'etat du service "
        f"Gemini. Detail : {final_error}"
    )


# -----------------------------------------------------------------------------
# C. INDEPENDENT MATHEMATICAL AUDIT CONTROLLER
# -----------------------------------------------------------------------------

NUMBER_REGEX = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_numbers(text: str) -> list:
    """Parses all numeric literals from text into (float, raw_str) tuples."""
    results = []
    for match in NUMBER_REGEX.findall(text):
        raw = match
        normalized = raw.replace(",", ".")
        try:
            val = float(normalized)
            results.append((val, raw))
        except ValueError:
            continue
    return results


def _collect_allowed_numbers(case_data: dict) -> set:
    """
    Assembles the certified whitelist of numeric values from case_data.
    """
    allowed = set()

    def add(v):
        try:
            fv = float(v)
            allowed.add(round(fv, 6))
        except Exception:
            pass

    for key in ["N", "K", "seed"]:
        if key in case_data:
            add(case_data[key])

    cid = str(case_data.get("case_id", "")).replace("#", "")
    if cid.isdigit():
        add(int(cid))

    scenario_str = str(case_data.get("scenario", ""))
    for num_str in re.findall(r"\d+", scenario_str):
        try:
            add(int(num_str))
        except ValueError:
            pass

    for n in range(1, 6):
        add(n)

    m = case_data.get("metrics", {})
    for key in ["cost_initial", "cost_final", "conflicts",
                "time_seconds", "iterations", "used_channels"]:
        if key in m:
            add(m[key])

    for base_name, base_data in case_data.get("baselines", {}).items():
        for k in ["cost", "conflicts", "time"]:
            if k in base_data:
                add(base_data[k])

    for c in case_data.get("conflicting_cells", []):
        add(c)

    for cid_pos, xy in case_data.get("cell_positions", {}).items():
        try:
            add(int(cid_pos))
        except Exception:
            pass
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            add(xy[0])
            add(xy[1])

    for (i, j, w) in case_data.get("topology_edges", []):
        try:
            add(int(i))
            add(int(j))
            add(int(w))
        except Exception:
            pass

    if "total_edges" in case_data:
        add(case_data["total_edges"])

    if "topology_edges" in case_data:
        add(len(case_data["topology_edges"]))

    tm = case_data.get("topology_meta", {})
    for key in ["seed", "threshold", "N", "K"]:
        if key in tm:
            add(tm[key])

    allocations = case_data.get("allocations", {})
    for method, alloc_list in allocations.items():
        for c in alloc_list:
            add(c)

    for entry in case_data.get("cell_summary", []):
        add(entry.get("cell", 0))
        add(entry.get("degree", 0))
        add(entry.get("strength", 0))

    return allowed


def verify_numbers(llm_response: str, case_data: dict, tolerance: float = 0.01) -> dict:
    """Cross-checks all extracted numbers against the certified whitelist."""
    extracted = extract_numbers(llm_response)
    allowed = _collect_allowed_numbers(case_data)

    valid = []
    invented = []

    for val, raw in extracted:
        found = False
        for a in allowed:
            if a == 0 and val == 0:
                found = True
                break
            if a != 0 and abs(val - a) / abs(a) <= tolerance:
                found = True
                break
            if abs(val - a) < 1e-9:
                found = True
                break
        if found:
            valid.append((val, raw))
        else:
            invented.append((val, raw))

    accuracy = (len(valid) / len(extracted) * 100.0) if extracted else 100.0

    return {
        "all_numbers": extracted,
        "allowed_numbers": allowed,
        "valid_numbers": valid,
        "invented_numbers": invented,
        "has_hallucination": len(invented) > 0,
        "accuracy_rate": accuracy,
    }


def regenerate_with_correction(case_data: dict, original_prompt: str,
                                llm_response: str, verification: dict) -> str:
    """Constrained regeneration targeting flagged numerical errors."""
    if not verification["has_hallucination"]:
        return llm_response

    faulty_str = "\n".join(
        f"  - valeur citee : {raw} (non presente dans les donnees)"
        for _, raw in verification["invented_numbers"]
    )

    m = case_data.get("metrics", {})
    model_label = case_data.get("model_label", "CCI-only")
    conflicts_label = _conflicts_label(model_label)

    allowed_summary = (
        f"Cout initial = {_fmt(m.get('cost_initial'))} ; "
        f"Cout final = {_fmt(m.get('cost_final'))} ; "
        f"{conflicts_label} = {_fmt(m.get('conflicts'))} ; "
        f"Temps = {_fmt(m.get('time_seconds'), 6)} s ; "
        f"Iterations = {_fmt(m.get('iterations'))} ; "
        f"Canaux utilises = {_fmt(m.get('used_channels'))}."
    )

    correction_prompt = f"""
=== RECTIFICATION DEMANDEE (REPRISE INTEGRALE) ===

Nous te fournissons a nouveau la REQUETE ORIGINALE COMPLETE (car tu n'as pas
de memoire entre nos echanges). Prends-en connaissance, puis corrige ta
reponse precedente qui contenait des ERREURS NUMERIQUES.

--- DEBUT DE LA REQUETE ORIGINALE ---
{original_prompt}
--- FIN DE LA REQUETE ORIGINALE ---

Dans ta reponse precedente a cette requete, tu as cite les valeurs
numeriques suivantes :

{faulty_str}

Ces valeurs NE FIGURENT PAS dans les donnees de simulation de la requete
originale. Ce sont donc des HALLUCINATIONS NUMERIQUES.

Rappel des donnees autorisees :
{allowed_summary}

CONSIGNES DE CORRECTION :
A) Reprends integralement ta reponse en repondant aux MEMES taches.
B) Remplace toute valeur incorrecte par la valeur exacte issue des donnees.
C) Si une information est absente, ecris simplement :
   "Information non disponible dans les donnees fournies."
D) Utilise des LETTRES MAJUSCULES (A), B), C)...) pour enumerer.
E) RAPPEL DE LA REGLE DES NOMBRES :
   - Les RESULTATS de simulation doivent etre ecrits EN CHIFFRES.
   - Tous les AUTRES nombres doivent etre ecrits EN TOUTES LETTRES.
F) Ne cite PAS de "conflits CCI" ni de "conflits ACI" separement si le mode
   est CCI+ACI. Utilise UNIQUEMENT le chiffre "{conflicts_label}" fourni.

=== REPONSE PRECEDENTE A CORRIGER ===
{llm_response}
"""
    return ask_llm(correction_prompt, inter_request_delay=INTER_REQUEST_DELAY)


# -----------------------------------------------------------------------------
# D. PDF REPORT GENERATION
# -----------------------------------------------------------------------------

def generate_pdf_report(
    case_data: dict,
    prompt: str,
    raw_response: str,
    verification_initial: dict,
    corrected_response: str,
    verification_final: dict,
    output_path: Path,
    model_name: str = None,
) -> None:
    """Generates a professional PDF audit report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        return

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "TitleCustom", parent=styles["Title"],
        fontSize=16, textColor=colors.HexColor("#1a3d6d"), alignment=TA_CENTER
    )
    style_h1 = ParagraphStyle(
        "H1Custom", parent=styles["Heading1"],
        fontSize=13, textColor=colors.HexColor("#1a3d6d"), spaceAfter=6
    )
    style_h2 = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"],
        fontSize=11, textColor=colors.HexColor("#2a5d9d"), spaceAfter=4
    )
    style_body = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"],
        fontSize=9.5, leading=13, alignment=TA_LEFT
    )
    style_mono = ParagraphStyle(
        "MonoCustom", parent=styles["BodyText"],
        fontName="Courier", fontSize=8.5, leading=11
    )

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        title=f"Rapport LLM - {case_data.get('case_id')}",
    )

    story = []
    model_label = case_data.get("model_label", "CCI-only")
    conflicts_label = _conflicts_label(model_label)

    story.append(Paragraph(
        f"Rapport d'audit LLM - Attribution de canaux ({model_label})",
        style_title
    ))
    story.append(Spacer(1, 0.3*cm))

    header_data = [
        ["Cas", case_data.get("case_id", "-")],
        ["Scenario", case_data.get("scenario", "-")],
        ["Configuration", f"N={case_data.get('N')}, K={case_data.get('K')}, seed={case_data.get('seed')}"],
        ["Mode d'interference", model_label],
        ["Contexte", case_data.get("context", "-")],
        ["Modele LLM", model_name if model_name else "Non specifie"],
        ["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    header_table = Table(header_data, colWidths=[4*cm, 12*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eef7")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    if verification_final["has_hallucination"]:
        badge = (
            f'<font color="#b30000"><b>RAPPORT NON CERTIFIE</b></font> - '
            f'{len(verification_final["invented_numbers"])} valeur(s) non conforme(s) detectee(s).'
        )
    else:
        badge = (
            f'<font color="#006600"><b>RAPPORT CERTIFIE - 0 hallucination numerique</b></font> '
            f'(taux d\'exactitude : {verification_final["accuracy_rate"]:.1f}%).'
        )
    story.append(Paragraph(badge, style_body))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("1. Donnees de simulation utilisees", style_h1))
    m = case_data.get("metrics", {})
    sim_data = [
        ["Metrique", "Valeur"],
        ["Cout initial J_0", _fmt(m.get("cost_initial"))],
        ["Cout final J(x)", _fmt(m.get("cost_final"))],
        [conflicts_label, _fmt(m.get("conflicts"))],
        ["Temps d'execution (s)", _fmt(m.get("time_seconds"), 6)],
        ["Iterations (BD-CeNN)", _fmt(m.get("iterations"))],
        ["Canaux utilises", _fmt(m.get("used_channels"))],
    ]
    sim_table = Table(sim_data, colWidths=[7*cm, 5*cm])
    sim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    story.append(sim_table)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2. Reponse brute du LLM", style_h1))
    raw_escaped = raw_response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(raw_escaped.replace("\n", "<br/>"), style_body))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("3. Audit numerique automatique", style_h1))
    audit_data = [
        ["Indicateur", "Valeur"],
        ["Nombres extraits", str(len(verification_initial["all_numbers"]))],
        ["Nombres conformes (initiaux)", str(len(verification_initial["valid_numbers"]))],
        ["Nombres inventes (initiaux)", str(len(verification_initial["invented_numbers"]))],
        ["Taux d'exactitude initial", f"{verification_initial['accuracy_rate']:.1f}%"],
        ["Hallucination detectee ?", "Oui" if verification_initial["has_hallucination"] else "Non"],
    ]
    audit_table = Table(audit_data, colWidths=[9*cm, 4*cm])
    audit_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    story.append(audit_table)

    if verification_initial["has_hallucination"]:
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("4. Reponse corrigee", style_h1))
        corr_escaped = corrected_response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(corr_escaped.replace("\n", "<br/>"), style_body))

    doc.build(story)


# -----------------------------------------------------------------------------
# E. HIGH-LEVEL AUDIT PIPELINE
# -----------------------------------------------------------------------------

def audit_case(case_data: dict, max_correction_attempts: int = 2,
               model_folder: str = "cochannel") -> dict:
    """Full audit pipeline: prompt -> LLM -> verify -> correct -> report."""
    case_id = case_data.get("case_id", "unknown")
    model_label = case_data.get("model_label", "CCI-only")

    raw_dir, reports_dir = _get_mode_folders(model_label)

    prompt = build_prompt(case_data)

    prompt_file = raw_dir / f"{case_id}_prompt.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    raw_response = ask_llm(prompt)

    if raw_response.startswith("[ERREUR LLM]"):
        err_file = raw_dir / f"{case_id}_error.txt"
        with open(err_file, "w", encoding="utf-8") as f:
            f.write(raw_response)
        return {
            "case_id": case_id,
            "scenario": case_data.get("scenario"),
            "N": case_data.get("N"),
            "K": case_data.get("K"),
            "seed": case_data.get("seed"),
            "context": case_data.get("context"),
            "model_label": model_label,
            "raw_response": raw_response,
            "corrected_response": raw_response,
            "verification_initial": {
                "all_numbers": [], "allowed_numbers": set(),
                "valid_numbers": [], "invented_numbers": [],
                "has_hallucination": False, "accuracy_rate": 0.0,
            },
            "verification_final": {
                "all_numbers": [], "allowed_numbers": set(),
                "valid_numbers": [], "invented_numbers": [],
                "has_hallucination": False, "accuracy_rate": 0.0,
            },
            "correction_attempts": 0,
            "pdf_path": None,
            "status": "LLM_FAILED",
        }

    raw_file = raw_dir / f"{case_id}_raw_response.txt"
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(raw_response)

    verification_initial = verify_numbers(raw_response, case_data)

    corrected_response = raw_response
    verification_final = verification_initial
    correction_attempts = 0
    while verification_final["has_hallucination"] and correction_attempts < max_correction_attempts:
        correction_attempts += 1
        corrected_response = regenerate_with_correction(
            case_data, prompt, corrected_response, verification_final
        )

        if corrected_response.startswith("[ERREUR LLM]"):
            corrected_response = raw_response
            break

        verification_final = verify_numbers(corrected_response, case_data)
        corr_file = raw_dir / f"{case_id}_corrected_v{correction_attempts}.txt"
        with open(corr_file, "w", encoding="utf-8") as f:
            f.write(corrected_response)

    pdf_path = reports_dir / f"rapport_{case_id}_{case_data.get('scenario')}.pdf"
    generate_pdf_report(
        case_data=case_data,
        prompt=prompt,
        raw_response=raw_response,
        verification_initial=verification_initial,
        corrected_response=corrected_response,
        verification_final=verification_final,
        output_path=pdf_path,
        model_name=get_active_model(),
    )

    return {
        "case_id": case_id,
        "scenario": case_data.get("scenario"),
        "N": case_data.get("N"),
        "K": case_data.get("K"),
        "seed": case_data.get("seed"),
        "context": case_data.get("context"),
        "model_label": model_label,
        "raw_response": raw_response,
        "corrected_response": corrected_response,
        "verification_initial": verification_initial,
        "verification_final": verification_final,
        "correction_attempts": correction_attempts,
        "pdf_path": str(pdf_path),
        "status": "OK",
    }


# Compatibility aliases for legacy imports
GOOGLE_MODEL = None  # populated after lazy init