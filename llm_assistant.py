#llm_assistant.py
"""
LLM Assistant module for BD-CeNN + LLM thesis project.

Non-blocking initialization architecture:
    - Background daemon thread probes each model sequentially
    - UI thread reads cached status instantly (never blocks)
    - Once a model responds, sidebar updates automatically
    - If all models fail, sidebar shows the final unavailable state

Fully compatible with main.py imports (get_llm_status_label,
is_llm_available, is_llm_initializing, get_initialization_error).

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
from enum import Enum

import numpy as np

# -----------------------------------------------------------------------------
# 0. SUPPRESS COSMETIC SDK WARNINGS
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*automatic function calling.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*AFC.*", category=UserWarning)


# -----------------------------------------------------------------------------
# 1. CONFIGURATION & ENVIRONMENT
# -----------------------------------------------------------------------------
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip().strip('"').strip("'")

# Qualified model names (verified against user's API key)
GOOGLE_MODEL_PREFERRED = "models/gemini-3.6-flash"
GOOGLE_MODEL_FALLBACKS = [
    "models/gemini-3.7-flash",
    "models/gemini-3.8-flash",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
]

INTER_REQUEST_DELAY = 3.0
LLM_TIMEOUT_SECONDS = 45
PROBE_TIMEOUT_SECONDS = 20

from config import LLM_LOGS_DIR, FIGURES_DIR, CSV_DIR

LLM_LOGS_DIR = Path(LLM_LOGS_DIR)
LLM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
LLM_REPORTS_DIR = LLM_LOGS_DIR / "reports_pdf"
LLM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LLM_RAW_DIR = LLM_LOGS_DIR / "raw_responses"
LLM_RAW_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. BACKGROUND INITIALIZATION STATE MACHINE
# -----------------------------------------------------------------------------

class LLMStatus(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


_state_lock = threading.Lock()
_init_lock = threading.Lock()

_state = {
    "status": LLMStatus.NOT_STARTED,
    "client": None,
    "active_model": None,
    "last_successful_model": None,
    "error_message": "",
    "probe_progress": "",
    "init_thread_started": False,
}


def _get_status() -> LLMStatus:
    with _state_lock:
        return _state["status"]


def _set_status(status: LLMStatus, model: str = None, error: str = ""):
    with _state_lock:
        _state["status"] = status
        if model is not None:
            _state["active_model"] = model
        if error:
            _state["error_message"] = error


def _set_progress(msg: str):
    with _state_lock:
        _state["probe_progress"] = msg


def _get_progress() -> str:
    with _state_lock:
        return _state["probe_progress"]


def _get_client():
    with _state_lock:
        return _state["client"]


def _set_client(client):
    with _state_lock:
        _state["client"] = client


def _get_active_model_internal() -> str:
    with _state_lock:
        return _state["active_model"] or ""


# -----------------------------------------------------------------------------
# BACKGROUND PROBE ENGINE
# -----------------------------------------------------------------------------

def _probe_single_model(client, model_name: str, types_module) -> bool:
    """
    Probes a single model with strict timeout. Returns True if responsive.
    Runs entirely in the caller thread (which is already the background init thread).
    """
    result = {"ok": False}

    def _ping():
        try:
            client.models.generate_content(
                model=model_name,
                contents="ping",
                config=types_module.GenerateContentConfig(max_output_tokens=5),
            )
            result["ok"] = True
        except Exception:
            result["ok"] = False

    t = threading.Thread(target=_ping, daemon=True)
    t.start()
    t.join(timeout=PROBE_TIMEOUT_SECONDS)

    return result["ok"]


def _background_init():
    """
    Runs in a daemon thread. Never touches Streamlit session state.
    Updates _state atomically as models are probed.
    """
    _set_status(LLMStatus.INITIALIZING)
    _set_progress("Chargement du package google-genai...")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        _set_status(
            LLMStatus.UNAVAILABLE,
            error="Package google-genai non installe. Executez : pip install google-genai"
        )
        return

    if not GOOGLE_API_KEY:
        _set_status(
            LLMStatus.UNAVAILABLE,
            error="Cle GOOGLE_API_KEY absente du fichier .env"
        )
        return

    _set_progress("Creation du client Gemini...")

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        _set_client(client)
    except Exception as e:
        _set_status(
            LLMStatus.UNAVAILABLE,
            error=f"Echec de creation du client : {type(e).__name__}: {e}"
        )
        return

    candidates = [GOOGLE_MODEL_PREFERRED] + [
        m for m in GOOGLE_MODEL_FALLBACKS if m != GOOGLE_MODEL_PREFERRED
    ]

    errors_collected = []

    for i, model_name in enumerate(candidates, start=1):
        display_name = model_name.replace("models/", "")
        _set_progress(f"Test du modele {i}/{len(candidates)} : {display_name}")

        try:
            if _probe_single_model(client, model_name, types):
                with _state_lock:
                    _state["last_successful_model"] = model_name
                _set_status(LLMStatus.AVAILABLE, model=model_name)
                _set_progress("")
                return
            else:
                errors_collected.append(f"{display_name}: timeout")
        except Exception as e:
            errors_collected.append(f"{display_name}: {type(e).__name__}")

    _set_status(
        LLMStatus.UNAVAILABLE,
        error="Aucun modele disponible. Details: " + "; ".join(errors_collected[:3])
    )
    _set_progress("")


def ensure_init_started():
    """
    Triggers the background initialization once per session.
    Fully non-blocking: returns immediately after spawning the thread.
    """
    with _init_lock:
        with _state_lock:
            if _state["init_thread_started"]:
                return
            _state["init_thread_started"] = True

    thread = threading.Thread(
        target=_background_init,
        daemon=True,
        name="llm-background-init",
    )
    thread.start()


# -----------------------------------------------------------------------------
# PUBLIC API (CALLED BY main.py AND OTHER DASHBOARD MODULES)
# -----------------------------------------------------------------------------

def is_llm_available() -> bool:
    """
    Non-blocking check. Triggers background init if not started.
    Returns True only when a model has been confirmed responsive.
    """
    ensure_init_started()
    return _get_status() == LLMStatus.AVAILABLE


def is_llm_initializing() -> bool:
    """
    Returns True while the background probe is running.
    Triggers background init if not started.
    """
    ensure_init_started()
    return _get_status() == LLMStatus.INITIALIZING


def get_active_model() -> str:
    """Returns the verified working model name (empty string if none)."""
    return _get_active_model_internal()


def get_initialization_error() -> str:
    """Returns the current error message (empty string if none)."""
    with _state_lock:
        return _state["error_message"] or ""


def get_llm_status_label() -> str:
    """
    Returns a human-readable status label for the sidebar.
    Updates dynamically as the background probe progresses.
    """
    ensure_init_started()
    status = _get_status()

    if status == LLMStatus.NOT_STARTED:
        return "Initialisation en cours..."

    if status == LLMStatus.INITIALIZING:
        progress = _get_progress()
        if progress:
            return f"Initialisation : {progress}"
        return "Initialisation en cours..."

    if status == LLMStatus.AVAILABLE:
        model = _get_active_model_internal().replace("models/", "")
        return f"Actif ({model})"

    if status == LLMStatus.UNAVAILABLE:
        return "Non disponible"

    return "Etat inconnu"


def _get_mode_folders(model_label: str):
    """Routes logging paths based on interference constraints."""
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
# 3. FORMATTING UTILITIES
# -----------------------------------------------------------------------------

def _fmt(value, ndigits=2):
    try:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{value:.{ndigits}f}"
    except Exception:
        pass
    return str(value)


def _conflicts_label(model_label: str) -> str:
    if model_label == "CCI+ACI":
        return "Conflits totaux (CCI+ACI)"
    return "Conflits co-canal (CCI)"


# -----------------------------------------------------------------------------
# A. PROMPT SYNTHESIZER WITH ZERO-HALLUCINATION DIRECTIVES (IN FRENCH)
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
        if conflicting_cells else "aucune (ou non disponible)"
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
        pos_lines = [
            f"  - Cellule {cid:3d} : (x = {xy[0]:.2f}, y = {xy[1]:.2f})"
            for cid, xy in sorted(cell_positions.items())
        ]
        positions_str = "\n".join(pos_lines)
    elif cell_positions:
        pos_lines = [
            f"  - Cellule {cid:3d} : (x = {cell_positions[cid][0]:.2f}, y = {cell_positions[cid][1]:.2f})"
            for cid in sorted(cell_positions.keys())[:20]
        ]
        pos_lines.append(f"  ... ({len(cell_positions) - 20} autres cellules non affichees)")
        positions_str = "\n".join(pos_lines)
    else:
        positions_str = "  (positions non disponibles)"

    topology_edges = case_data.get("topology_edges", [])
    total_edges = case_data.get("total_edges", 0)
    if topology_edges:
        topology_lines = [
            f"  - Cellule {i} <-> Cellule {j} : poids W = {w}"
            for (i, j, w) in topology_edges
        ]
        topology_body = "\n".join(topology_lines)
        if total_edges > len(topology_edges):
            topology_header = (
                f"({len(topology_edges)} aretes affichees sur {total_edges} au total ; "
                f"triees par poids decroissant)"
            )
        else:
            topology_header = f"({total_edges} aretes au total)"
        topology_str = f"{topology_header}\n{topology_body}"
    else:
        topology_str = "  (topologie non disponible)"

    cell_summary = case_data.get("cell_summary", [])
    if cell_summary:
        cs_lines = [
            f"  - Cellule {e['cell']:3d} : {e['degree']:2d} voisins, force cumulee = {e['strength']}"
            for e in cell_summary
        ]
        cell_summary_str = "\n".join(cs_lines)
    else:
        cell_summary_str = "  (resume non disponible)"

    allocations = case_data.get("allocations", {})
    if allocations:
        alloc_lines = [
            f"  - {mn:8s} : [{', '.join(str(c) for c in allocations[mn])}]"
            for mn in ["Random", "Greedy", "DSATUR", "BD-CeNN"]
            if mn in allocations
        ]
        allocations_str = "\n".join(alloc_lines)
        allocations_header = "(La i-eme valeur est le canal attribue a la cellule i.)"
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
- Pour tout autre nombre, utilise des LETTRES (ex. "trois", "quatre sur cinq").
- Ne cite PAS de "conflits CCI" ni de "conflits ACI" separement dans ce mode.
  Utilise UNIQUEMENT le chiffre "{conflicts_label}" fourni ci-dessus.
"""
    return prompt.strip()


# -----------------------------------------------------------------------------
# B. API INTERACTION ENGINE (FULL ERROR COVERAGE)
# -----------------------------------------------------------------------------

def _call_with_timeout(model_to_try: str, prompt: str,
                       max_tokens: int, timeout_seconds: float):
    """Executes model generation with strict timeout enforcement."""
    client = _get_client()
    if client is None:
        return None, "SDK_ERROR", "Client Gemini non initialise"

    try:
        from google.genai import types
    except ImportError:
        return None, "SDK_ERROR", "Package google-genai manquant"

    result = {"response": None, "error": None, "done": False}

    def _worker():
        try:
            resp = client.models.generate_content(
                model=model_to_try,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=max_tokens,
                    top_p=0.9,
                ),
            )
            result["response"] = resp
        except Exception as exc:
            result["error"] = exc
        finally:
            result["done"] = True

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)

    if not result["done"]:
        return None, "TIMEOUT", f"Aucune reponse de '{model_to_try}' apres {timeout_seconds:.0f}s"

    if result["error"] is not None:
        e = result["error"]
        return None, "SDK_ERROR", f"{type(e).__name__}: {e}"

    return result["response"], None, None


def _extract_text_from_response(response) -> str:
    """Safely extracts text from a Gemini response."""
    if response is None:
        return ""

    if hasattr(response, "candidates") and response.candidates:
        cand = response.candidates[0]
        if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts"):
            text_parts = []
            for part in cand.content.parts:
                if getattr(part, "thought", False):
                    continue
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            combined = "".join(text_parts).strip()
            if combined:
                return combined

    if hasattr(response, "text") and response.text:
        text = response.text.strip()
        if text:
            return text

    return ""


def ask_llm(prompt: str, timeout: int = 180, max_retries: int = 8,
            inter_request_delay: float = INTER_REQUEST_DELAY) -> str:
    """
    Queries Gemini with comprehensive error handling and model rotation.
    Waits briefly if background init is still running.
    """
    ensure_init_started()
    status = _get_status()

    if status == LLMStatus.UNAVAILABLE:
        err = get_initialization_error()
        return f"[ERREUR LLM] LLM non disponible. {err}"

    # Wait for background init if still running (max 30 seconds)
    if status in (LLMStatus.NOT_STARTED, LLMStatus.INITIALIZING):
        waited = 0.0
        while _get_status() == LLMStatus.INITIALIZING and waited < 30.0:
            time.sleep(1.0)
            waited += 1.0

        if _get_status() != LLMStatus.AVAILABLE:
            return (
                "[ERREUR LLM] LLM en cours d'initialisation ou indisponible. "
                "Veuillez patienter et reessayer."
            )

    if inter_request_delay > 0:
        time.sleep(inter_request_delay)

    with _state_lock:
        last_ok = _state["last_successful_model"]

    active = _get_active_model_internal()
    all_models = [active] + [m for m in GOOGLE_MODEL_FALLBACKS if m != active]
    if last_ok and last_ok in all_models:
        candidates = [last_ok] + [m for m in all_models if m != last_ok]
    else:
        candidates = all_models

    tried_models = set()
    last_error = None
    empty_response_count = 0
    current_max_tokens = 8000

    for attempt in range(1, max_retries + 1):
        model_to_try = next(
            (c for c in candidates if c not in tried_models),
            candidates[0]
        )

        response, error_type, error_msg = _call_with_timeout(
            model_to_try=model_to_try,
            prompt=prompt,
            max_tokens=current_max_tokens,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
        )

        if error_type == "TIMEOUT":
            last_error = error_msg
            tried_models.add(model_to_try)
            if attempt < max_retries:
                time.sleep(2 + random.uniform(0, 2))
            continue

        if error_type == "SDK_ERROR":
            last_error = error_msg
            error_str = error_msg.upper()
            is_transient = any(
                code in error_str
                for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                             "404", "NOT_FOUND", "DEADLINE_EXCEEDED"]
            )
            tried_models.add(model_to_try)

            if attempt < max_retries:
                wait = 1 + random.uniform(0, 3) if is_transient else min(30, 2 ** attempt)
                time.sleep(wait)
            continue

        # Response received
        extracted = _extract_text_from_response(response)

        if extracted:
            with _state_lock:
                _state["last_successful_model"] = model_to_try
            return extracted

        # Empty response
        empty_response_count += 1
        last_error = f"Reponse vide de '{model_to_try}'"
        tried_models.add(model_to_try)

        if empty_response_count >= 3:
            return (
                f"[ERREUR LLM] Le modele a renvoye {empty_response_count} reponses vides "
                "consecutives. Simplifiez votre requete."
            )

        if attempt < max_retries:
            time.sleep(2 + random.uniform(0, 2))

    return f"[ERREUR LLM] Echec definitif apres {max_retries} tentatives. Detail : {last_error}"


# -----------------------------------------------------------------------------
# C. INDEPENDENT NUMERICAL VERIFIER
# -----------------------------------------------------------------------------

NUMBER_REGEX = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_numbers(text: str) -> list:
    results = []
    for match in NUMBER_REGEX.findall(text):
        normalized = match.replace(",", ".")
        try:
            val = float(normalized)
            results.append((val, match))
        except ValueError:
            continue
    return results


def _collect_allowed_numbers(case_data: dict) -> set:
    allowed = set()

    def add(v):
        try:
            allowed.add(round(float(v), 6))
        except Exception:
            pass

    for key in ["N", "K", "seed"]:
        if key in case_data:
            add(case_data[key])

    cid = str(case_data.get("case_id", "")).replace("#", "")
    if cid.isdigit():
        add(int(cid))

    for num_str in re.findall(r"\d+", str(case_data.get("scenario", ""))):
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

    for base_data in case_data.get("baselines", {}).values():
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

    for alloc_list in case_data.get("allocations", {}).values():
        for c in alloc_list:
            add(c)

    for entry in case_data.get("cell_summary", []):
        add(entry.get("cell", 0))
        add(entry.get("degree", 0))
        add(entry.get("strength", 0))

    return allowed


def verify_numbers(llm_response: str, case_data: dict, tolerance: float = 0.01) -> dict:
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

Nous te fournissons a nouveau la REQUETE ORIGINALE COMPLETE. Corrige ta reponse precedente.

--- DEBUT DE LA REQUETE ORIGINALE ---
{original_prompt}
--- FIN DE LA REQUETE ORIGINALE ---

Chiffres erronees detectes :
{faulty_str}

Donnees certifiees :
{allowed_summary}

Consignes :
A) Reresous les taches A) a F) integralement.
B) Utilise des majuscules A), B)... pour la structure.
C) Ecris les donnees de simulation en chiffres et epelle tout le reste.
"""
    return ask_llm(correction_prompt, inter_request_delay=INTER_REQUEST_DELAY)


# -----------------------------------------------------------------------------
# D. PDF REPORT GENERATOR
# -----------------------------------------------------------------------------

def generate_pdf_report(
    case_data: dict, prompt: str, raw_response: str,
    verification_initial: dict, corrected_response: str,
    verification_final: dict, output_path: Path, model_name: str = None,
) -> None:
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
    style_body = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"],
        fontSize=9.5, leading=13, alignment=TA_LEFT
    )

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
    )

    story = []
    model_label = case_data.get("model_label", "CCI-only")
    conflicts_label = _conflicts_label(model_label)

    story.append(Paragraph(
        f"Rapport d'audit LLM - Attribution de canaux ({model_label})", style_title
    ))
    story.append(Spacer(1, 0.3*cm))

    header_data = [
        ["Cas", case_data.get("case_id", "-")],
        ["Scenario", case_data.get("scenario", "-")],
        ["Configuration", f"N={case_data.get('N')}, K={case_data.get('K')}, seed={case_data.get('seed')}"],
        ["Mode", model_label],
        ["Modele", model_name or "Non specifie"],
        ["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    header_table = Table(header_data, colWidths=[4*cm, 12*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eef7")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    if verification_final["has_hallucination"]:
        badge = f'<font color="#b30000"><b>NON CERTIFIE</b></font> - {len(verification_final["invented_numbers"])} valeur(s) suspecte(s).'
    else:
        badge = f'<font color="#006600"><b>CERTIFIE</b></font> - exactitude : {verification_final["accuracy_rate"]:.1f}%.'
    story.append(Paragraph(badge, style_body))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("1. Donnees de simulation", style_h1))
    m = case_data.get("metrics", {})
    sim_data = [
        ["Metrique", "Valeur"],
        ["Cout initial", _fmt(m.get("cost_initial"))],
        ["Cout final", _fmt(m.get("cost_final"))],
        [conflicts_label, _fmt(m.get("conflicts"))],
        ["Temps (s)", _fmt(m.get("time_seconds"), 6)],
        ["Iterations", _fmt(m.get("iterations"))],
        ["Canaux", _fmt(m.get("used_channels"))],
    ]
    sim_table = Table(sim_data, colWidths=[7*cm, 5*cm])
    sim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(sim_table)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2. Reponse brute", style_h1))
    raw_esc = raw_response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(raw_esc.replace("\n", "<br/>"), style_body))

    doc.build(story)


# -----------------------------------------------------------------------------
# E. MASTER AUDIT PIPELINE
# -----------------------------------------------------------------------------

def audit_case(case_data: dict, max_correction_attempts: int = 2,
               model_folder: str = "cochannel") -> dict:
    case_id = case_data.get("case_id", "unknown")
    model_label = case_data.get("model_label", "CCI-only")

    raw_dir, reports_dir = _get_mode_folders(model_label)
    prompt = build_prompt(case_data)

    with open(raw_dir / f"{case_id}_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    raw_response = ask_llm(prompt)

    if raw_response.startswith("[ERREUR LLM]"):
        with open(raw_dir / f"{case_id}_error.txt", "w", encoding="utf-8") as f:
            f.write(raw_response)
        empty_verif = {
            "all_numbers": [], "allowed_numbers": set(),
            "valid_numbers": [], "invented_numbers": [],
            "has_hallucination": False, "accuracy_rate": 0.0,
        }
        return {
            "case_id": case_id, "scenario": case_data.get("scenario"),
            "N": case_data.get("N"), "K": case_data.get("K"),
            "seed": case_data.get("seed"), "context": case_data.get("context"),
            "model_label": model_label,
            "raw_response": raw_response, "corrected_response": raw_response,
            "verification_initial": empty_verif, "verification_final": empty_verif,
            "correction_attempts": 0, "pdf_path": None, "status": "LLM_FAILED",
        }

    with open(raw_dir / f"{case_id}_raw_response.txt", "w", encoding="utf-8") as f:
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
        with open(raw_dir / f"{case_id}_corrected_v{correction_attempts}.txt", "w", encoding="utf-8") as f:
            f.write(corrected_response)

    pdf_path = reports_dir / f"rapport_{case_id}_{case_data.get('scenario')}.pdf"
    generate_pdf_report(
        case_data=case_data, prompt=prompt,
        raw_response=raw_response, verification_initial=verification_initial,
        corrected_response=corrected_response, verification_final=verification_final,
        output_path=pdf_path, model_name=get_active_model(),
    )

    return {
        "case_id": case_id, "scenario": case_data.get("scenario"),
        "N": case_data.get("N"), "K": case_data.get("K"),
        "seed": case_data.get("seed"), "context": case_data.get("context"),
        "model_label": model_label,
        "raw_response": raw_response, "corrected_response": corrected_response,
        "verification_initial": verification_initial,
        "verification_final": verification_final,
        "correction_attempts": correction_attempts,
        "pdf_path": str(pdf_path), "status": "OK",
    }