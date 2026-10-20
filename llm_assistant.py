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

"""

import os
import re
import json
import time
import random
import threading
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import numpy as np

# Supprime les avertissements cosmétiques de l'API Google GenAI
warnings.filterwarnings("ignore", message=".*automatic function calling.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*AFC.*", category=UserWarning)

# --- 1. CONFIGURATION ET CHARGEMENT DU FICHIER ENVIRONNEMENTAL ---
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip().strip('"').strip("'")

# Modèle unique recommandé par l'encadreur : aucune substitution, ni en dashboard ni en E10
GOOGLE_MODEL_PREFERRED = "models/gemini-3.6-flash"
GOOGLE_MODEL_FALLBACKS = []

INTER_REQUEST_DELAY = 3.0
LLM_TIMEOUT_SECONDS = 45
PROBE_TIMEOUT_SECONDS = 20
# Sonde répétée : un 429 ou 503 passager ne doit pas déclarer le modèle indisponible
PROBE_ATTEMPTS = 3
PROBE_RETRY_DELAY_SECONDS = 4.0

# Paramètres d'inférence gelés pour la campagne scientifique (recommandation E10)
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 1.0
LLM_TOP_K = 1
LLM_MAX_OUTPUT_TOKENS = 8000
LLM_SEED = 2026
# Dashboard et E10 interrogent le même modèle unique, sans rotation
LLM_CAMPAIGN_FIXED_MODEL = True
LLM_CAMPAIGN_MODEL = GOOGLE_MODEL_PREFERRED
# Version de la directive, archivée hors prompt pour ne pas polluer la whitelist
ZERO_HALLUCINATION_DIRECTIVE_VERSION = "ZH-1.3"
# Typologie des échecs d'appel, reportée telle quelle dans le CSV E10
LLM_ERROR_CODES = ("QUOTA_DAILY", "RATE_LIMIT", "UNAVAILABLE", "TIMEOUT",
                   "EMPTY_RESPONSE", "TRUNCATED", "OTHER")
# Quota gratuit Gemini réinitialisé à minuit, heure du Pacifique
QUOTA_RESET_TIMEZONE = "America/Los_Angeles"
# Résumé JSON d'un cas E10, relu lors d'une reprise de campagne
AUDIT_RECORD_SUFFIX = "_audit.json"

from config import LLM_LOGS_DIR, FIGURES_DIR, CSV_DIR
from verifier import extract_metric_claims, check_channel_claims

LLM_LOGS_DIR = Path(LLM_LOGS_DIR)
LLM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
LLM_REPORTS_DIR = LLM_LOGS_DIR / "reports_pdf"
LLM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LLM_RAW_DIR = LLM_LOGS_DIR / "raw_responses"
LLM_RAW_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. CONSOLE D'INITIALISATION SÉCURISÉE (LOCK ET THREAD DAEMON)
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


def _probe_single_model(client, model_name: str, types_module) -> bool:
    """
    Envoie un ping réseau léger pour vérifier l'accès au modèle.
    """
    result = {"ok": False}

    def _ping():
        try:
            # Lecture des métadonnées du modèle : vérifie l'accès sans consommer le quota de génération
            client.models.get(model=model_name)
            result["ok"] = True
        except Exception:
            result["ok"] = False

    t = threading.Thread(target=_ping, daemon=True)
    t.start()
    t.join(timeout=PROBE_TIMEOUT_SECONDS)

    return result["ok"]


def _background_init():
    """
    Moteur de validation de connectivité s'exécutant de manière asynchrone.
    """
    _set_status(LLMStatus.INITIALIZING)
    _set_progress("Chargement du package google-genai...")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        _set_status(
            LLMStatus.UNAVAILABLE,
            error="Le package 'google-genai' est requis pour ce module."
        )
        return

    if not GOOGLE_API_KEY:
        _set_status(
            LLMStatus.UNAVAILABLE,
            error="La variable GOOGLE_API_KEY est manquante dans .env"
        )
        return

    _set_progress("Instanciation de l'API...")

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        _set_client(client)
    except Exception as e:
        _set_status(
            LLMStatus.UNAVAILABLE,
            error=f"Erreur d'initialisation du client : {e}"
        )
        return

    candidates = [GOOGLE_MODEL_PREFERRED] + [
        m for m in GOOGLE_MODEL_FALLBACKS if m != GOOGLE_MODEL_PREFERRED
    ]

    errors_collected = []

    for i, model_name in enumerate(candidates, start=1):
        display_name = model_name.replace("models/", "")
        _set_progress(f"Vérification {i}/{len(candidates)} : {display_name}")

        try:
            probe_ok = False
            for probe_attempt in range(PROBE_ATTEMPTS):
                if _probe_single_model(client, model_name, types):
                    probe_ok = True
                    break
                time.sleep(PROBE_RETRY_DELAY_SECONDS * (probe_attempt + 1))
            if probe_ok:
                with _state_lock:
                    _state["last_successful_model"] = model_name
                _set_status(LLMStatus.AVAILABLE, model=model_name)
                _set_progress("")
                return
            else:
                errors_collected.append(f"{display_name} : hors-ligne")
        except Exception as e:
            errors_collected.append(f"{display_name} : {type(e).__name__}")

    _set_status(
        LLMStatus.UNAVAILABLE,
        error="Tous les pings API ont échoué : " + "; ".join(errors_collected[:3])
    )
    _set_progress("")


def ensure_init_started():
    """Démarre le thread de contrôle en arrière-plan sans bloquer l'UI."""
    with _init_lock:
        with _state_lock:
            if _state["init_thread_started"]:
                return
            _state["init_thread_started"] = True

    thread = threading.Thread(
        target=_background_init,
        daemon=True,
        name="llm-background-init-sequence",
    )
    thread.start()


def is_llm_available() -> bool:
    ensure_init_started()
    return _get_status() == LLMStatus.AVAILABLE


def is_llm_initializing() -> bool:
    ensure_init_started()
    return _get_status() == LLMStatus.INITIALIZING


def get_active_model() -> str:
    return _get_active_model_internal()


def get_last_responding_model() -> str:
    """Identifiant exact du dernier modèle ayant effectivement répondu."""
    with _state_lock:
        return (_state["last_successful_model"] or "").replace("models/", "")


def get_llm_inference_params() -> dict:
    """Paramètres d'inférence exacts, affichés dans le dashboard et les rapports."""
    model = (get_last_responding_model()
             or _get_active_model_internal().replace("models/", "")
             or GOOGLE_MODEL_PREFERRED.replace("models/", ""))
    return {
        "model": model,
        "campaign_model": LLM_CAMPAIGN_MODEL.replace("models/", ""),
        "fixed_model_for_campaign": LLM_CAMPAIGN_FIXED_MODEL,
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "top_k": LLM_TOP_K,
        "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
        "seed": LLM_SEED,
        "directive_version": ZERO_HALLUCINATION_DIRECTIVE_VERSION,
        "semantic_audit_version": SEMANTIC_AUDIT_VERSION,
    }


def get_initialization_error() -> str:
    with _state_lock:
        return _state["error_message"] or ""


def get_llm_status_label() -> str:
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


def _fmt_us(seconds) -> str:
    """Durée stockée en secondes, restituée en millisecondes (4 décimales)."""
    try:
        return f"{float(seconds) * 1e3:.4f}"
    except (TypeError, ValueError):
        return str(seconds)


def _conflicts_summary_text(m: dict, model_label: str) -> str:
    """
    [FIX-LLM-CCI-ACI] Retourne un texte synthetique des conflits adapte au regime.
    En mode adjacent, expose C_CCI, C_ACI et C_tot separement.
    En mode co-canal, n'expose que C_CCI (C_ACI nul par construction).
    """
    if model_label == "CCI+ACI":
        return (
            f"C_CCI = {int(m.get('conflicts_cci', 0))}, "
            f"C_ACI = {int(m.get('conflicts_aci', 0))}, "
            f"C_tot = {int(m.get('conflicts', 0))}"
        )
    return f"C_CCI = {int(m.get('conflicts_cci', 0))}"

# -----------------------------------------------------------------------------
# A. SYSTEM DIRECTIVES AND PROMPT SCHEMAS (Chantiers A + D)
# -----------------------------------------------------------------------------

ZERO_HALLUCINATION_DIRECTIVE = """
=== DIRECTIVE SYSTEME (ZERO-HALLUCINATION) ===
Tu es un ingenieur radio senior. Ton role est d'analyser et d'auditer
les resultats de l'optimisation BD-CeNN appliquee a l'attribution de canaux.

RÈGLES DE CONDUITE MATHEMATIQUE :
- Tu ne dois sous aucun pretexte modifier, inventer ou deviner un chiffre.
- Tout chiffre cite reprend exactement une valeur du bloc de donnees, avec la meme
  precision et la meme unite (temps en ms, quatre decimales, recopies tels quels).
- N'effectue aucun calcul derive (difference, ratio, pourcentage, moyenne, ecart) :
  un tel chiffre est absent des donnees. Exprime les comparaisons en mots
  ("plus faible", "nettement plus rapide").
- Si une variable n'est pas explicitee dans le bloc de donnees, declare-la non disponible.
- Ne cite jamais l'identifiant du run.
- Redige un rapport technique sobre, sans qualificatif emphatique ni formule creuse
  ("il convient de", "il est important de noter", "en conclusion").
- Rapporte uniquement les resultats obtenus : ne t'attribue aucune note, aucun
  score de confiance et aucune auto-evaluation de ton analyse.

=== FORMAT DE SORTIE (OBLIGATOIRE) ===
- Exactement cinq rubriques, dans l'ordre A), B), C), D), E), chacune commencant
  en debut de ligne par sa lettre suivie d'une parenthese fermante.
- N'emploie jamais de listes numerotees qui fausseraient le parsing automatique.
- Entre quatre-vingts et cent cinquante mots par rubrique ; la reponse complete reste
  sous sept cent cinquante mots et se termine par la rubrique E.
- Ne recopie ni la liste des aretes, ni les vecteurs d'affectation, ni les positions ;
  cite au plus cinq cellules a titre d'exemple.

=== ECRITURE DES NOMBRES (TRES IMPORTANT) ===
Un parseur regex externe audite tous les chiffres de ta reponse.

  REGLE 1 - TOUTE VALEUR ISSUE DES DONNEES : EN CHIFFRES.
  Couts J(x), conflits C_CCI, C_ACI et C_tot, temps d'execution, sweeps, restarts,
  canaux utilises, numeros de cellules, nombres de voisins, forces cumulees et poids W_ij
  s'ecrivent en chiffres, exactement comme dans les donnees. Exemple : "C_CCI = <valeur>",
  jamais la valeur ecrite en lettres.

  REGLE 2 - COMPTEURS DE DISCOURS : EN TOUTES LETTRES.
  Seuls les denombrements absents des donnees s'ecrivent en lettres :
  "trois algorithmes de reference", "deux regimes d'interference", "cinq rubriques".
  La regle 1 prime toujours sur la regle 2.
"""


def build_prompt(case_data: dict) -> str:
    """Construit le prompt de simulation."""
    m = case_data.get("metrics", {})
    b = case_data.get("baselines", {})
    model_label = case_data.get("model_label", "CCI-only")

    # Construction du bloc baselines (Chantier A)
    baseline_lines = []
    for name in ["Random", "Greedy", "DSATUR"]:
        if name in b:
            # En mode adjacent, on fournit conflicts_cci et conflicts_aci distinctement
            if model_label == "CCI+ACI":
                conflict_text = (
                    f"conflits co-canal C_CCI = {int(b[name].get('conflicts_cci', 0))}, "
                    f"conflits adjacents C_ACI = {int(b[name].get('conflicts_aci', 0))}, "
                    f"conflits totaux C_tot = {int(b[name].get('conflicts_total', 0))}"
                )
            else:
                conflict_text = f"conflits co-canal C_CCI = {int(b[name].get('conflicts_cci', 0))}"

            baseline_lines.append(
                f"  - {name} : cout J = {_fmt(b[name].get('cost'))}, "
                f"{conflict_text}, "
                f"temps t_exec = {_fmt_us(b[name].get('time'))} ms"
            )
    baseline_str = "\n".join(baseline_lines) if baseline_lines else "  (aucune baseline)"

    conflicting_cells = case_data.get("conflicting_cells", [])
    conflicting_str = (
        ", ".join(str(c) for c in conflicting_cells)
        if conflicting_cells else "aucune"
    )

    # Note sur le régime et distinctions des conflits (Chantier A)
    if model_label == "CCI+ACI":
        mode_note = (
            "Le regime d'interference considere est le modele etendu CCI+ACI.\n"
            "Les conflits sont categorises sous deux compteurs mutuellement exclusifs :\n"
            f"  - C_CCI (co-canal stricts, même frequence) : {int(m.get('conflicts_cci', 0))}\n"
            f"  - C_ACI (canal adjacent strict, distance <= 2) : {int(m.get('conflicts_aci', 0))}\n"
            f"  - C_total (somme des deux compteurs) : {int(m.get('conflicts', 0))}"
        )
    else:
        mode_note = (
            "Le regime d'interference considere est le modele co-canal seul CCI-only.\n"
            "Aucun debordement spectral n'est evalue.\n"
            f"  - C_CCI (conflits co-canal stricts) : {int(m.get('conflicts_cci', 0))}\n"
            "  - C_ACI (conflits adjacents) : 0 (par definition)"
        )

    topology_meta = case_data.get("topology_meta", {})
    if topology_meta:
        meta_str = (
            f"  - Graine de topologie (seed) : {topology_meta.get('seed', 'N/A')}\n"
            f"  - Seuil d'interference radio (threshold) : {topology_meta.get('threshold', 'N/A')}\n"
            f"  - Nombre de cellules N : {topology_meta.get('N', 'N/A')}\n"
            f"  - Nombre de canaux K : {topology_meta.get('K', 'N/A')}"
        )
    else:
        meta_str = "  (metadonnees indisponibles)"

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
        pos_lines.append(f"  ... ({len(cell_positions) - 20} autres cellules)")
        positions_str = "\n".join(pos_lines)
    else:
        positions_str = "  (positions indisponibles)"

    topology_edges = case_data.get("topology_edges", [])
    total_edges = case_data.get("total_edges", 0)
    if topology_edges:
        topology_lines = [
            f"  - Cellule {i} <-> Cellule {j} : poids W_ij = {w}"
            for (i, j, w) in topology_edges
        ]
        topology_body = "\n".join(topology_lines)
        if total_edges > len(topology_edges):
            topology_header = (
                f"({len(topology_edges)} aretes affichees sur {total_edges} au total)"
            )
        else:
            topology_header = f"({total_edges} aretes au total)"
        topology_str = f"{topology_header}\n{topology_body}"
    else:
        topology_str = "  (topologie indisponible)"

    cell_summary = case_data.get("cell_summary", [])
    if cell_summary:
        cs_lines = [
            f"  - Cellule {e['cell']:3d} : {e['degree']:2d} voisins, force cumulee = {e['strength']}"
            for e in cell_summary
        ]
        cell_summary_str = "\n".join(cs_lines)
    else:
        cell_summary_str = "  (resume indisponible)"

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
        allocations_str = "  (affectations indisponibles)"
        allocations_header = ""

    prompt = f"""
{ZERO_HALLUCINATION_DIRECTIVE}

=== DONNEES DE SIMULATION EXPERIMENTALE ===
- Identifiant du run : {case_data.get('case_id')}
- Scenario : {case_data.get('scenario')}
- Taille du reseau : N = {case_data.get('N')} cellules, K = {case_data.get('K')} canaux
- Regime spectral : {model_label}
- Description contextuelle : {case_data.get('context')}

=== REGIME SPECTRAL ET CONFLITS (Eq. 4, 5, 6) ===
{mode_note}

=== METRIQUES DE CONVERGENCE (BD-CeNN) ===
- Cout initial de l'energie J_0 : {_fmt(m.get('cost_initial'))}
- Cout final J(x*) : {_fmt(m.get('cost_final'))}
- Temps d'execution du solveur t_exec : {_fmt_us(m.get('time_seconds'))} ms
- Nombre de balayages (sweeps) effectues : {_fmt(m.get('sweeps'))}
- Nombre de canaux utilises : {_fmt(m.get('used_channels'))}
- Nombre de restarts du solveur : {_fmt(m.get('num_restarts'))}

=== ANALYSE COMPARATIVE (BASELINES) ===
{baseline_str}

=== TOPOLOGIE DU RESEAU CELLULAIRE ===
{topology_str}

=== CELLULES LES PLUS CONTRAINTES (TOP 10) ===
{cell_summary_str}

=== VECTEURS D'AFFECTATION ===
{allocations_header}
{allocations_str}

=== CELLULES EN CONFLIT APRES OPTIMISATION ===
{conflicting_str}

=== TACHES D'AUDIT DEMANDEES (Enumerer par A), B), C), D), E)) ===
A) Compare l'intensite du cout final J(x*) du BD-CeNN face aux trois baselines.
   Analyse le vecteur d'allocation de canal pour expliquer les ecarts de performance.
   Discute specifiquement de la repartition spectrale obtenue.

B) Evalue la dynamique interne du solveur en citant le nombre de sweeps (balayages)
   necessaires pour converger vers la solution stable.

C) Confronte le temps de calcul (t_exec) de chaque methode. Concede honnetement la primaute
   chronometrique aux heuristiques gloutonnes si tel est le cas.

D) Evalue les contraintes topologiques du reseau geres par le solveur. Identifie les cellules
   les plus saturees de voisinages et discute de la capacite du BD-CeNN a resoudre les
   liens conflictuels de forte valeur (poids W_ij = 4).

E) Formule des recommandations pour eviter les pieges des minima locaux (modification stochastique,
   augmentation des restarts).
"""
    return prompt.strip()


# -----------------------------------------------------------------------------
# B. API INTERACTION ENGINE
# -----------------------------------------------------------------------------

def _call_with_timeout(model_to_try: str, prompt: str,
                       max_tokens: int, timeout_seconds: float):
    client = _get_client()
    if client is None:
        return None, "SDK_ERROR", "Client non initialise"

    try:
        from google.genai import types
    except ImportError:
        return None, "SDK_ERROR", "Module google-genai absent"

    result = {"response": None, "error": None, "done": False}

    def _worker():
        try:
            resp = client.models.generate_content(
                model=model_to_try,
                contents=prompt,
                # Inférence déterministe et paramètres archivés (E10)
                config=types.GenerateContentConfig(
                    temperature=LLM_TEMPERATURE,
                    max_output_tokens=max_tokens,
                    top_p=LLM_TOP_P,
                    top_k=LLM_TOP_K,
                    seed=LLM_SEED,
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
        return None, "TIMEOUT", f"Le modele '{model_to_try}' n'a pas repondu."

    if result["error"] is not None:
        e = result["error"]
        return None, "SDK_ERROR", f"{type(e).__name__}: {e}"

    return result["response"], None, None


def _extract_text_from_response(response) -> str:
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


def _classify_sdk_error(error_msg: str) -> str:
    """Code d'échec normalisé à partir du message d'erreur du SDK."""
    upper = (error_msg or "").upper()
    if "PERDAY" in upper or "PER_DAY" in upper:
        return "QUOTA_DAILY"
    if "429" in upper or "RESOURCE_EXHAUSTED" in upper:
        return "RATE_LIMIT"
    if "503" in upper or "UNAVAILABLE" in upper or "OVERLOADED" in upper:
        return "UNAVAILABLE"
    if "DEADLINE_EXCEEDED" in upper or "504" in upper:
        return "TIMEOUT"
    return "OTHER"


def _parse_retry_seconds(error_msg: str):
    """Délai de relance annoncé par l'API (« retry in 1h2m3.5s » ou retryDelay), en secondes."""
    match = re.search(r"retry\s+in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s",
                      error_msg or "", re.IGNORECASE)
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", error_msg or "")
    return float(match.group(1)) if match else None


def seconds_until_quota_reset(now: datetime = None):
    """Secondes avant le prochain minuit du Pacifique ; None si le fuseau est indisponible."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(QUOTA_RESET_TIMEZONE)
    except Exception:
        # Windows sans le paquet tzdata
        return None
    now = now.astimezone(tz) if now else datetime.now(tz)
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (next_midnight - now).total_seconds()


def format_duration(seconds) -> str:
    if seconds is None:
        return "duree inconnue"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def _extract_finish_reason(response):
    """finish_reason du premier candidat (STOP, MAX_TOKENS, SAFETY...), ou None."""
    try:
        reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None
    if reason is None:
        return None
    return str(getattr(reason, "name", reason)).split(".")[-1].upper()


def ask_llm_detailed(prompt: str, timeout: int = 180, max_retries: int = 8,
                     inter_request_delay: float = INTER_REQUEST_DELAY,
                     allow_fallback: bool = True) -> dict:
    """
    Appel LLM avec typologie explicite des échecs (LLM_ERROR_CODES). Le texte
    n'est renvoyé qu'en cas de succès ; finish_reason est conservé pour que
    l'appelant détecte la troncature.
    """
    result = {"text": "", "error_code": None, "error_message": "",
              "finish_reason": None, "retry_after_s": None,
              "quota_reset_s": None, "model": ""}

    def _fail(code, message, **extra):
        result.update(error_code=code, error_message=message, **extra)
        return result

    ensure_init_started()
    status = _get_status()

    if status == LLMStatus.UNAVAILABLE:
        return _fail("OTHER", f"Assistant indisponible. {get_initialization_error()}")

    if status in (LLMStatus.NOT_STARTED, LLMStatus.INITIALIZING):
        waited = 0.0
        while _get_status() == LLMStatus.INITIALIZING and waited < 30.0:
            time.sleep(1.0)
            waited += 1.0
        if _get_status() != LLMStatus.AVAILABLE:
            return _fail("OTHER", "Connexion en cours. Reessayez.")

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
    if not allow_fallback:
        # Modèle unique imposé (campagne E10) : aucune substitution silencieuse
        candidates = [LLM_CAMPAIGN_MODEL]

    tried_models = set()
    last_code, last_error = "OTHER", None
    empty_response_count = 0

    for attempt in range(1, max_retries + 1):
        model_to_try = next((c for c in candidates if c not in tried_models), candidates[0])
        result["model"] = model_to_try.replace("models/", "")

        response, error_type, error_msg = _call_with_timeout(
            model_to_try=model_to_try, prompt=prompt,
            max_tokens=LLM_MAX_OUTPUT_TOKENS, timeout_seconds=LLM_TIMEOUT_SECONDS,
        )

        if error_type in ("TIMEOUT", "SDK_ERROR"):
            last_error = error_msg
            last_code = ("TIMEOUT" if error_type == "TIMEOUT"
                         else _classify_sdk_error(error_msg))
            retry_after = _parse_retry_seconds(error_msg)
            # Quota journalier : toute relance échouerait jusqu'à la réinitialisation
            if last_code == "QUOTA_DAILY":
                reset_s = seconds_until_quota_reset()
                return _fail(
                    "QUOTA_DAILY",
                    f"Quota journalier gratuit du modele {result['model']} epuise. "
                    f"Reinitialisation estimee dans {format_duration(reset_s)}.",
                    retry_after_s=retry_after, quota_reset_s=reset_s)
            tried_models.add(model_to_try)
            if attempt < max_retries:
                if last_code == "RATE_LIMIT" and retry_after:
                    # Limite par minute : attente du délai annoncé plutôt qu'un tirage court
                    wait = min(retry_after + 1.0, 65.0)
                elif last_code in ("RATE_LIMIT", "UNAVAILABLE", "TIMEOUT"):
                    wait = 1 + random.uniform(0, 3)
                else:
                    wait = min(30, 2 ** attempt)
                time.sleep(wait)
            continue

        finish_reason = _extract_finish_reason(response)
        result["finish_reason"] = finish_reason
        extracted = _extract_text_from_response(response)
        if extracted:
            with _state_lock:
                _state["last_successful_model"] = model_to_try
            result["text"] = extracted
            return result

        empty_response_count += 1
        last_code = "EMPTY_RESPONSE"
        last_error = f"Reponse vide de '{result['model']}' (finish_reason = {finish_reason})"
        tried_models.add(model_to_try)
        # Budget consommé par le raisonnement interne : à température nulle, une
        # relance reproduirait le même échec et entamerait le quota journalier
        if finish_reason == "MAX_TOKENS" or empty_response_count >= 3:
            return _fail("EMPTY_RESPONSE", last_error)
        if attempt < max_retries:
            time.sleep(2 + random.uniform(0, 2))

    return _fail(last_code, f"Echec apres {max_retries} tentatives. Detail : {last_error}")


def ask_llm(prompt: str, timeout: int = 180, max_retries: int = 8,
            inter_request_delay: float = INTER_REQUEST_DELAY,
            allow_fallback: bool = True) -> str:
    """Interface texte historique : le code d'échec suit le préfixe [ERREUR LLM]."""
    result = ask_llm_detailed(prompt, timeout=timeout, max_retries=max_retries,
                              inter_request_delay=inter_request_delay,
                              allow_fallback=allow_fallback)
    if result["error_code"]:
        return f"[ERREUR LLM] {result['error_code']} : {result['error_message']}"
    return result["text"]


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

    # Fragments numériques de l'identifiant alphanumérique du run
    cid = str(case_data.get("case_id", "")).replace("#", "")
    for num_str in re.findall(r"\d+", cid):
        add(int(num_str))

    for num_str in re.findall(r"\d+", str(case_data.get("scenario", ""))):
        try:
            add(int(num_str))
        except ValueError:
            pass

    for n in range(1, 6):
        add(n)

    m = case_data.get("metrics", {})
    # [FIX-LLM-SWEEPS] Whitelist mise a jour : "iterations" remplace par
    # "sweeps", et ajout de "num_restarts" pour E10.
    for key in ["cost_initial", "cost_final", "conflicts", "conflicts_cci",
                "conflicts_aci", "time_seconds", "sweeps", "used_channels",
                "num_restarts"]:
        if key in m:
            add(m[key])
    # Durées admises aussi en millisecondes, unité d'affichage du prompt et du PDF
    if "time_seconds" in m:
        add(float(m["time_seconds"]) * 1e3)

    for base_data in case_data.get("baselines", {}).values():
        for k in ["cost", "conflicts_cci", "conflicts_aci", "conflicts_total", "time"]:
            if k in base_data:
                add(base_data[k])
        if "time" in base_data:
            add(float(base_data["time"]) * 1e3)

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
            # Poids bruité (S6) conservé à deux décimales, sans troncature
            add(w)
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

# -----------------------------------------------------------------------------
# C-bis. VERIFICATEUR STRUCTURE (scenario, methode, metrique, valeur)
# -----------------------------------------------------------------------------
# Le superviseur a demande de depasser la simple whitelist numerique pour
# verifier l'association complete (methode, metrique, valeur), afin de
# detecter les cas ou le LLM attribue a une methode un chiffre qui
# appartient en realite a une autre.

STRUCTURED_CLAIM_PATTERNS = [
    # "le coût du BD-CeNN vaut 19,30"
    (re.compile(
        r"(?:co[ûu]t|J\(x\)|[ée]nergie)\s+(?:du|de\s+la\s+m[ée]thode|de)\s+"
        r"(BD-CeNN|Greedy|DSATUR|Random)\s+"
        r"(?:vaut|est|atteint|s'[ée]l[èe]ve\s+[àa]|s'[ée]tablit\s+[àa])\s+"
        r"(-?\d+(?:[.,]\d+)?)",
        re.IGNORECASE),
     "cost"),

    # "C_CCI du BD-CeNN vaut 24,7"
    (re.compile(
        r"(C_CCI|C_ACI|C_tot|C_\{CCI\}|C_\{ACI\}|C_\{tot\})\s+"
        r"(?:du|de)\s+(BD-CeNN|Greedy|DSATUR|Random)\s+"
        r"(?:vaut|est|atteint|s'[ée]l[èe]ve\s+[àa])\s+"
        r"(-?\d+(?:[.,]\d+)?)",
        re.IGNORECASE),
     "conflict_metric"),

    # "temps d'exécution du BD-CeNN atteint 90,88"
    (re.compile(
        r"temps\s+d['\u2019]ex[ée]cution\s+(?:du|de)\s+"
        r"(BD-CeNN|Greedy|DSATUR|Random)\s+"
        r"(?:atteint|vaut|est)\s+"
        r"(-?\d+(?:[.,]\d+)?)",
        re.IGNORECASE),
     "time"),
]


# Mention explicite d'un scénario (S1 à S7) dans la phrase d'une revendication
_SCENARIO_TOKEN = re.compile(r"\bS([1-7])\b")
_SENTENCE_BREAK = re.compile(r"[.!?;](?=\s|$)|\n")


def _numbers_match(val, ref, rel_tol=0.01, abs_tol=0.01):
    if val == ref:
        return True
    if ref == 0:
        return abs(val) <= abs_tol
    return (abs(val - ref) / abs(ref) <= rel_tol) or (abs(val - ref) <= abs_tol)


def _build_structured_facts(case_data: dict) -> list:
    """Base de quadruplets certifiés (scénario, méthode, métrique, valeur)."""
    facts = []
    scenario = case_data.get("scenario", "")

    m = case_data.get("metrics", {})
    bd_metrics = [
        ("cost_initial", "cost_initial"),
        ("cost_final",   "cost_final"),
        # Une revendication « coût du BD-CeNN » désigne J_0 ou J(x*)
        ("cost",         "cost_final"),
        ("cost",         "cost_initial"),
        ("C_CCI",        "conflicts_cci"),
        ("C_ACI",        "conflicts_aci"),
        ("C_tot",        "conflicts"),
        ("time",         "time_seconds"),
        ("sweeps",       "sweeps"),
        ("used_channels", "used_channels"),
    ]
    for metric_label, key in bd_metrics:
        if key in m:
            facts.append({
                "scenario": scenario,
                "method":   "BD-CeNN",
                "metric":   metric_label,
                # Le temps est cité en ms dans le prompt
                "value":    float(m[key]) * 1e3 if metric_label == "time" else float(m[key]),
            })

    for base_name, base_data in case_data.get("baselines", {}).items():
        for metric_label, key in [
            ("cost",  "cost"),
            ("C_CCI", "conflicts_cci"),
            ("C_ACI", "conflicts_aci"),
            ("C_tot", "conflicts_total"),
            ("time",  "time"),
        ]:
            if key in base_data:
                facts.append({
                    "scenario": scenario,
                    "method":   base_name,
                    "metric":   metric_label,
                    "value":    (float(base_data[key]) * 1e3 if metric_label == "time"
                                 else float(base_data[key])),
                })

    return facts


def _claimed_scenario(text: str, position: int):
    """Scénario cité dans la phrase qui contient la revendication, sinon None."""
    starts = [m.end() for m in _SENTENCE_BREAK.finditer(text, 0, position)]
    start = starts[-1] if starts else 0
    end_match = _SENTENCE_BREAK.search(text, position)
    end = end_match.start() if end_match else len(text)
    found = _SCENARIO_TOKEN.findall(text[start:end])
    return f"S{found[0]}" if len(set(found)) == 1 else (None if not found else "AMBIGU")


def _extract_structured_claims(text: str) -> list:
    """Extrait les quadruplets (scénario, méthode, métrique, valeur) du texte LLM."""
    claims = []
    # Motifs élargis : méthode cherchée avant ou après la métrique (verifier.py)
    for claim in extract_metric_claims(text):
        claim["scenario"] = _claimed_scenario(text, claim["start"])
        claims.append(claim)
    return claims


def verify_structured_claims(text: str, case_data: dict,
                             rel_tol: float = 0.01,
                             abs_tol: float = 0.01) -> dict:
    """
    Vérifie chaque quadruplet (scénario, méthode, métrique, valeur). Un
    scénario non cité dans la phrase est celui du cas ; un scénario cité
    différent, ou plusieurs scénarios dans la même phrase, invalide la ligne.
    """
    facts = _build_structured_facts(case_data)
    claims = _extract_structured_claims(text)
    case_scenario = case_data.get("scenario", "")

    valid, invalid = [], []
    for claim in claims:
        claim["scenario_explicit"] = claim["scenario"] is not None
        if claim["scenario"] is None:
            claim["scenario"] = case_scenario
        matched = any(
            fact["scenario"] == claim["scenario"]
            and fact["method"].lower() == claim["method"].lower()
            and fact["metric"] == claim["metric"]
            and _numbers_match(claim["value"], fact["value"], rel_tol, abs_tol)
            for fact in facts
        )
        (valid if matched else invalid).append(claim)

    # Attribution « cellule i / canal c » fausse : quadruplet invalide
    channel_claims = check_channel_claims(text, case_data.get("allocations", {}), case_scenario)
    for claim in channel_claims:
        claim["scenario_explicit"] = False
        (valid if claim.pop("valid") else invalid).append(claim)

    total = len(claims) + len(channel_claims)
    return {
        "valid_claims":           valid,
        "invalid_claims":         invalid,
        "has_structured_error":   len(invalid) > 0,
        "total_claims":           total,
        "channel_claims_checked": len(channel_claims),
        "accuracy_rate_struct":   (len(valid) / total * 100.0) if total else 100.0,
    }


def numeric_certification(verification: dict, structured: dict = None,
                          truncated: bool = False) -> dict:
    """
    Badge de l'audit numérique : la whitelist ET les quadruplets doivent être
    conformes. Un rapport tronqué n'est jamais certifié (motif TRONQUE).
    """
    structured = structured or {}
    invalid = structured.get("invalid_claims", [])
    if truncated:
        status, reason = "NON_CERTIFIE", "TRONQUE"
    elif verification.get("has_hallucination") or invalid:
        status, reason = "NON_CERTIFIE", "VALEURS"
    elif not verification.get("all_numbers"):
        status, reason = "AVERTISSEMENT", "AUCUNE_VALEUR"
    else:
        status, reason = "CERTIFIE", ""
    return {
        "status": status,
        "reason": reason,
        # Libellé affiché (PDF, CSV) : le motif TRONQUE reste visible
        "label": f"{status} (TRONQUE)" if reason == "TRONQUE" else status,
        "certified": status == "CERTIFIE",
        "invented_count": len(verification.get("invented_numbers", [])),
        "checked_quadruplets": int(structured.get("total_claims", 0)),
        "invalid_quadruplets": len(invalid),
        "accuracy_rate": float(verification.get("accuracy_rate", 0.0)),
    }


# -----------------------------------------------------------------------------
# C-ter. AUDIT SÉMANTIQUE (séparé de l'audit numérique)
# -----------------------------------------------------------------------------
SEMANTIC_AUDIT_VERSION = "SEM-1.2"

_SECTION_LETTERS = ("A", "B", "C", "D", "E")
_NUMBERED_LIST = re.compile(r"(?m)^\s*\d+[.)]\s")
_OVERCLAIM = re.compile(
    r"optimum\s+global\s+(?:est\s+)?(?:garanti|certain)|fiable\s+[àa]\s+100"
    r"|garanti[t]?\s+l['’]optim|parfaitement\s+optimal",
    re.IGNORECASE)
# Primauté face à une baseline, confrontée ensuite au coût réel de cette baseline
_SUPERIORITY_CLAIM = re.compile(
    r"BD-CeNN(?P<span>[^.\n]{0,120}?)\b(?P<word>sup[ée]rieure?s?|meilleure?s?"
    r"|surpass(?:e|ent)|surclass(?:e|ent)|devanc(?:e|ent))\b"
    r"(?P<tail>[^.\n]{0,80}?)\b(?P<baseline>Random|Greedy|DSATUR)\b",
    re.IGNORECASE)
# « coût supérieur à DSATUR » décrit honnêtement un coût plus élevé
_COST_NOUN_BEFORE = re.compile(
    r"(?:co[ûu]t|J\s*\(|conflits?|temps|[ée]nergie|C_)[^.\n]{0,15}$", re.IGNORECASE)
_NEGATION = re.compile(r"\bne\b|\bn['’]|\bpas\b|\bjamais\b|\baucun", re.IGNORECASE)
_GUARANTEE_WORD = re.compile(
    r"\b(garantie?s?|garantit|garantissent|assurer?|assurent)\b", re.IGNORECASE)
_TERMINAL_PUNCTUATION = re.compile(r"[.!?…][\s\"'»)\]*_]*$")
# Le rapport restitue des résultats : toute note ou auto-évaluation est proscrite
_SELF_ASSESSMENT = re.compile(
    r"note\s+de\s+confiance|niveau\s+de\s+confiance|auto-?[ée]valuation"
    r"|\b(?:sur|/)\s*(?:5|cinq|10|dix)\b\s*[.)]?\s*$",
    re.IGNORECASE | re.MULTILINE)


def _section_pattern(letter: str):
    return re.compile(rf"(?m)^[\s#>*_-]*{letter})")


def _section_text(text: str, letter: str) -> str:
    """Contenu d'une rubrique, de sa lettre jusqu'à la rubrique suivante présente."""
    start = _section_pattern(letter).search(text)
    if not start:
        return ""
    following = [m.start() for s in _SECTION_LETTERS[_SECTION_LETTERS.index(letter) + 1:]
                 for m in [_section_pattern(s).search(text, start.end())] if m]
    return text[start.end():min(following) if following else len(text)]


def detect_truncation(text: str, finish_reason: str = None) -> dict:
    """
    Rapport tronqué si l'API signale MAX_TOKENS, si une rubrique A) à E) manque
    ou si la dernière phrase n'est pas terminée par une ponctuation finale.
    """
    reasons = []
    finish = (finish_reason or "").upper() or None
    if finish == "MAX_TOKENS":
        reasons.append("finish_reason = MAX_TOKENS")
    missing = [s for s in _SECTION_LETTERS if not _section_pattern(s).search(text or "")]
    if missing:
        reasons.append("rubrique(s) absente(s) : " + ", ".join(missing))
    if not _TERMINAL_PUNCTUATION.search((text or "").rstrip()):
        reasons.append("derniere phrase inachevee")
    return {"truncated": bool(reasons), "reasons": reasons,
            "missing_sections": missing, "finish_reason": finish}


def _false_superiority(text: str, cheaper: list) -> list:
    """BD-CeNN déclaré supérieur à une baseline dont le coût est strictement inférieur."""
    cheaper_upper = {name.upper() for name in cheaper}
    hits = []
    for match in _SUPERIORITY_CLAIM.finditer(text):
        if match.group("baseline").upper() not in cheaper_upper:
            continue
        if (match.group("word").lower().startswith("sup")
                and _COST_NOUN_BEFORE.search(match.group("span"))):
            continue
        if _NEGATION.search(match.group("span")[-20:] + " " + match.group("tail")[:12]):
            continue
        hits.append(f"{match.group('word')} / {match.group('baseline')}")
    return hits


def _unhedged_guarantees(text: str) -> list:
    """Garanties affirmées ; les formes niées (« ne garantit pas ») sont admises."""
    hits = []
    for match in _GUARANTEE_WORD.finditer(text):
        context = text[max(0, match.start() - 25):match.start()] + " " + \
            text[match.end():match.end() + 8]
        if not _NEGATION.search(context) and "sans" not in context.lower():
            hits.append(match.group(1))
    return hits


def semantic_audit(text: str, case_data: dict, truncated: bool = False) -> dict:
    """
    Audit sémantique déterministe : structure A) à E), style imposé, cohérence
    avec le régime spectral et honnêteté des comparaisons (règle R5). Sur un
    rapport tronqué, S4 à S6 sont NON EVALUABLES et sortent du score. Il
    n'intervient jamais dans le badge numérique.
    """
    checks = []

    def record(code, label, passed, detail="", evaluable=True):
        checks.append({"code": code, "label": label,
                       "passed": bool(passed) if evaluable else False,
                       "evaluable": evaluable, "detail": detail})

    missing = [s for s in _SECTION_LETTERS if not _section_pattern(s).search(text)]
    record("S1", "Rubriques A) à E) présentes", not missing, ", ".join(missing))
    record("S2", "Aucune liste numérotée", _NUMBERED_LIST.search(text) is None)

    if case_data.get("model_label", "CCI-only") == "CCI-only":
        aci_values = re.findall(
            r"C_\{?ACI\}?[^.\n]{0,30}?(?:=|vaut|est|atteint)\s*(\d+)", text)
        nonzero = [v for v in aci_values if int(v) > 0]
        record("S3", "Aucun conflit adjacent non nul en régime CCI-only",
               not nonzero, ", ".join(nonzero))

    metrics = case_data.get("metrics", {})
    baselines = case_data.get("baselines", {})
    bd_time = metrics.get("time_seconds")
    faster = [n for n, b in baselines.items()
              if bd_time is not None and b.get("time", np.inf) < bd_time]
    if faster:
        label_s4 = "Baseline plus rapide reconnue en rubrique C (R5)"
        if truncated:
            record("S4", label_s4, False, "rapport tronque", evaluable=False)
        else:
            section_c = _section_text(text, "C")
            record("S4", label_s4,
                   bool(section_c) and any(n in section_c for n in faster),
                   ", ".join(faster) if section_c else "rubrique C absente")

    bd_cost = metrics.get("cost_final")
    cheaper = [n for n, b in baselines.items()
               if bd_cost is not None and b.get("cost", np.inf) < bd_cost]
    if cheaper:
        label_s5 = "Baseline de coût inférieur reconnue (R5)"
        label_s6 = "Aucune primauté de coût revendiquée à tort"
        if truncated:
            record("S5", label_s5, False, "rapport tronque", evaluable=False)
            record("S6", label_s6, False, "rapport tronque", evaluable=False)
        else:
            record("S5", label_s5, all(n in text for n in cheaper), ", ".join(cheaper))
            lowest_cost_claim = re.search(
                r"BD-CeNN[^.\n]{0,80}(?:co[ûu]t|J\(x)[^.\n]{0,40}le\s+plus\s+(?:bas|faible)",
                text, re.IGNORECASE)
            violations = _false_superiority(text, cheaper)
            if lowest_cost_claim:
                violations.append("cout le plus bas")
            record("S6", label_s6, not violations, ", ".join(violations))

    overclaims = [m.group(0) for m in _OVERCLAIM.finditer(text)] + _unhedged_guarantees(text)
    record("S7", "Aucune sur-affirmation (optimum garanti, garantit, assure, fiabilité totale)",
           not overclaims, ", ".join(overclaims))

    self_rating = _SELF_ASSESSMENT.search(text)
    record("S8", "Aucune note ni auto-évaluation du modèle",
           self_rating is None, self_rating.group(0) if self_rating else "")

    evaluated = [c for c in checks if c["evaluable"]]
    passed = sum(c["passed"] for c in evaluated)
    score = 100.0 * passed / len(evaluated) if evaluated else 0.0
    if evaluated and passed == len(evaluated):
        status = "SEMANTIQUE_OK"
    elif score >= 75.0:
        status = "SEMANTIQUE_RESERVES"
    else:
        status = "SEMANTIQUE_ECHEC"
    return {"status": status, "score": score, "checks": checks,
            "version": SEMANTIC_AUDIT_VERSION, "truncated": truncated,
            "n_not_evaluable": len(checks) - len(evaluated)}

def regenerate_with_correction(case_data: dict, original_prompt: str,
                                llm_response: str, verification: dict,
                                structured_verification: dict = None) -> str:
    invalid_claims = (structured_verification or {}).get("invalid_claims", [])
    if not verification["has_hallucination"] and not invalid_claims:
        return llm_response

    faulty_lines = [
        f"  - valeur citee : {raw} (non presente dans les donnees)"
        for _, raw in verification["invented_numbers"]
    ]
    # Quadruplets mal attribués : valeur certifiée rattachée à la mauvaise méthode,
    # métrique ou au mauvais scénario
    faulty_lines += [
        f"  - association invalide : \"{c['raw']}\" "
        f"(scenario {c['scenario']}, methode {c['method']}, metrique {c['metric']})"
        for c in invalid_claims
    ]
    faulty_str = "\n".join(faulty_lines)

    m = case_data.get("metrics", {})
    model_label = case_data.get("model_label", "CCI-only")

    # [FIX-LLM-CCI-ACI] Decomposition explicite en mode adjacent
    allowed_summary = (
        f"Cout initial = {_fmt(m.get('cost_initial'))} ; "
        f"Cout final = {_fmt(m.get('cost_final'))} ; "
        f"{_conflicts_summary_text(m, model_label)} ; "
        f"Temps = {_fmt_us(m.get('time_seconds'))} ms ; "
        f"Balayages (sweeps) = {_fmt(m.get('sweeps'))} ; "
        f"Canaux utilises = {_fmt(m.get('used_channels'))}."
    )

    correction_prompt = f"""
=== DIRECTIVE DE CORRECTION SPECTRALE ===

Tu as cite des valeurs numeriques erronees dans ta reponse precedente.

--- PROMPT INITIAL ---
{original_prompt}
--- FIN PROMPT INITIAL ---

Erreurs numeriques detectees :
{faulty_str}

Donnees physiques valides :
{allowed_summary}

Redige a nouveau la reponse complete, rubriques A) a E), en remplacant chaque valeur
erronee par la valeur exacte des donnees ou par une formulation qualitative.
"""
    return ask_llm(correction_prompt, inter_request_delay=INTER_REQUEST_DELAY,
                   allow_fallback=not LLM_CAMPAIGN_FIXED_MODEL)


# -----------------------------------------------------------------------------
# D. PDF REPORT GENERATOR
# -----------------------------------------------------------------------------

def generate_pdf_report(
    case_data: dict, prompt: str, raw_response: str,
    verification_initial: dict, corrected_response: str,
    verification_final: dict, output_path: Path, model_name: str = None,
    verification_struct_initial: dict = None,
    verification_struct_final: dict = None,
    semantic_final: dict = None,
    truncation: dict = None,
) -> None:
    """
    [FIX-REPORT] Rapport PDF d'audit LLM enrichi :
      Section 1 : Donnees de simulation
      Section 2 : Prompt de simulation (integral)
      Section 3 : Rapport Assistant LLM (reponse brute)
      Section 4 : Audit numerique du verificateur LLM
      Section 5 : Rapport Assistant LLM corrige (si applicable)
    """
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
    # [FIX-REPORT] Style monospace pour le prompt integral
    style_mono = ParagraphStyle(
        "MonoCustom", parent=styles["BodyText"],
        fontName="Courier", fontSize=7.5, leading=9.5,
        alignment=TA_LEFT, textColor=colors.HexColor("#1e293b")
    )

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
    )

    story = []
    model_label = case_data.get("model_label", "CCI-only")

    story.append(Paragraph(
        f"Rapport d'audit LLM - Attribution de canaux ({model_label})", style_title
    ))
    story.append(Spacer(1, 0.3*cm))

    # Traçabilité complète du cas : graines, hyperparamètres, version et inférence
    run_meta = case_data.get("run_metadata", {})
    llm_params = get_llm_inference_params()
    header_data = [
        ["Cas", case_data.get("case_id", "-")],
        ["Scenario", case_data.get("scenario", "-")],
        ["Configuration", f"N={case_data.get('N')}, K={case_data.get('K')}, seed={case_data.get('seed')}"],
        ["Graines", f"topology_seed = {case_data.get('seed')}, "
                    f"algorithm_seed = {run_meta.get('algorithm_seed', '-')}"],
        ["Solveur", f"num_restarts = {run_meta.get('num_restarts', '-')}, "
                    f"patience_sweeps = {run_meta.get('patience_sweeps', '-')}, "
                    f"max_sweeps = {run_meta.get('max_sweeps', '-')}"],
        ["Mode", model_label],
        ["Version du code", run_meta.get("code_version", "-")],
        ["Modele", model_name or "Non specifie"],
        ["Inference", f"temperature = {llm_params['temperature']}, top_p = {llm_params['top_p']}, "
                      f"top_k = {llm_params['top_k']}, max_output_tokens = "
                      f"{llm_params['max_output_tokens']}, seed = {llm_params['seed']}"],
        ["Directive", f"Zero-Hallucination {llm_params['directive_version']}, "
                      f"audit semantique {llm_params['semantic_audit_version']}"],
        ["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        # Complétude de la réponse, visible dès l'en-tête
        ["Reponse LLM", Paragraph(
            f"<b>TRONQUE</b> ({'; '.join(truncation['reasons'])})"
            if truncation and truncation.get("truncated")
            else f"complete (finish_reason = {(truncation or {}).get('finish_reason') or '-'})",
            style_body)],
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

    # Deux statuts distincts : audit numérique (bloquant) et audit sémantique
    is_truncated = bool(truncation and truncation.get("truncated"))
    numeric = numeric_certification(verification_final, verification_struct_final,
                                    truncated=is_truncated)
    if numeric["reason"] == "TRONQUE":
        badge = ('<font color="#b30000"><b>AUDIT NUMERIQUE : NON CERTIFIE (TRONQUE)</b></font> - '
                 'reponse interrompue avant sa fin, conservee telle quelle sans correction.')
    elif numeric["status"] == "NON_CERTIFIE":
        badge = (f'<font color="#b30000"><b>AUDIT NUMERIQUE : NON CERTIFIE</b></font> - '
                 f'{numeric["invented_count"]} valeur(s) hors whitelist, '
                 f'{numeric["invalid_quadruplets"]} quadruplet(s) mal attribue(s).')
    elif numeric["status"] == "AVERTISSEMENT":
        badge = ('<font color="#b45309"><b>AUDIT NUMERIQUE : AVERTISSEMENT</b></font> - '
                 'aucune valeur numerique a verifier.')
    else:
        badge = (f'<font color="#006600"><b>AUDIT NUMERIQUE : CERTIFIE</b></font> - '
                 f'exactitude : {numeric["accuracy_rate"]:.1f}%, '
                 f'{numeric["checked_quadruplets"]} quadruplet(s) conforme(s).')
    story.append(Paragraph(badge, style_body))
    if semantic_final:
        story.append(Paragraph(
            f'<b>AUDIT SEMANTIQUE : {semantic_final["status"]}</b> - score '
            f'{semantic_final["score"]:.1f}% ({semantic_final["version"]}, non bloquant'
            + (f', {semantic_final["n_not_evaluable"]} controle(s) non evaluable(s)'
               if semantic_final.get("n_not_evaluable") else "") + ').',
            style_body))
    story.append(Spacer(1, 0.4*cm))

    # =========================================================================
    # SECTION 1 : Donnees de simulation
    # =========================================================================
    story.append(Paragraph("1. Donnees de simulation", style_h1))
    m = case_data.get("metrics", {})

    # [FIX-LLM-CCI-ACI] Tableau adapte au regime : en adjacent, les trois
    # composantes C_CCI, C_ACI, C_tot sont affichees separement.
    if model_label == "CCI+ACI":
        sim_data = [
            ["Metrique", "Valeur"],
            ["Cout initial", _fmt(m.get("cost_initial"))],
            ["Cout final", _fmt(m.get("cost_final"))],
            ["C_CCI", _fmt(m.get("conflicts_cci"))],
            ["C_ACI", _fmt(m.get("conflicts_aci"))],
            ["C_tot", _fmt(m.get("conflicts"))],
            ["Temps d'exécution (ms)", _fmt_us(m.get("time_seconds"))],
            ["Balayages (sweeps)", _fmt(m.get("sweeps"))],
            ["Canaux utilises", _fmt(m.get("used_channels"))],
            ["Restarts", _fmt(m.get("num_restarts"))],
        ]
    else:
        sim_data = [
            ["Metrique", "Valeur"],
            ["Cout initial", _fmt(m.get("cost_initial"))],
            ["Cout final", _fmt(m.get("cost_final"))],
            ["C_CCI", _fmt(m.get("conflicts_cci"))],
            ["Temps d'exécution (ms)", _fmt_us(m.get("time_seconds"))],
            ["Balayages (sweeps)", _fmt(m.get("sweeps"))],
            ["Canaux utilises", _fmt(m.get("used_channels"))],
            ["Restarts", _fmt(m.get("num_restarts"))],
        ]
    sim_table = Table(sim_data, colWidths=[7*cm, 5*cm])
    sim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(sim_table)
    story.append(Paragraph(
        "<i>C_CCI : nombre entier de paires voisines partageant le même canal, "
        "distinct du coût pondéré J(x). Temps en millisecondes (ms), quatre décimales.</i>",
        style_body,
    ))
    story.append(Spacer(1, 0.4*cm))

    # =========================================================================
    # SECTION 2 : Prompt de simulation (integral)
    # =========================================================================
    # [FIX-REPORT] Le prompt envoye au LLM est reproduit dans le rapport
    # pour garantir la tracabilite complete de la requete.
    story.append(Paragraph("2. Prompt de simulation", style_h1))
    story.append(Spacer(1, 0.15*cm))
    prompt_esc = (prompt.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
    story.append(Paragraph(prompt_esc.replace("\n", "<br/>"), style_mono))
    story.append(Spacer(1, 0.4*cm))

    # =========================================================================
    # SECTION 3 : Rapport Assistant LLM
    # =========================================================================
    # [FIX-REPORT] Renommage de "Reponse brute" en "Rapport Assistant LLM"
    # pour refleter la nature qualitative du contenu produit par le modele.
    story.append(Paragraph("3. Rapport Assistant LLM", style_h1))
    raw_esc = (raw_response.replace("&", "&amp;")
                          .replace("<", "&lt;")
                          .replace(">", "&gt;"))
    story.append(Paragraph(raw_esc.replace("\n", "<br/>"), style_body))
    story.append(Spacer(1, 0.4*cm))

    # =========================================================================
    # SECTION 4 : Audit numerique du verificateur LLM
    # =========================================================================
    # [FIX-REPORT] Tableau recapitulatif du travail de verification.
    # Affiche les nombres extraits, valides et inventes, avec les taux
    # d'exactitude avant et apres correction.
    story.append(Paragraph("4. Audit numerique du verificateur LLM", style_h1))
    story.append(Spacer(1, 0.15*cm))

    def _fmt_num_list(pairs, max_show=8):
        """Formate une liste de (valeur, representation brute)."""
        if not pairs:
            return "aucune"
        shown = pairs[:max_show]
        txt = ", ".join(f"{raw}" for _, raw in shown)
        if len(pairs) > max_show:
            txt += f" ... (+{len(pairs) - max_show})"
        return txt

    audit_data = [
        ["Indicateur", "Avant correction", "Apres correction"],
        [
            "Nombre total de valeurs extraites",
            str(len(verification_initial["all_numbers"])),
            str(len(verification_final["all_numbers"])),
        ],
        [
            "Valeurs conformes",
            str(len(verification_initial["valid_numbers"])),
            str(len(verification_final["valid_numbers"])),
        ],
        [
            "Valeurs inventees (hallucinations)",
            str(len(verification_initial["invented_numbers"])),
            str(len(verification_final["invented_numbers"])),
        ],
        [
            "Taux d'exactitude",
            f"{verification_initial['accuracy_rate']:.1f}%",
            f"{verification_final['accuracy_rate']:.1f}%",
        ],
        [
            "Quadruplets (scenario, methode, metrique, valeur) verifies",
            str((verification_struct_initial or {}).get("total_claims", "-")),
            str((verification_struct_final or {}).get("total_claims", "-")),
        ],
        [
            "Quadruplets mal attribues",
            str(len((verification_struct_initial or {}).get("invalid_claims", []))),
            str(len((verification_struct_final or {}).get("invalid_claims", []))),
        ],
        [
            "Statut numerique",
            numeric_certification(verification_initial, verification_struct_initial,
                                  truncated=is_truncated)["label"],
            numeric_certification(verification_final, verification_struct_final,
                                  truncated=is_truncated)["label"],
        ],
    ]
    audit_table = Table(audit_data, colWidths=[7*cm, 4.5*cm, 4.5*cm])
    audit_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(audit_table)
    story.append(Spacer(1, 0.3*cm))

    # Detail des valeurs inventees (avant correction)
    if verification_initial["invented_numbers"]:
        detail_txt = (
            "<b>Valeurs inventees detectees (avant correction) :</b> "
            + _fmt_num_list(verification_initial["invented_numbers"])
        )
        story.append(Paragraph(detail_txt, style_body))
        story.append(Spacer(1, 0.2*cm))

    # Nombre de tentatives de correction
    story.append(Paragraph(
        f"<b>Nombre de tentatives de correction :</b> "
        f"{case_data.get('correction_attempts', 0)}",
        style_body
    ))
    story.append(Spacer(1, 0.4*cm))

    # Audit sémantique : qualité rédactionnelle, cohérence logique et style,
    # restitué séparément et sans effet sur la certification numérique
    if semantic_final:
        story.append(Paragraph("4 bis. Audit semantique (non bloquant)", style_h1))
        sem_rows = [["Code", "Controle", "Resultat"]]
        for check in semantic_final["checks"]:
            sem_rows.append([check["code"], Paragraph(check["label"], style_body),
                             ("NON EVALUABLE" if not check.get("evaluable", True)
                              else "OK" if check["passed"]
                              else f"ECHEC {check['detail']}".strip())])
        sem_table = Table(sem_rows, colWidths=[1.5*cm, 10*cm, 4.5*cm])
        sem_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(sem_table)
        story.append(Spacer(1, 0.4*cm))

    # =========================================================================
    # SECTION 5 : Rapport Assistant LLM corrige (si applicable)
    # =========================================================================
    if corrected_response and corrected_response != raw_response:
        story.append(Paragraph("5. Rapport Assistant LLM corrige", style_h1))
        corr_esc = (corrected_response.replace("&", "&amp;")
                                     .replace("<", "&lt;")
                                     .replace(">", "&gt;"))
        story.append(Paragraph(corr_esc.replace("\n", "<br/>"), style_body))

    doc.build(story)


# -----------------------------------------------------------------------------
# E. MASTER AUDIT PIPELINE
# -----------------------------------------------------------------------------

def make_audit_record(case_data: dict, status: str, error_code: str = None,
                      error_message: str = "", finish_reason: str = None,
                      llm_params: dict = None, **fields) -> dict:
    """
    Résumé sérialisable d'un cas E10, seule source du CSV de fidélité et de la
    reprise de campagne. Un cas sans rapport garde ses indicateurs à None.
    """
    params = llm_params or get_llm_inference_params()
    record = {
        "case_id": case_data.get("case_id"),
        "scenario": case_data.get("scenario"),
        "N": case_data.get("N"), "K": case_data.get("K"), "seed": case_data.get("seed"),
        "context": case_data.get("context"),
        "model_label": case_data.get("model_label", "CCI-only"),
        "status": status,
        "error_code": error_code,
        "error_message": error_message,
        "finish_reason": finish_reason,
        "truncated": None, "truncation_reasons": [],
        "response_words": None,
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "hallucination_initial": None, "invented_initial": [], "hallucination_final": None,
        "accuracy_initial": None, "accuracy_final": None,
        "numeric_status": None, "quadruplets_checked": None, "quadruplets_invalid": None,
        "semantic_status": None, "semantic_score": None,
        "correction_attempts": 0, "correction_error_code": None,
        "llm_params": params,
        "directive_version": params.get("directive_version"),
        "code_version": (case_data.get("run_metadata") or {}).get("code_version"),
        "pdf_path": None,
    }
    record.update(fields)
    return record


def save_audit_record(record: dict, raw_dir) -> Path:
    path = Path(raw_dir) / f"{record['case_id']}{AUDIT_RECORD_SUFFIX}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    return path


def audit_case(case_data: dict, max_correction_attempts: int = 2,
               model_folder: str = "cochannel") -> dict:
    case_id = case_data.get("case_id", "unknown")
    model_label = case_data.get("model_label", "CCI-only")
    # Modèle unique pour la campagne scientifique : aucun basculement silencieux
    allow_fallback = not LLM_CAMPAIGN_FIXED_MODEL

    raw_dir, reports_dir = _get_mode_folders(model_label)
    prompt = build_prompt(case_data)

    with open(raw_dir / f"{case_id}_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    call = ask_llm_detailed(prompt, allow_fallback=allow_fallback)
    llm_params = get_llm_inference_params()
    base_result = {
        "case_id": case_id, "scenario": case_data.get("scenario"),
        "N": case_data.get("N"), "K": case_data.get("K"),
        "seed": case_data.get("seed"), "context": case_data.get("context"),
        "model_label": model_label, "llm_params": llm_params,
        "error_code": call["error_code"], "finish_reason": call["finish_reason"],
        "retry_after_s": call["retry_after_s"], "quota_reset_s": call["quota_reset_s"],
    }

    if call["error_code"]:
        error_text = f"[ERREUR LLM] {call['error_code']} : {call['error_message']}"
        with open(raw_dir / f"{case_id}_error.txt", "w", encoding="utf-8") as f:
            f.write(error_text)
        # Aucun indicateur fabriqué : taux et statuts restent indéfinis (NaN au CSV)
        empty_verif = {
            "all_numbers": [], "allowed_numbers": set(),
            "valid_numbers": [], "invented_numbers": [],
            "has_hallucination": False, "accuracy_rate": float("nan"),
        }
        record = make_audit_record(case_data, "LLM_FAILED", call["error_code"],
                                   call["error_message"], call["finish_reason"], llm_params)
        save_audit_record(record, raw_dir)
        return {
            **base_result,
            "raw_response": error_text, "corrected_response": error_text,
            "verification_initial": empty_verif, "verification_final": empty_verif,
            "correction_attempts": 0, "correction_error_code": None,
            "truncated": None, "pdf_path": None, "status": "LLM_FAILED",
            "audit_record": record,
        }

    raw_response = call["text"]
    with open(raw_dir / f"{case_id}_raw_response.txt", "w", encoding="utf-8") as f:
        f.write(raw_response)

    truncation = detect_truncation(raw_response, call["finish_reason"])
    truncated = truncation["truncated"]
    verification_initial = verify_numbers(raw_response, case_data)
    verification_struct_initial = verify_structured_claims(raw_response, case_data)
    numeric_initial = numeric_certification(verification_initial, verification_struct_initial,
                                            truncated=truncated)

    corrected_response = raw_response
    verification_final = verification_initial
    verification_struct_final = verification_struct_initial
    numeric_final = numeric_initial
    correction_attempts = 0
    correction_error_code = None

    # Rapport tronqué : conservé tel quel, sans correction. Sinon, un chiffre
    # inventé OU un quadruplet mal attribué déclenche la régénération.
    while (not truncated and numeric_final["status"] == "NON_CERTIFIE"
           and correction_attempts < max_correction_attempts):
        correction_attempts += 1
        candidate = regenerate_with_correction(
            case_data, prompt, corrected_response, verification_final,
            structured_verification=verification_struct_final,
        )
        if candidate.startswith("[ERREUR LLM]"):
            code_match = re.match(r"\[ERREUR LLM\] ([A-Z_]+)", candidate)
            correction_error_code = code_match.group(1) if code_match else "OTHER"
            break
        with open(raw_dir / f"{case_id}_corrected_v{correction_attempts}.txt", "w",
                  encoding="utf-8") as f:
            f.write(candidate)
        # Correction elle-même tronquée : la version précédente, complète, est conservée
        if detect_truncation(candidate)["truncated"]:
            correction_error_code = "TRUNCATED"
            break
        corrected_response = candidate
        verification_final = verify_numbers(corrected_response, case_data)
        verification_struct_final = verify_structured_claims(corrected_response, case_data)
        numeric_final = numeric_certification(verification_final, verification_struct_final)

    semantic_final = semantic_audit(corrected_response, case_data, truncated=truncated)
    pdf_path = reports_dir / f"rapport_{case_id}_{case_data.get('scenario')}.pdf"

    case_data_with_attempts = dict(case_data)
    case_data_with_attempts["correction_attempts"] = correction_attempts

    generate_pdf_report(
        case_data=case_data_with_attempts, prompt=prompt,
        raw_response=raw_response, verification_initial=verification_initial,
        corrected_response=corrected_response, verification_final=verification_final,
        output_path=pdf_path, model_name=llm_params["model"],
        verification_struct_initial=verification_struct_initial,
        verification_struct_final=verification_struct_final,
        semantic_final=semantic_final,
        truncation=truncation,
    )

    status = "TRONQUE" if truncated else "OK"
    record = make_audit_record(
        case_data, status, "TRUNCATED" if truncated else None,
        "; ".join(truncation["reasons"]), call["finish_reason"], llm_params,
        truncated=truncated,
        truncation_reasons=truncation["reasons"],
        response_words=len(raw_response.split()),
        hallucination_initial=bool(verification_initial["has_hallucination"]),
        invented_initial=[raw for _, raw in verification_initial["invented_numbers"]],
        hallucination_final=bool(verification_final["has_hallucination"]),
        accuracy_initial=float(verification_initial["accuracy_rate"]),
        accuracy_final=float(verification_final["accuracy_rate"]),
        numeric_status=numeric_final["label"],
        quadruplets_checked=int(numeric_final["checked_quadruplets"]),
        quadruplets_invalid=int(numeric_final["invalid_quadruplets"]),
        semantic_status=semantic_final["status"],
        semantic_score=float(semantic_final["score"]),
        correction_attempts=correction_attempts,
        correction_error_code=correction_error_code,
        pdf_path=str(pdf_path),
    )
    save_audit_record(record, raw_dir)

    return {
        **base_result,
        "error_code": record["error_code"],
        "raw_response": raw_response, "corrected_response": corrected_response,
        "verification_initial": verification_initial,
        "verification_final": verification_final,
        "verification_struct_initial": verification_struct_initial,
        "verification_struct_final": verification_struct_final,
        "numeric_initial": numeric_initial,
        "numeric_final": numeric_final,
        "semantic_final": semantic_final,
        "correction_attempts": correction_attempts,
        "correction_error_code": correction_error_code,
        "truncated": truncated, "truncation": truncation,
        "pdf_path": str(pdf_path), "status": status,
        "audit_record": record,
    }