# llm_assistant.py
"""
Module LLM pour le mémoire BD-CeNN + LLM.

Ce module contient :
    A. Le générateur de Prompt avec Système de Garde-fous (Zero-Hallucination Directive).
    B. L'interface avec le LLM (Google AI Studio / Gemini, via API).
    C. Le Contrôleur Mathématique Indépendant (Regex & Parser Garde-fou).
    D. La génération de rapports PDF professionnels.
    E. Le mécanisme de régénération sous contrainte (correction des hallucinations).

NOTE : L'évaluation académique automatique (grille 5 critères × 2 pts) a été
RETIRÉE de ce module. Les cotes sont attribuées MANUELLEMENT par l'auteur
dans le rapport final, en se basant sur les PDF individuels et le CSV.

Auteur : Nelson Ngombo
"""

import os
import re
import json
import time
import random
import threading            # ← nécessaire pour la gestion de timeout
import requests
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 0. SUPPRESSION DES WARNINGS COSMÉTIQUES DU SDK GOOGLE GENAI
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*automatic function calling.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*AFC.*", category=UserWarning)


# -----------------------------------------------------------------------------
# 1. CONFIGURATION GLOBALE
# -----------------------------------------------------------------------------
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "❌ La variable GOOGLE_API_KEY est absente. "
        "Vérifiez votre fichier .env à la racine du projet."
    )

# -----------------------------------------------------------------------------
# Modèles Gemini à utiliser
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
# Constantes de temporisation et de retry
# -----------------------------------------------------------------------------
# Pause avant chaque NOUVELLE requête (hors retry).
# Objectif : laisser gemini-3.6-flash se stabiliser entre deux appels.
INTER_REQUEST_DELAY = 5.0  # secondes

# --- Gestion de timeout sur la réponse du LLM ---
# STRATÉGIE À DEUX COUCHES :
#   Couche 1 (timeout) : si le LLM ne répond pas dans LLM_TIMEOUT_SECONDS
#                        secondes, on abandonne la tentative et on relance.
#   Couche 2 (erreurs SDK) : rotation de modèles + backoff exponentiel.
LLM_TIMEOUT_SECONDS = 60     # Délai max d'attente par appel LLM (secondes)
LLM_TIMEOUT_RETRIES = 3      # Nombre de relances supplémentaires après timeout
LLM_TIMEOUT_DELAY = 60       # Délai entre deux relances globales (secondes)

from config import LLM_LOGS_DIR, FIGURES_DIR, CSV_DIR

LLM_LOGS_DIR = Path(LLM_LOGS_DIR)
LLM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
LLM_REPORTS_DIR = LLM_LOGS_DIR / "reports_pdf"
LLM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LLM_RAW_DIR = LLM_LOGS_DIR / "raw_responses"
LLM_RAW_DIR.mkdir(parents=True, exist_ok=True)


def _get_mode_folders(model_label: str):
    """
    Retourne (raw_dir, reports_dir) correspondant au mode d'interférence.

    - "CCI-only"  → sous-dossier "cochannel"
    - "CCI+ACI"   → sous-dossier "adjacent"
    - autre       → sous-dossier "misc" (fallback)
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
# 2. FONCTIONS UTILITAIRES
# -----------------------------------------------------------------------------

def _fmt(value, ndigits=2):
    """Formate un nombre pour l'affichage dans les prompts et rapports."""
    try:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{value:.{ndigits}f}"
    except Exception:
        pass
    return str(value)


def _conflicts_label(model_label: str) -> str:
    """
    Retourne le libellé à utiliser pour la ligne "conflits" selon le mode.

    - CCI-only → "Conflits co-canal (CCI)"
    - CCI+ACI  → "Conflits totaux (CCI+ACI)"
    """
    if model_label == "CCI+ACI":
        return "Conflits totaux (CCI+ACI)"
    return "Conflits co-canal (CCI)"


# -----------------------------------------------------------------------------
# A. GÉNÉRATEUR DE PROMPT AVEC GARDE-FOUS
# -----------------------------------------------------------------------------

ZERO_HALLUCINATION_DIRECTIVE = """
=== DIRECTIVE SYSTÈME (ZERO-HALLUCINATION) ===
Tu es un assistant d'ingénierie radio. Ton rôle est d'auditer et d'expliquer
les résultats d'une simulation d'attribution de canaux.

RÈGLES STRICTES :
1. Tu ne dois JAMAIS inventer, altérer ou extrapoler un chiffre.
2. Tu dois citer EXCLUSIVEMENT les valeurs numériques présentes dans les
   données fournies.
3. Si une information n'est pas présente dans les données, écris simplement :
   "Information non disponible dans les données fournies."
4. Toute valeur numérique que tu cites doit pouvoir être retrouvée textuellement
   dans le bloc "DONNÉES DE SIMULATION" ci-dessous.
5. Si tu détectes une incohérence dans les données, signale-le sans inventer.
6. Rédige une analyse structurée et concise, comme un rapport d'ingénieur.

=== RÈGLE D'ÉNUMÉRATION ===
- Pour énumérer tes points, utilise OBLIGATOIREMENT des LETTRES MAJUSCULES
  suivies d'une parenthèse fermante : A), B), C), D)...
- N'utilise JAMAIS de chiffres suivis d'un point (1., 2., 3...), car tout
  chiffre dans ta réponse est interprété comme une DONNÉE NUMÉRIQUE de la
  simulation et sera vérifié par le contrôleur automatique.

=== RÈGLE SUR LES NOMBRES (TRÈS IMPORTANTE) ===
Un contrôleur automatique indépendant vérifie CHAQUE nombre écrit en chiffres
dans ta réponse. Tout chiffre qui n'est pas un résultat de la simulation
fournie sera CONSIDÉRÉ COMME UNE HALLUCINATION NUMÉRIQUE, même s'il s'agit
d'un simple numéro dans une phrase.

  RÈGLE 1 — RÉSULTATS DE SIMULATION : EN CHIFFRES.
  Écris en chiffres UNIQUEMENT les valeurs qui proviennent du bloc
  "DONNÉES DE SIMULATION" ci-dessous (coût, conflits, temps, itérations,
  canaux utilisés, N, K, seed, identifiant du cas, nom du scénario).

  RÈGLE 2 — NOMBRES HORS SIMULATION : EN TOUTES LETTRES.
  Tous les autres nombres que tu souhaites introduire dans ton texte
  (numéros d'ordre, bornes d'échelle, effectifs génériques, dates,
  numéros de version, etc.) DOIVENT être écrits EN TOUTES LETTRES.

  EXEMPLES CORRECTS :
    ✓ "sur les trois baselines..."
    ✓ "au cours des deux premières itérations..."
    ✓ "la note de confiance est quatre sur cinq"
    ✓ "Le coût final s'élève à 37.75"  (37.75 = résultat, donc en chiffres)

  EXEMPLES INTERDITS :
    ✗ "sur les 3 baselines..."
    ✗ "pendant les 2 premières..."
    ✗ "la note de confiance est 4 sur 5"

POURQUOI CETTE RÈGLE ?
Le contrôleur automatique ne fait pas la différence entre un chiffre qui est
un RÉSULTAT de la simulation et un chiffre qui est un simple NUMÉRO dans une
phrase. Écrire les numéros non-résultats en toutes lettres élimine toute
ambiguïté et garantit qu'aucun faux positif d'hallucination n'est déclenché.

=== RÈGLE DE SINCÉRITÉ ===
- Reste strictement factuel dans tes comparaisons. Si une baseline (Random,
  Greedy, DSATUR) est meilleure que BD-CeNN sur un critère (coût ou temps),
  dis-le explicitement sans enjoliver.
=== FIN DIRECTIVE ===
"""


def build_prompt(case_data: dict) -> str:
    """
    Construit un prompt structuré avec garde-fous pour un cas de simulation.

    SÉMANTIQUE DES MODES (IMPORTANT) :
      - "CCI-only" : coût et conflits = interférences co-canal uniquement.
      - "CCI+ACI"  : coût et conflits = interférences co-canal ET adjacent
                     (UN SEUL chiffre combiné).

    CONTENU SUPPLÉMENTAIRE FOURNI AU LLM :
      - Les métadonnées de la topologie (seed, threshold, N, K).
      - Les positions (x, y) des cellules du réseau.
      - La topologie du réseau (arêtes d'interférence avec poids).
      - Les affectations (cellule → canal) pour les 4 méthodes.
      - Un résumé des cellules les plus contraintes.
    Ces informations permettent au LLM de mieux argumenter ses analyses.
    """
    m = case_data.get("metrics", {})
    b = case_data.get("baselines", {})
    model_label = case_data.get("model_label", "CCI-only")

    # -------------------------------------------------------------------------
    # Bloc BASELINES
    # -------------------------------------------------------------------------
    baseline_lines = []
    for name in ["Random", "Greedy", "DSATUR"]:
        if name in b:
            baseline_lines.append(
                f"  - {name} : coût = {_fmt(b[name].get('cost'))}, "
                f"conflits = {_fmt(b[name].get('conflicts'))}, "
                f"temps = {_fmt(b[name].get('time'), 6)} s"
            )
    baseline_str = "\n".join(baseline_lines) if baseline_lines else "  (aucune baseline fournie)"

    # -------------------------------------------------------------------------
    # Bloc CELLULES EN CONFLIT
    # -------------------------------------------------------------------------
    conflicting_cells = case_data.get("conflicting_cells", [])
    conflicting_str = (
        ", ".join(str(c) for c in conflicting_cells)
        if conflicting_cells
        else "aucune (ou non disponible)"
    )

    # -------------------------------------------------------------------------
    # Bloc NOTE SUR LE MODE
    # -------------------------------------------------------------------------
    if model_label == "CCI+ACI":
        conflicts_label = "Conflits totaux (CCI+ACI)"
        mode_note = (
            "Le coût et le nombre de conflits présentés ci-dessous intègrent\n"
            "À LA FOIS les interférences co-canal (même canal) ET les\n"
            "interférences entre canaux adjacents. C'est UN SEUL chiffre\n"
            "combiné, pas deux chiffres séparés."
        )
    else:
        conflicts_label = "Conflits co-canal (CCI)"
        mode_note = (
            "Le coût et le nombre de conflits présentés ci-dessous prennent en\n"
            "compte UNIQUEMENT les interférences co-canal (même canal attribué\n"
            "à deux cellules interférentes)."
        )

    # -------------------------------------------------------------------------
    # Bloc MÉTADONNÉES DE LA TOPOLOGIE (seed, threshold, N, K)
    # -------------------------------------------------------------------------
    # Ces informations proviennent du fichier scenarios_data.json et
    # permettent de rappeler au LLM dans quelles conditions exactes la
    # topologie a été générée (reproductibilité scientifique).
    topology_meta = case_data.get("topology_meta", {})
    if topology_meta:
        meta_str = (
            f"  - Seed de la topologie (fichier JSON) : {topology_meta.get('seed', 'N/A')}\n"
            f"  - Threshold (seuil de portée radio)    : {topology_meta.get('threshold', 'N/A')}\n"
            f"  - Nombre de cellules N                 : {topology_meta.get('N', 'N/A')}\n"
            f"  - Nombre de canaux K                   : {topology_meta.get('K', 'N/A')}"
        )
    else:
        meta_str = "  (métadonnées non disponibles)"

    # -------------------------------------------------------------------------
    # Bloc POSITIONS DES CELLULES (x, y)
    # -------------------------------------------------------------------------
    # Pour les petits réseaux (≤ 20 cellules), toutes les positions sont
    # affichées. Pour les réseaux plus grands, on montre un extrait (20
    # premières cellules) pour éviter d'exploser le prompt.
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
        pos_lines.append(f"  ... ({len(cell_positions) - 20} autres cellules non affichées)")
        positions_str = "\n".join(pos_lines)
    else:
        positions_str = "  (positions non disponibles)"

    # -------------------------------------------------------------------------
    # Bloc TOPOLOGIE DU RÉSEAU (arêtes d'interférence)
    # -------------------------------------------------------------------------
    topology_edges = case_data.get("topology_edges", [])
    total_edges = case_data.get("total_edges", 0)
    if topology_edges:
        topology_lines = []
        for (i, j, w) in topology_edges:
            topology_lines.append(f"  - Cellule {i} <-> Cellule {j} : poids W = {w}")
        topology_body = "\n".join(topology_lines)
        if total_edges > len(topology_edges):
            topology_header = (
                f"({len(topology_edges)} arêtes affichées sur {total_edges} au total ; "
                f"triées par poids décroissant — les plus critiques en premier)"
            )
        else:
            topology_header = f"({total_edges} arêtes au total)"
        topology_str = f"{topology_header}\n{topology_body}"
    else:
        topology_str = "  (topologie non disponible)"

    # -------------------------------------------------------------------------
    # Bloc RÉSUMÉ DES CELLULES LES PLUS CONTRAINTES
    # -------------------------------------------------------------------------
    cell_summary = case_data.get("cell_summary", [])
    if cell_summary:
        cs_lines = []
        for entry in cell_summary:
            cs_lines.append(
                f"  - Cellule {entry['cell']:3d} : "
                f"{entry['degree']:2d} voisins, "
                f"force cumulée = {entry['strength']}"
            )
        cell_summary_str = "\n".join(cs_lines)
    else:
        cell_summary_str = "  (résumé non disponible)"

    # -------------------------------------------------------------------------
    # Bloc AFFECTATIONS PAR MÉTHODE (cellule → canal)
    # -------------------------------------------------------------------------
    # Format compact : la i-ème valeur est le canal de la cellule i.
    allocations = case_data.get("allocations", {})
    if allocations:
        alloc_lines = []
        for method_name in ["Random", "Greedy", "DSATUR", "BD-CeNN"]:
            if method_name in allocations:
                alloc_list = allocations[method_name]
                alloc_arr = "[" + ", ".join(str(c) for c in alloc_list) + "]"
                # Aligner proprement les noms de méthodes
                alloc_lines.append(f"  - {method_name:8s} : {alloc_arr}")
        allocations_str = "\n".join(alloc_lines)
        allocations_header = (
            "(La i-ème valeur est le canal attribué à la cellule i. "
            "Les indices commencent à zéro.)"
        )
    else:
        allocations_str = "  (affectations non disponibles)"
        allocations_header = ""

    # -------------------------------------------------------------------------
    # Assemblage final du prompt
    # -------------------------------------------------------------------------
    prompt = f"""
{ZERO_HALLUCINATION_DIRECTIVE}

=== CONTEXTE DU CAS ===
- Cas : {case_data.get('case_id')}
- Scénario : {case_data.get('scenario')}
- Configuration : N = {case_data.get('N')} cellules, K = {case_data.get('K')} canaux
- Seed de topologie : {case_data.get('seed')}
- Mode d'interférence : {model_label}
- Contexte : {case_data.get('context')}

=== NOTE SUR LE MODE D'INTERFÉRENCE ===
{mode_note}

=== DONNÉES DE SIMULATION (SOLVEUR BD-CeNN) ===
- Coût initial J_0 : {_fmt(m.get('cost_initial'))}
- Coût final J(x) : {_fmt(m.get('cost_final'))}
- {conflicts_label} : {_fmt(m.get('conflicts'))}
- Temps d'exécution du BD-CeNN : {_fmt(m.get('time_seconds'), 6)} s
- Nombre d'itérations avant convergence : {_fmt(m.get('iterations'))}
- Canaux utilisés : {_fmt(m.get('used_channels'))}

=== BASELINES (COMPARAISON) ===
{baseline_str}

=== MÉTADONNÉES DE LA TOPOLOGIE (fichier JSON) ===
{meta_str}

=== POSITIONS DES CELLULES (coordonnées x, y) ===
{positions_str}

=== TOPOLOGIE DU RÉSEAU (arêtes d'interférence) ===
{topology_str}

=== RÉSUMÉ DES CELLULES LES PLUS CONTRAINTES (top 10) ===
{cell_summary_str}

=== AFFECTATIONS PAR MÉTHODE (cellule → canal) ===
{allocations_header}
{allocations_str}

=== CELLULES ENCORE EN CONFLIT ===
{conflicting_str}

=== TÂCHES DEMANDÉES (à énumérer avec des LETTRES : A), B), C)...) ===
A) Résume la qualité de la solution BD-CeNN en comparant le COÛT FINAL aux
   baselines. Appuie-toi sur les AFFECTATIONS fournies ci-dessus pour
   identifier les cellules où BD-CeNN attribue un canal différent des
   baselines et qui expliquent les écarts observés.

B) Évalue la convergence en citant explicitement le nombre d'itérations
   avant convergence.

C) Compare les PERFORMANCES DE CALCUL (TEMPS D'EXÉCUTION) :
   - Cite le temps d'exécution du BD-CeNN et celui de chaque baseline.
   - Si une baseline est PLUS RAPIDE que BD-CeNN, dis-le franchement.

D) Analyse la TOPOLOGIE DU RÉSEAU :
   - Identifie les cellules les plus contraintes à l'aide du RÉSUMÉ fourni.
   - Explique comment ces contraintes locales influencent la difficulté du
     problème et la qualité de la solution obtenue.
   - Si pertinent, cite explicitement certaines arêtes critiques (poids 4)
     et les cellules concernées.

E) Dans le cas où les résultats du BD-CeNN ne sont pas fameux, as-tu une
   recommandation à faire en tant qu'assistant de l'ingénieur radio pour
   améliorer les résultats ? Si les résultats sont satisfaisants, tu peux
   dire "pas de recommandation à faire vu les bons résultats obtenus".

F) Termine par un avis de confiance : attribue une note de confiance
   sur une échelle à cinq niveaux (le niveau maximal étant le plus élevé),
   puis justifie-la brièvement. Écris la note EN TOUTES LETTRES
   (par exemple : "quatre sur cinq"), PAS en chiffres.

Rédige ton rapport de façon détaillée, longue et argumentée en français,
de manière professionnelle et structurée.

Rappels :
- N'utilise JAMAIS de chiffres pour énumérer. Utilise A), B), C)...
- Tous les chiffres cités doivent provenir EXCLUSIVEMENT des données ci-dessus.
- Pour tout autre nombre (numéros d'ordre, bornes d'échelle), utilise des
  LETTRES (ex. "trois", "quatre sur cinq").
- Ne cite PAS de "conflits CCI" ni de "conflits ACI" séparément dans ce mode.
  Utilise UNIQUEMENT le chiffre "{conflicts_label}" fourni ci-dessus.
- Tu peux citer des indices de cellules (par exemple "la cellule 17")
  et des canaux (par exemple "canal 2") car ils font partie des données
  de simulation fournies.
"""
    return prompt.strip()


# -----------------------------------------------------------------------------
# B. INTERFACE AVEC LE LLM
# -----------------------------------------------------------------------------

from google import genai
from google.genai import types

_client = genai.Client(api_key=GOOGLE_API_KEY)


def _check_model_available(preferred: str, fallbacks: list) -> str:
    candidates = [preferred] + [f for f in fallbacks if f != preferred]
    for c in candidates:
        try:
            _client.models.generate_content(
                model=c,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return c
        except Exception as e:
            print(f"  ⚠️ Modèle '{c}' non disponible : {type(e).__name__}")
            continue
    raise RuntimeError(
        f"❌ Aucun modèle disponible parmi {candidates}. "
        f"Vérifiez votre clé API ou la liste des modèles."
    )


GOOGLE_MODEL = _check_model_available(GOOGLE_MODEL_PREFERRED, GOOGLE_MODEL_FALLBACKS)
print(f"✅ Modèle Gemini retenu : {GOOGLE_MODEL}")

_LAST_SUCCESSFUL_MODEL = None


def _call_generate_content_with_timeout(model_to_try: str, prompt: str,
                                         max_tokens: int, timeout_seconds: float):
    """
    Appelle `_client.models.generate_content` dans un thread daemon et
    applique un timeout global.

    POURQUOI UN THREAD ?
        Le SDK `google-genai` n'expose pas de paramètre `timeout` pour
        `generate_content` de manière portable. Pour garantir que l'appel
        ne bloque pas indéfiniment, on l'exécute dans un thread séparé et
        on utilise `thread.join(timeout)` pour l'abandonner au bout du
        délai.

    RETOUR :
        (response, error_type, error_message)
          - En cas de succès : (response, None, None)
          - En cas de timeout : (None, "TIMEOUT", message)
          - En cas d'exception SDK : (None, "SDK_ERROR", message)
    """
    result_container = {
        "response": None,
        "error": None,
        "done": False,
    }

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

    # Thread daemon : s'il ne termine pas, il ne bloquera pas l'arrêt du programme.
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)

    if not result_container["done"]:
        # Le thread tourne toujours → timeout
        return None, "TIMEOUT", (
            f"Aucune réponse du modèle '{model_to_try}' après "
            f"{timeout_seconds:.0f}s."
        )

    if result_container["error"] is not None:
        e = result_container["error"]
        return None, "SDK_ERROR", f"{type(e).__name__} : {e}"

    return result_container["response"], None, None


def ask_llm(prompt: str, timeout: int = 180, max_retries: int = 8,
            inter_request_delay: float = INTER_REQUEST_DELAY) -> str:
    """
    Interroge le LLM Google Gemini via le SDK `google-genai`.

    GESTION DE TIMEOUT ET DE RETRY (nouvelle version) :

    COUCHE 1 — TIMEOUT GLOBAL :
        Chaque appel au LLM est limité à LLM_TIMEOUT_SECONDS (60s par défaut).
        Si le LLM ne répond pas dans ce délai, on abandonne l'appel et on
        relance une tentative globale après LLM_TIMEOUT_DELAY (60s par défaut).
        On effectue au total LLM_TIMEOUT_RETRIES + 1 = 4 tentatives globales
        (1 initiale + 3 relances).

    COUCHE 2 — ERREURS SDK :
        Dans chaque tentative globale, on gère les erreurs transitoires du SDK
        (503 surcharge, 429 quota, 404 modèle indisponible) par rotation de
        modèles + backoff exponentiel. Chaque modèle a son propre quota, ce
        qui multiplie nos chances de succès.

    ÉCHEC TOTAL :
        Si aucune tentative globale ne réussit, on renvoie un message explicite
        invitant à vérifier la connexion internet et l'état du service Gemini.
    """
    global _LAST_SUCCESSFUL_MODEL

    # Pause inter-requête (une seule fois en début de fonction).
    if inter_request_delay > 0:
        print(f"       Pause de {inter_request_delay:.1f}s avant la requête...")
        time.sleep(inter_request_delay)

    # Construire la liste ordonnée des modèles à essayer.
    all_models = [GOOGLE_MODEL] + [m for m in GOOGLE_MODEL_FALLBACKS if m != GOOGLE_MODEL]
    if _LAST_SUCCESSFUL_MODEL and _LAST_SUCCESSFUL_MODEL in all_models:
        candidates = [_LAST_SUCCESSFUL_MODEL] + [m for m in all_models if m != _LAST_SUCCESSFUL_MODEL]
    else:
        candidates = all_models

    total_timeout_attempts = LLM_TIMEOUT_RETRIES + 1
    final_error = None

    # =========================================================================
    # COUCHE 1 — BOUCLE DE RETRY GLOBAL SUR TIMEOUT
    # =========================================================================
    for timeout_attempt in range(1, total_timeout_attempts + 1):

        # Délai entre deux tentatives globales (sauf avant la première).
        if timeout_attempt > 1:
            print(f"     🔁  Relance globale {timeout_attempt}/{total_timeout_attempts} "
                  f"dans {LLM_TIMEOUT_DELAY:.0f}s (le LLM n'a pas répondu dans "
                  f"les {LLM_TIMEOUT_SECONDS:.0f}s)...")
            time.sleep(LLM_TIMEOUT_DELAY)

        tried_models = set()
        last_error = None
        current_max_tokens = 8000
        timeout_triggered_this_round = False

        # =====================================================================
        # COUCHE 2 — BOUCLE DE RETRY INTERNE SUR ERREURS SDK (rotation modèles)
        # =====================================================================
        for attempt in range(1, max_retries + 1):
            # Sélectionner le prochain modèle non encore essayé.
            model_to_try = None
            for c in candidates:
                if c not in tried_models:
                    model_to_try = c
                    break
            if model_to_try is None:
                tried_models.clear()
                model_to_try = candidates[0]

            # Appel du LLM avec timeout.
            response, error_type, error_msg = _call_generate_content_with_timeout(
                model_to_try=model_to_try,
                prompt=prompt,
                max_tokens=current_max_tokens,
                timeout_seconds=LLM_TIMEOUT_SECONDS,
            )

            # ---------------------------------------------------------------
            # CAS 1 — TIMEOUT : on sort de la boucle interne et on relance
            # une tentative globale après le délai.
            # ---------------------------------------------------------------
            if error_type == "TIMEOUT":
                print(f"     ⏱️   TIMEOUT : {error_msg}")
                timeout_triggered_this_round = True
                last_error = f"Timeout global ({LLM_TIMEOUT_SECONDS}s) sur '{model_to_try}'"
                break  # sortir de la boucle interne

            # ---------------------------------------------------------------
            # CAS 2 — ERREUR SDK : on gère la rotation / backoff.
            # ---------------------------------------------------------------
            if error_type == "SDK_ERROR":
                last_error = error_msg
                error_str = error_msg
                is_503 = "503" in error_str or "UNAVAILABLE" in error_str
                is_429 = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                is_404 = "404" in error_str or "NOT_FOUND" in error_str
                needs_rotation = is_503 or is_429 or is_404

                short_err = last_error.split("\n")[0][:140]
                print(f"  ⚠️ Tentative {attempt}/{max_retries} sur '{model_to_try}' échouée : {short_err}")
                tried_models.add(model_to_try)

                if attempt < max_retries:
                    if needs_rotation:
                        next_model = next(
                            (c for c in candidates if c not in tried_models),
                            candidates[0],
                        )
                        if is_503:
                            reason = "503 (surcharge)"
                        elif is_429:
                            reason = "429 (quota épuisé)"
                        else:
                            reason = "404 (modèle indisponible)"
                        wait = 1 + random.uniform(0, 3)
                        print(f"     🔄 {reason} → rotation vers '{next_model}'. Attente {wait:.1f}s...")
                    else:
                        wait = min(60, (2 ** (attempt + 1)) + random.uniform(0, 3))
                        print(f"     ⏳ Attente de {wait:.1f}s avant nouvelle tentative...")
                    time.sleep(wait)
                continue  # passer à la tentative suivante de la boucle interne

            # ---------------------------------------------------------------
            # CAS 3 — SUCCÈS : on extrait le texte et on retourne.
            # ---------------------------------------------------------------
            extracted_text = ""
            finish_reason_str = None

            if response.candidates:
                cand = response.candidates[0]
                # Détecter la raison d'arrêt.
                if hasattr(cand, "finish_reason"):
                    finish_reason_str = str(cand.finish_reason).upper()

                # Parcourir les parts et ignorer celles marquées "thought".
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        if getattr(part, "thought", False):
                            continue
                        if hasattr(part, "text") and part.text:
                            extracted_text += part.text

            # Récupérer le texte via les parts.
            if extracted_text.strip():
                extracted_text = extracted_text.strip()

                # Détection de troncature (MAX_TOKENS).
                if finish_reason_str and "MAX_TOKENS" in finish_reason_str:
                    print(f"     ⚠️  Réponse TRONQUÉE (MAX_TOKENS atteint, "
                          f"budget = {current_max_tokens}). La réponse peut "
                          f"être incomplète.")
                    current_max_tokens = min(current_max_tokens * 2, 16000)

                _LAST_SUCCESSFUL_MODEL = model_to_try
                return extracted_text

            # Fallback : response.text.
            if hasattr(response, "text") and response.text:
                text = response.text.strip()
                if text:
                    _LAST_SUCCESSFUL_MODEL = model_to_try
                    return text

            return "[ERREUR LLM] Réponse vide renvoyée par le modèle."

        # Fin de la boucle interne.
        # Si on arrive ici :
        #   - Soit on a fait un break sur TIMEOUT → on passe à la tentative
        #     globale suivante.
        #   - Soit on a épuisé les max_retries sur erreurs SDK → on passe
        #     aussi à la tentative globale suivante.
        final_error = last_error
        if not timeout_triggered_this_round:
            print(f"     ⚠️  Tentatives internes épuisées (dernière erreur : {last_error}).")

    # =========================================================================
    # ÉCHEC TOTAL après toutes les tentatives globales.
    # =========================================================================
    print(f"\n     ❌  ÉCHEC DÉFINITIF après {total_timeout_attempts} tentative(s) globale(s).")
    print(f"     🔌  Vérifiez votre connexion internet.")
    print(f"     🔌  Vérifiez également votre clé API et l'état du service Gemini :")
    print(f"         https://status.cloud.google.com/")
    print(f"     🔌  Dernière erreur : {final_error}")
    return (
        f"[ERREUR LLM] Échec après {total_timeout_attempts} tentative(s) globale(s) "
        f"de {LLM_TIMEOUT_SECONDS:.0f}s chacune. "
        f"Vérifiez votre connexion internet, votre clé API et l'état du service "
        f"Gemini (https://status.cloud.google.com/). "
        f"Détail : {final_error}"
    )


# -----------------------------------------------------------------------------
# C. CONTRÔLEUR MATHÉMATIQUE INDÉPENDANT
# -----------------------------------------------------------------------------

NUMBER_REGEX = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_numbers(text: str) -> list:
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
    Construit l'ensemble des valeurs numériques autorisées à partir du case_data.

    Sources :
      - Paramètres structurels : N, K, seed.
      - Identifiant du cas (ex. "#3" → 3).
      - Chiffres du scénario (ex. "S3" → 3, "S7" → 7).
      - Échelle de confiance 1-5.
      - Métriques : cost_initial, cost_final, conflicts, time_seconds,
        iterations, used_channels.
      - Baselines : cost, conflicts, time de chaque baseline.
      - Cellules en conflit (indices).
      - Positions (x, y) des cellules.
      - Topologie : poids des arêtes (i, j, w).
      - total_edges : NOMBRE TOTAL d'arêtes du réseau (utilisé dans le header
        du bloc TOPOLOGIE : "X arêtes au total"). Sans cette whitelist, le
        LLM qui cite ce nombre (ce qu'il fait naturellement pour décrire la
        topologie) serait faussement accusé d'halluciner.
      - Nombre d'arêtes affichées (len(topology_edges)) : utilisé dans le
        header lorsqu'il y a troncature ("80 arêtes affichées sur X au total").
      - Métadonnées : seed, threshold, N, K.
      - Affectations : canaux attribués (0 à K-1).
      - Résumé cellules : degré et force cumulée.
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

    # Identifiant du cas (ex. "#3" → 3)
    cid = str(case_data.get("case_id", "")).replace("#", "")
    if cid.isdigit():
        add(int(cid))

    # Chiffres du scénario (ex. "S3" → 3, "S7" → 7)
    scenario_str = str(case_data.get("scenario", ""))
    for num_str in re.findall(r"\d+", scenario_str):
        try:
            add(int(num_str))
        except ValueError:
            pass

    # Échelle de confiance 1-5 (par sécurité)
    for n in range(1, 6):
        add(n)

    m = case_data.get("metrics", {})
    for key in [
        "cost_initial", "cost_final", "conflicts",
        "time_seconds", "iterations", "used_channels",
    ]:
        if key in m:
            add(m[key])

    for base_name, base_data in case_data.get("baselines", {}).items():
        for k in ["cost", "conflicts", "time"]:
            if k in base_data:
                add(base_data[k])

    for c in case_data.get("conflicting_cells", []):
        add(c)

    # Positions (x, y)
    for cid_pos, xy in case_data.get("cell_positions", {}).items():
        try:
            add(int(cid_pos))
        except Exception:
            pass
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            add(xy[0])
            add(xy[1])

    # Poids des arêtes de la topologie
    for (i, j, w) in case_data.get("topology_edges", []):
        try:
            add(int(i))
            add(int(j))
            add(int(w))
        except Exception:
            pass

    # --- AJOUT CRITIQUE : total_edges (nombre total d'arêtes) ---
    # Ce champ est affiché dans le prompt sous la forme :
    #   "(X arêtes au total)"
    # ou "(Y arêtes affichées sur X au total ; ...)".
    # Si le LLM cite X (ou Y), il doit être whitelisté.
    if "total_edges" in case_data:
        add(case_data["total_edges"])

    # Nombre d'arêtes affichées (len(topology_edges)) : utile lorsqu'il y a
    # troncature (affichage partiel). Le LLM peut citer ce nombre dans le
    # header.
    if "topology_edges" in case_data:
        add(len(case_data["topology_edges"]))

    # Métadonnées de la topologie
    tm = case_data.get("topology_meta", {})
    for key in ["seed", "threshold", "N", "K"]:
        if key in tm:
            add(tm[key])

    # Affectations (canaux attribués à chaque cellule)
    allocations = case_data.get("allocations", {})
    for method, alloc_list in allocations.items():
        for c in alloc_list:
            add(c)

    # Résumé cellules (degré et force)
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
    """
    Régénération sous contrainte. On renvoie le PROMPT INITIAL COMPLET
    (car chaque appel est indépendant, pas de mémoire conversationnelle).
    """
    if not verification["has_hallucination"]:
        return llm_response

    faulty_str = "\n".join(
        f"  - valeur citée : {raw} (non présente dans les données)"
        for _, raw in verification["invented_numbers"]
    )

    m = case_data.get("metrics", {})
    model_label = case_data.get("model_label", "CCI-only")
    conflicts_label = _conflicts_label(model_label)

    allowed_summary = (
        f"Coût initial = {_fmt(m.get('cost_initial'))} ; "
        f"Coût final = {_fmt(m.get('cost_final'))} ; "
        f"{conflicts_label} = {_fmt(m.get('conflicts'))} ; "
        f"Temps = {_fmt(m.get('time_seconds'), 6)} s ; "
        f"Itérations = {_fmt(m.get('iterations'))} ; "
        f"Canaux utilisés = {_fmt(m.get('used_channels'))}."
    )

    correction_prompt = f"""
=== RECTIFICATION DEMANDÉE (REPRISE INTÉGRALE) ===

Nous te fournissons à nouveau la REQUÊTE ORIGINALE COMPLÈTE (car tu n'as pas
de mémoire entre nos échanges). Prends-en connaissance, puis corrige ta
réponse précédente qui contenait des ERREURS NUMÉRIQUES.

--- DÉBUT DE LA REQUÊTE ORIGINALE ---
{original_prompt}
--- FIN DE LA REQUÊTE ORIGINALE ---

Dans ta réponse précédente à cette requête, tu as cité les valeurs
numériques suivantes :

{faulty_str}

Ces valeurs NE FIGURENT PAS dans les données de simulation de la requête
originale. Ce sont donc des HALLUCINATIONS NUMÉRIQUES.

Rappel des données autorisées :
{allowed_summary}

CONSIGNES DE CORRECTION :
A) Reprends intégralement ta réponse en répondant aux MÊMES tâches
   (A) Résumé, B) Convergence, C) Temps d'exécution, D) Avis de confiance).
B) Remplace toute valeur incorrecte par la valeur exacte issue des données.
C) Si une information est absente, écris simplement :
   "Information non disponible dans les données fournies."
D) Utilise des LETTRES MAJUSCULES (A), B), C)...) pour énumérer.
E) RAPPEL DE LA RÈGLE DES NOMBRES :
   - Les RÉSULTATS de simulation (coût, conflits, temps, itérations,
     canaux) doivent être écrits EN CHIFFRES.
   - Tous les AUTRES nombres (numéros d'ordre, bornes d'échelle,
     note de confiance, etc.) doivent être écrits EN TOUTES LETTRES.
F) Ne cite PAS de "conflits CCI" ni de "conflits ACI" séparément si le mode
   est CCI+ACI. Utilise UNIQUEMENT le chiffre "{conflicts_label}" fourni.

=== RÉPONSE PRÉCÉDENTE À CORRIGER ===
{llm_response}
"""
    return ask_llm(correction_prompt, inter_request_delay=INTER_REQUEST_DELAY)


# -----------------------------------------------------------------------------
# D. GÉNÉRATION DE RAPPORT PDF PROFESSIONNEL
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
    """
    Génère un rapport PDF professionnel pour un cas LLM.

    NOTE : la section "6. Évaluation académique (grille 5 critères × 2 pts)"
    a été RETIRÉE. Les cotes sont attribuées manuellement par l'auteur.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        print("⚠️ reportlab non installé. Installez-le via : pip install reportlab")
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

    # ---- Titre ----
    story.append(Paragraph(
        f"Rapport d'audit LLM — Attribution de canaux ({model_label})",
        style_title
    ))
    story.append(Spacer(1, 0.3*cm))

    # ---- En-tête ----
    header_data = [
        ["Cas", case_data.get("case_id", "-")],
        ["Scénario", case_data.get("scenario", "-")],
        ["Configuration", f"N={case_data.get('N')}, K={case_data.get('K')}, seed={case_data.get('seed')}"],
        ["Mode d'interférence", model_label],
        ["Contexte", case_data.get("context", "-")],
        ["Modèle LLM", model_name if model_name else "Non spécifié"],
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

    # ---- Badge de conformité ----
    if verification_final["has_hallucination"]:
        badge = (
            f'<font color="#b30000"><b>⚠️ RAPPORT NON CERTIFIÉ</b></font> — '
            f'{len(verification_final["invented_numbers"])} valeur(s) non conforme(s) détectée(s).'
        )
    else:
        badge = (
            f'<font color="#006600"><b> RAPPORT CERTIFIÉ - 0 hallucination numérique</b></font> '
            f'(taux d\'exactitude : {verification_final["accuracy_rate"]:.1f}%).'
        )
    story.append(Paragraph(badge, style_body))
    story.append(Spacer(1, 0.4*cm))

    # ---- Données de simulation ----
    story.append(Paragraph("1. Données de simulation utilisées", style_h1))
    m = case_data.get("metrics", {})
    sim_data = [
        ["Métrique", "Valeur"],
        ["Coût initial J_0", _fmt(m.get("cost_initial"))],
        ["Coût final J(x)", _fmt(m.get("cost_final"))],
        [conflicts_label, _fmt(m.get("conflicts"))],
        ["Temps d'exécution (s)", _fmt(m.get("time_seconds"), 6)],
        ["Itérations (BD-CeNN)", _fmt(m.get("iterations"))],
        ["Canaux utilisés", _fmt(m.get("used_channels"))],
    ]
    sim_table = Table(sim_data, colWidths=[7*cm, 5*cm])
    sim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(sim_table)
    story.append(Spacer(1, 0.4*cm))

    # ---- Prompt ----
    story.append(Paragraph("2. Prompt envoyé au LLM", style_h1))
    prompt_escaped = prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(prompt_escaped.replace("\n", "<br/>"), style_mono))
    story.append(Spacer(1, 0.4*cm))

    # ---- Réponse brute ----
    story.append(Paragraph("3. Réponse brute du LLM (avant audit)", style_h1))
    raw_escaped = raw_response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(raw_escaped.replace("\n", "<br/>"), style_body))
    story.append(Spacer(1, 0.4*cm))

    # ---- Audit numérique ----
    story.append(Paragraph("4. Audit numérique automatique", style_h1))
    audit_data = [
        ["Indicateur", "Valeur"],
        ["Nombres extraits de la réponse", str(len(verification_initial["all_numbers"]))],
        ["Nombres conformes (initiaux)", str(len(verification_initial["valid_numbers"]))],
        ["Nombres inventés (initiaux)", str(len(verification_initial["invented_numbers"]))],
        ["Taux d'exactitude initial", f"{verification_initial['accuracy_rate']:.1f}%"],
        ["Hallucination détectée ?", "Oui" if verification_initial["has_hallucination"] else "Non"],
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

    if verification_initial["invented_numbers"]:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Valeurs détectées comme non conformes :", style_h2))
        for val, raw in verification_initial["invented_numbers"]:
            story.append(Paragraph(f"• <b>{raw}</b> (interprété comme {val})", style_body))

    story.append(Spacer(1, 0.4*cm))

    # ---- Réponse corrigée ----
    if verification_initial["has_hallucination"]:
        story.append(Paragraph("5. Réponse corrigée après régénération sous contrainte", style_h1))
        corr_escaped = corrected_response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(corr_escaped.replace("\n", "<br/>"), style_body))

        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Audit de la réponse corrigée :", style_h2))
        audit2 = [
            ["Nombres conformes après correction", str(len(verification_final["valid_numbers"]))],
            ["Nombres inventés après correction", str(len(verification_final["invented_numbers"]))],
            ["Taux d'exactitude final", f"{verification_final['accuracy_rate']:.1f}%"],
        ]
        audit2_table = Table(audit2, colWidths=[9*cm, 4*cm])
        audit2_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ]))
        story.append(audit2_table)
        story.append(Spacer(1, 0.4*cm))

    # NOTE : la section "6. Évaluation académique (grille 5 critères × 2 pts)"
    # a été RETIRÉE. Les cotes seront saisies manuellement dans le mémoire.

    # ---- Pied de page ----
    footer_text = (
        "<i>Rapport généré automatiquement dans le cadre du mémoire "
        "« Attribution de canaux/fréquences avec BD-CeNN + LLM » — "
        f"Nelson Ngombo, Bastion-Lab.<br/>"
        f"Mode d'interférence : <b>{model_label}</b>. "
        f"Modèle LLM utilisé : <b>{model_name if model_name else 'Non spécifié'}</b>."
        "</i>"
    )
    story.append(Paragraph(footer_text, style_body))

    doc.build(story)
    print(f"📄 Rapport PDF enregistré : {output_path}")


# -----------------------------------------------------------------------------
# E. FONCTION DE HAUT NIVEAU : audit complet d'un cas
# -----------------------------------------------------------------------------

def audit_case(case_data: dict, max_correction_attempts: int = 2,
               model_folder: str = "cochannel") -> dict:
    """
    Réalise l'audit complet d'un cas : prompt -> LLM -> vérification -> correction.

    NOTE : le score académique n'est plus calculé automatiquement. Les cotes
    seront attribuées manuellement par l'auteur dans le mémoire.
    """
    case_id = case_data.get("case_id", "unknown")
    model_label = case_data.get("model_label", "CCI-only")
    print(f"\n🔎 Audit du cas {case_id} ({case_data.get('scenario')}) [{model_label}]...")

    raw_dir, reports_dir = _get_mode_folders(model_label)

    prompt = build_prompt(case_data)

    prompt_file = raw_dir / f"{case_id}_prompt.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    raw_response = ask_llm(prompt)

    if raw_response.startswith("[ERREUR LLM]"):
        print(f"  ❌ Échec de l'appel LLM pour le cas {case_id}. Cas marqué comme LLM_FAILED.")
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
        print(f"  ⚠️ Hallucination détectée ({len(verification_final['invented_numbers'])} valeur(s)) "
              f"— tentative de correction {correction_attempts}/{max_correction_attempts}")
        corrected_response = regenerate_with_correction(
            case_data, prompt, corrected_response, verification_final
        )

        if corrected_response.startswith("[ERREUR LLM]"):
            print(f"  ❌ Échec de l'appel LLM lors de la correction du cas {case_id}. "
                  f"On conserve la réponse initiale.")
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
        model_name=GOOGLE_MODEL,
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