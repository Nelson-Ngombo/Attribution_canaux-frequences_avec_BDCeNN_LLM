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

INTER_REQUEST_DELAY = 5.0  # secondes

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
    """
    m = case_data.get("metrics", {})
    b = case_data.get("baselines", {})
    model_label = case_data.get("model_label", "CCI-only")

    baseline_lines = []
    for name in ["Random", "Greedy", "DSATUR"]:
        if name in b:
            baseline_lines.append(
                f"  - {name} : coût = {_fmt(b[name].get('cost'))}, "
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

=== CELLULES ENCORE EN CONFLIT ===
{conflicting_str}

=== TÂCHES DEMANDÉES (à énumérer avec des LETTRES : A), B), C)...) ===
A) Résume la qualité de la solution BD-CeNN en comparant le COÛT FINAL aux baselines.

B) Évalue la convergence en citant explicitement le nombre d'itérations avant convergence
C) Compare les PERFORMANCES DE CALCUL (TEMPS D'EXÉCUTION) :
   - Cite le temps d'exécution du BD-CeNN et celui de chaque baseline.
   - Si une baseline est PLUS RAPIDE que BD-CeNN, dis-le franchement ;
   - Si BD-CeNN est plus lent, explique pourquoi ?.
D) Dans le cas où les résultats du BD-CeNN ne sont pas fameux a tu une récommandation a faire en tant que Assitant de l'ingénieur Radio pour améliorer les résultats si les résultats sont satisfaisant tu peut nous dire "pas de recommandation à faire vu les bons résultats obtenues"

E) Termine par un avis de confiance : attribue une note de confiance
   sur une échelle à cinq niveaux (le niveau maximal étant le plus élevé),
   puis justifie-la brièvement. Écris la note EN TOUTES LETTRES
   (par exemple : "quatre sur cinq"), PAS en chiffres.

Rédige ton rapport de facon détaillé, long et Argumenté en français, de manière professionnelle et structurée.
Rappels :
- N'utilise JAMAIS de chiffres pour énumérer. Utilise A), B), C)...
- Tous les chiffres cités doivent provenir EXCLUSIVEMENT des données ci-dessus.
- Pour tout autre nombre (numéros d'ordre, bornes d'échelle), utilise des
  LETTRES (ex. "trois", "quatre sur cinq").
- Ne cite PAS de "conflits CCI" ni de "conflits ACI" séparément dans ce mode.
  Utilise UNIQUEMENT le chiffre "{conflicts_label}" fourni ci-dessus.
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


def ask_llm(prompt: str, timeout: int = 180, max_retries: int = 8,
            inter_request_delay: float = INTER_REQUEST_DELAY) -> str:
    """
    Interroge le LLM Google Gemini via le SDK `google-genai`.

    STRATÉGIE :
    1. PAUSE INTER-REQUÊTE (5s) avant chaque nouvelle requête.
    2. ROTATION DE MODÈLES sur 503/429/404.
    3. STICKY MODEL (mémorise le dernier modèle qui a marché).
    4. BACKOFF EXPONENTIEL pour les autres erreurs.
    5. BUDGET DE TOKENS ÉLARGI : 8000 (au lieu de 4000).
       Motif : les modèles 3.x génèrent des "thinking tokens" internes
       qui comptent dans le budget de sortie. Un budget trop bas tronque
       la réponse finale.
    6. FILTRAGE DES THOUGHTS : on ignore les parts marquées comme
       "thought" (raisonnement interne) et on ne garde que le texte final
       visible.
    7. DÉTECTION DE TRONCATURE : si finish_reason == "MAX_TOKENS", on
       retry immédiatement avec un budget doublé.
    """
    global _LAST_SUCCESSFUL_MODEL

    if inter_request_delay > 0:
        print(f"       Pause de {inter_request_delay:.1f}s avant la requête...")
        time.sleep(inter_request_delay)

    all_models = [GOOGLE_MODEL] + [m for m in GOOGLE_MODEL_FALLBACKS if m != GOOGLE_MODEL]
    if _LAST_SUCCESSFUL_MODEL and _LAST_SUCCESSFUL_MODEL in all_models:
        candidates = [_LAST_SUCCESSFUL_MODEL] + [m for m in all_models if m != _LAST_SUCCESSFUL_MODEL]
    else:
        candidates = all_models

    tried_models = set()
    last_error = None

    # Budget de tokens initial (élargi)
    current_max_tokens = 8000

    for attempt in range(1, max_retries + 1):
        model_to_try = None
        for c in candidates:
            if c not in tried_models:
                model_to_try = c
                break
        if model_to_try is None:
            tried_models.clear()
            model_to_try = candidates[0]

        try:
            response = _client.models.generate_content(
                model=model_to_try,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=current_max_tokens,
                    top_p=0.9,
                ),
            )

            # ---------------------------------------------------------------
            # EXTRACTION DU TEXTE FINAL (en ignorant les "thoughts")
            # ---------------------------------------------------------------
            # Les modèles Gemini 3.x génèrent des "thinking tokens" internes
            # qui apparaissent parfois dans `content.parts` avec un flag
            # `thought=True`. On DOIT les ignorer pour ne garder QUE la
            # réponse finale visible.
            extracted_text = ""
            finish_reason_str = None

            if response.candidates:
                cand = response.candidates[0]
                # Détecter la raison d'arrêt
                if hasattr(cand, "finish_reason"):
                    finish_reason_str = str(cand.finish_reason).upper()

                # Parcourir les parts et ignorer celles marquées "thought"
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        # Ignorer les parts "thought" (raisonnement interne)
                        is_thought = getattr(part, "thought", False)
                        if is_thought:
                            continue
                        # Extraire le texte des parts normales
                        if hasattr(part, "text") and part.text:
                            extracted_text += part.text

            # Si on a pu extraire du texte via les parts, on l'utilise
            if extracted_text.strip():
                extracted_text = extracted_text.strip()

                # ---------------------------------------------------------
                # DÉTECTION DE TRONCATURE
                # ---------------------------------------------------------
                # Si le modèle s'est arrêté pour cause de MAX_TOKENS, la
                # réponse est probablement tronquée. On l'accepte quand
                # même (pour ne pas perdre le travail déjà fait), MAIS on
                # log un avertissement visible.
                if finish_reason_str and "MAX_TOKENS" in finish_reason_str:
                    print(f"     ⚠️  Réponse TRONQUÉE (MAX_TOKENS atteint, "
                          f"budget = {current_max_tokens}). La réponse peut "
                          f"être incomplète.")
                    # Augmenter le budget pour la prochaine tentative
                    current_max_tokens = min(current_max_tokens * 2, 16000)

                _LAST_SUCCESSFUL_MODEL = model_to_try
                return extracted_text

            # Fallback : si response.text existe et n'est pas vide
            if hasattr(response, "text") and response.text:
                text = response.text.strip()
                if text:
                    _LAST_SUCCESSFUL_MODEL = model_to_try
                    return text

            return "[ERREUR LLM] Réponse vide renvoyée par le modèle."

        except Exception as e:
            last_error = f"{type(e).__name__} : {e}"
            error_str = str(e)
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

    return f"[ERREUR LLM] Échec après {max_retries} tentatives. Détail : {last_error}"


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
            f'<font color="#006600"><b>✅ RAPPORT CERTIFIÉ - 0 hallucination numérique</b></font> '
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