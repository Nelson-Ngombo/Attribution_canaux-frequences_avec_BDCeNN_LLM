# llm_assistant.py
"""
Module LLM pour le mémoire BD-CeNN + LLM.

Ce module contient :
    A. Le générateur de Prompt avec Système de Garde-fous (Zero-Hallucination Directive).
    B. L'interface avec le LLM (Google AI Studio / Gemini, via API).
    C. Le Contrôleur Mathématique Indépendant (Regex & Parser Garde-fou).
    D. La génération de rapports PDF professionnels.
    E. Le mécanisme de régénération sous contrainte (correction des hallucinations).

Auteur : Nelson Ngombo
"""

import os
import re
import json
import time
import requests
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. CONFIGURATION GLOBALE
# -----------------------------------------------------------------------------
# Charger les variables d'environnement depuis le fichier .env
from pathlib import Path
from dotenv import load_dotenv
import os

# Charger .env situé à la racine du projet
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Récupérer la clé API (variable d'environnement GOOGLE_API_KEY)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "❌ La variable GOOGLE_API_KEY est absente. "
        "Vérifiez votre fichier .env à la racine du projet."
    )

# Modèle Gemini à utiliser.
# On utilise l'alias "gemini-flash-latest" qui pointe automatiquement vers
# le dernier modèle Flash disponible pour la clé. Cela évite les 404 en cas
# de dépréciation (Google renomme ses modèles tous les 2-3 mois).
GOOGLE_MODEL_PREFERRED = "gemini-flash-latest"
GOOGLE_MODEL_FALLBACKS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]

# Dossiers de sortie
from config import LLM_LOGS_DIR, FIGURES_DIR, CSV_DIR

LLM_LOGS_DIR = Path(LLM_LOGS_DIR)
LLM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
LLM_REPORTS_DIR = LLM_LOGS_DIR / "reports_pdf"
LLM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LLM_RAW_DIR = LLM_LOGS_DIR / "raw_responses"
LLM_RAW_DIR.mkdir(parents=True, exist_ok=True)

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


# -----------------------------------------------------------------------------
# A. GÉNÉRATEUR DE PROMPT AVEC GARDE-FOUS
# -----------------------------------------------------------------------------

ZERO_HALLUCINATION_DIRECTIVE = """
=== DIRECTIVE SYSTÈME (ZERO-HALLUCINATION) ===
Tu es un assistant d'ingénierie radio. Ton rôle est d'auditer et d'expliquer
les résultats d'une simulation d'attribution de canaux.

RÈGLES STRICTES :
1. Tu ne dois JAMAIS inventer, altérer ou extrapoler un chiffre.
2. Tu dois citer EXCLUSIVEMENT les valeurs numériques présentes dans les données fournies.
3. Si une information n'est pas présente dans les données, écris simplement :
   "Information non disponible dans les données fournies."
4. Toute valeur numérique que tu cites doit pouvoir être retrouvée textuellement
   dans le bloc "DONNÉES DE SIMULATION" ci-dessous.
5. Si tu détectes une incohérence dans les données, signale-le sans inventer.
6. Rédige une analyse structurée et concise, comme un rapport d'ingénieur.
=== FIN DIRECTIVE ===
"""

def build_prompt(case_data: dict) -> str:
    """
    Construit un prompt structuré avec garde-fous pour un cas de simulation.

    Parameters
    ----------
    case_data : dict
        Dictionnaire contenant les informations du cas. Doit contenir :
        - 'case_id'      : identifiant du cas (#1..#20)
        - 'scenario'     : nom du scénario (S1..S7)
        - 'N'            : nombre de cellules
        - 'K'            : nombre de canaux
        - 'seed'         : seed de topologie
        - 'context'      : contexte (nominal, bruit 5%, etc.)
        - 'metrics'      : dict contenant les métriques du solveur BD-CeNN :
              * 'cost_initial'
              * 'cost_final'
              * 'conflicts_cochannel'
              * 'conflicts_adjacent'
              * 'time_seconds'
              * 'iterations'
              * 'used_channels'
        - 'baselines'    : dict avec les métriques des baselines :
              * 'Random'   : {'cost': .., 'conflicts': .., 'time': ..}
              * 'Greedy'   : {...}
              * 'DSATUR'   : {...}
        - 'conflicting_cells' : liste des cellules en conflit (optionnel)

    Returns
    -------
    prompt : str
    """
    m = case_data.get("metrics", {})
    b = case_data.get("baselines", {})

    # Construire le bloc des baselines
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

    prompt = f"""
{ZERO_HALLUCINATION_DIRECTIVE}

=== CONTEXTE DU CAS ===
- Cas : {case_data.get('case_id')}
- Scénario : {case_data.get('scenario')}
- Configuration : N = {case_data.get('N')} cellules, K = {case_data.get('K')} canaux
- Seed de topologie : {case_data.get('seed')}
- Contexte : {case_data.get('context')}

=== DONNÉES DE SIMULATION (SOLVEUR BD-CeNN) ===
- Coût initial J_0 : {_fmt(m.get('cost_initial'))}
- Coût final J(x) : {_fmt(m.get('cost_final'))}
- Nombre de conflits co-canal (CCI) : {_fmt(m.get('conflicts_cochannel'))}
- Nombre de conflits adjacent (ACI) : {_fmt(m.get('conflicts_adjacent'))}
- Temps d'exécution : {_fmt(m.get('time_seconds'), 6)} s
- Nombre d'itérations avant convergence : {_fmt(m.get('iterations'))}
- Canaux utilisés : {_fmt(m.get('used_channels'))}

=== BASELINES (COMPARAISON) ===
{baseline_str}

=== CELLULES ENCORE EN CONFLIT ===
{conflicting_str}

=== TÂCHES DEMANDÉES ===
1. Résume la qualité de la solution BD-CeNN en comparant le coût final aux baselines.
2. Identifie la cause probable des conflits restants (topologie, pénurie de canaux, minimum local).
3. Évalue la convergence (nombre d'itérations, temps d'exécution).
4. Donne une recommandation opérationnelle pour un ingénieur radio.
5. Termine par un avis de confiance (1-5) justifié.

Rédige ton rapport en français, de manière professionnelle et structurée.
N'oublie pas : tu ne dois citer AUCUN chiffre qui ne soit présent dans les données ci-dessus.
"""
    return prompt.strip()


# -----------------------------------------------------------------------------
# B. INTERFACE AVEC LE LLM (GOOGLE GENAI – SDK MODERNE)
# -----------------------------------------------------------------------------

# Initialiser le client Google GenAI une seule fois
from google import genai
from google.genai import types

_client = genai.Client(api_key=GOOGLE_API_KEY)


def _check_model_available(preferred: str, fallbacks: list) -> str:
    """
    Vérifie que `preferred` est utilisable pour cette clé API.
    Sinon, essaie dans l'ordre les modèles de `fallbacks`.
    Retourne le nom du premier modèle fonctionnel.

    NB : `client.models.list()` peut lister des modèles qui ne sont plus
    accessibles aux nouveaux utilisateurs. On vérifie donc par un appel réel.
    """
    candidates = [preferred] + [f for f in fallbacks if f != preferred]
    for c in candidates:
        try:
            # Test minimal : on demande une réponse triviale
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


# On détermine une bonne fois pour toutes le modèle à utiliser
GOOGLE_MODEL = _check_model_available(GOOGLE_MODEL_PREFERRED, GOOGLE_MODEL_FALLBACKS)
print(f"✅ Modèle Gemini retenu : {GOOGLE_MODEL}")


def ask_llm(prompt: str, timeout: int = 60, max_retries: int = 3) -> str:
    """
    Interroge le LLM Google Gemini via le nouveau SDK `google-genai`.

    Parameters
    ----------
    prompt : str
        Prompt complet à envoyer.
    timeout : int
        Délai d'attente (le SDK gère en interne, ce paramètre est indicatif).
    max_retries : int
        Nombre de tentatives en cas d'erreur transitoire.

    Returns
    -------
    response : str
        Texte généré par le LLM. En cas d'échec total, retourne un message d'erreur.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = _client.models.generate_content(
                model=GOOGLE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,          # faible pour limiter la créativité
                    max_output_tokens=1200,
                    top_p=0.9,
                ),
            )
            # Récupérer le texte
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            # Fallback si response.text est vide
            if response.candidates and response.candidates[0].content.parts:
                text = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
                if text:
                    return text.strip()
            return "[ERREUR LLM] Réponse vide renvoyée par le modèle."
        except Exception as e:
            last_error = f"{type(e).__name__} : {e}"
            print(f"  ⚠️ Tentative {attempt}/{max_retries} échouée : {last_error}")
            time.sleep(2 * attempt)   # backoff exponentiel
    return f"[ERREUR LLM] Échec après {max_retries} tentatives. Détail : {last_error}"


# -----------------------------------------------------------------------------
# C. CONTRÔLEUR MATHÉMATIQUE INDÉPENDANT (REGEX & PARSER GARDE-FOU)
# -----------------------------------------------------------------------------

# Regex pour capturer les entiers et flottants (positifs/négatifs, avec ou sans décimale)
NUMBER_REGEX = re.compile(r"-?\d+(?:[.,]\d+)?")

def extract_numbers(text: str) -> list:
    """
    Extrait tous les nombres (entiers et flottants) présents dans un texte.

    Retourne une liste de tuples (valeur_float, valeur_str_originale).
    Les virgules décimales sont converties en points.
    """
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
    On y ajoute aussi quelques valeurs structurelles (N, K, seed, case_id).
    """
    allowed = set()

    def add(v):
        try:
            fv = float(v)
            allowed.add(round(fv, 6))
        except Exception:
            pass

    # Paramètres structurels
    for key in ["N", "K", "seed"]:
        if key in case_data:
            add(case_data[key])

    # Identifiant du cas (ex. "#3" -> 3)
    cid = str(case_data.get("case_id", "")).replace("#", "")
    if cid.isdigit():
        add(int(cid))

    # Métriques du solveur
    m = case_data.get("metrics", {})
    for key in [
        "cost_initial", "cost_final",
        "conflicts_cochannel", "conflicts_adjacent",
        "time_seconds", "iterations", "used_channels",
    ]:
        if key in m:
            add(m[key])

    # Baselines
    for base_name, base_data in case_data.get("baselines", {}).items():
        for k in ["cost", "conflicts", "time"]:
            if k in base_data:
                add(base_data[k])

    # Cellules en conflit
    for c in case_data.get("conflicting_cells", []):
        add(c)

    return allowed


def verify_numbers(llm_response: str, case_data: dict, tolerance: float = 0.01) -> dict:
    """
    Contrôleur mathématique indépendant : vérifie que tous les nombres cités
    par le LLM sont présents (à la tolérance près) dans le case_data.

    Parameters
    ----------
    llm_response : str
        Texte généré par le LLM.
    case_data : dict
        Dictionnaire du cas (voir build_prompt).
    tolerance : float
        Tolérance relative pour la comparaison des flottants (1% par défaut).

    Returns
    -------
    report : dict avec :
        - 'all_numbers'          : liste de tous les nombres extraits
        - 'allowed_numbers'      : ensemble des nombres autorisés
        - 'valid_numbers'        : liste des nombres validés
        - 'invented_numbers'     : liste des nombres NON trouvés (hallucinations)
        - 'has_hallucination'    : booléen
        - 'accuracy_rate'        : taux d'exactitude numérique en %
    """
    extracted = extract_numbers(llm_response)
    allowed = _collect_allowed_numbers(case_data)

    valid = []
    invented = []

    for val, raw in extracted:
        # Vérifier la présence à tolérance près
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


def regenerate_with_correction(case_data: dict, llm_response: str, verification: dict) -> str:
    """
    Mécanisme de régénération sous contrainte : réinterroge le LLM en lui
    signalant explicitement les valeurs incorrectes détectées.

    Retourne la nouvelle réponse du LLM.
    """
    if not verification["has_hallucination"]:
        return llm_response

    # Construire la liste des valeurs fautives
    faulty_str = "\n".join(
        f"  - valeur citée : {raw} (non présente dans les données)"
        for _, raw in verification["invented_numbers"]
    )

    # Reconstruire un mini-rappel des données autorisées
    m = case_data.get("metrics", {})
    allowed_summary = (
        f"Coût initial = {_fmt(m.get('cost_initial'))} ; "
        f"Coût final = {_fmt(m.get('cost_final'))} ; "
        f"Conflits CCI = {_fmt(m.get('conflicts_cochannel'))} ; "
        f"Conflits ACI = {_fmt(m.get('conflicts_adjacent'))} ; "
        f"Temps = {_fmt(m.get('time_seconds'), 6)} s ; "
        f"Itérations = {_fmt(m.get('iterations'))} ; "
        f"Canaux utilisés = {_fmt(m.get('used_channels'))}."
    )

    correction_prompt = f"""
=== RECTIFICATION DEMANDÉE ===
Dans ta réponse précédente, tu as cité les valeurs numériques suivantes
qui NE FIGURENT PAS dans les données fournies :

{faulty_str}

RAPPEL DES DONNÉES AUTORISÉES :
{allowed_summary}

Consigne : réécris intégralement ton rapport en remplaçant toute valeur
incorrecte par la valeur exacte issue des données. Si une information
est absente, écris "Information non disponible dans les données fournies."

Ta nouvelle réponse doit contenir UNIQUEMENT des chiffres autorisés.

=== RÉPONSE PRÉCÉDENTE À CORRIGER ===
{llm_response}
"""
    return ask_llm(correction_prompt)


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
    score_total: float,
    score_details: dict,
    output_path: Path,
    model_name: str = None,   # ← nom du modèle LLM utilisé
) -> None:
    """
    Génère un rapport PDF professionnel pour un cas LLM.

    Le rapport contient :
        - En-tête avec le cas, le scénario et la date.
        - Données de simulation utilisées (table).
        - Prompt envoyé au LLM.
        - Réponse brute du LLM.
        - Résultats de l'audit numérique (inventions, taux d'exactitude).
        - Réponse corrigée (si applicable).
        - Score académique (5 critères x 2 pts).
        - Badge de conformité.
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

    # ---- Titre ----
    story.append(Paragraph("Rapport d'audit LLM - Attribution de canaux", style_title))
    story.append(Spacer(1, 0.3*cm))

    # ---- En-tête ----
    header_data = [
        ["Cas", case_data.get("case_id", "-")],
        ["Scénario", case_data.get("scenario", "-")],
        ["Configuration", f"N={case_data.get('N')}, K={case_data.get('K')}, seed={case_data.get('seed')}"],
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
        ["Conflits co-canal (CCI)", _fmt(m.get("conflicts_cochannel"))],
        ["Conflits adjacent (ACI)", _fmt(m.get("conflicts_adjacent"))],
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

        # Audit de la réponse corrigée
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

    # ---- Score académique ----
    story.append(Paragraph("6. Évaluation académique (grille 5 critères × 2 pts)", style_h1))
    score_data = [["Critère", "Note /2"]]
    for crit, val in score_details.items():
        score_data.append([crit, str(val)])
    score_data.append(["TOTAL", f"{score_total:.1f} / 10"])
    score_table = Table(score_data, colWidths=[9*cm, 4*cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8eef7")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.5*cm))

    # ---- Pied de page ----
    footer_text = (
        "<i>Rapport généré automatiquement dans le cadre du mémoire "
        "« Attribution de canaux/fréquences avec BD-CeNN + LLM » — "
        f"Nelson Ngombo, Bastion-Lab.<br/>"
        f"Modèle LLM utilisé : <b>{model_name if model_name else 'Non spécifié'}</b>."
        "</i>"
    )
    story.append(Paragraph(footer_text, style_body))

    doc.build(story)
    print(f"📄 Rapport PDF enregistré : {output_path}")


# -----------------------------------------------------------------------------
# E. FONCTION DE HAUT NIVEAU : audit complet d'un cas
# -----------------------------------------------------------------------------

def audit_case(case_data: dict, max_correction_attempts: int = 2) -> dict:
    """
    Réalise l'audit complet d'un cas : prompt -> LLM -> vérification -> correction.

    Retourne un dictionnaire contenant toutes les informations du cas.
    """
    case_id = case_data.get("case_id", "unknown")
    print(f"\n🔎 Audit du cas {case_id} ({case_data.get('scenario')})...")

    # 1. Construire le prompt
    prompt = build_prompt(case_data)

    # Sauvegarder le prompt brut
    prompt_file = LLM_RAW_DIR / f"{case_id}_prompt.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    # 2. Interroger le LLM
    raw_response = ask_llm(prompt)

    # Sauvegarder la réponse brute
    raw_file = LLM_RAW_DIR / f"{case_id}_raw_response.txt"
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(raw_response)

    # 3. Vérification numérique initiale
    verification_initial = verify_numbers(raw_response, case_data)

    # 4. Mécanisme de régénération sous contrainte (si nécessaire)
    corrected_response = raw_response
    verification_final = verification_initial
    correction_attempts = 0
    while verification_final["has_hallucination"] and correction_attempts < max_correction_attempts:
        correction_attempts += 1
        print(f"  ⚠️ Hallucination détectée ({len(verification_final['invented_numbers'])} valeur(s)) "
              f"— tentative de correction {correction_attempts}/{max_correction_attempts}")
        corrected_response = regenerate_with_correction(case_data, corrected_response, verification_final)
        verification_final = verify_numbers(corrected_response, case_data)
        # Sauvegarder chaque tentative corrigée
        corr_file = LLM_RAW_DIR / f"{case_id}_corrected_v{correction_attempts}.txt"
        with open(corr_file, "w", encoding="utf-8") as f:
            f.write(corrected_response)

    # 5. Calcul du score académique (5 critères × 2 pts)
    score_details = compute_academic_score(
        case_data, corrected_response, verification_final
    )
    score_total = sum(score_details.values())

    # 6. Générer le rapport PDF
    pdf_path = LLM_REPORTS_DIR / f"rapport_{case_id}_{case_data.get('scenario')}.pdf"
    generate_pdf_report(
        case_data=case_data,
        prompt=prompt,
        raw_response=raw_response,
        verification_initial=verification_initial,
        corrected_response=corrected_response,
        verification_final=verification_final,
        score_total=score_total,
        score_details=score_details,
        output_path=pdf_path,
        model_name=GOOGLE_MODEL,   # passage du modèle utilisé
    )

    return {
        "case_id": case_id,
        "scenario": case_data.get("scenario"),
        "N": case_data.get("N"),
        "K": case_data.get("K"),
        "seed": case_data.get("seed"),
        "context": case_data.get("context"),
        "raw_response": raw_response,
        "corrected_response": corrected_response,
        "verification_initial": verification_initial,
        "verification_final": verification_final,
        "correction_attempts": correction_attempts,
        "score_total": score_total,
        "score_details": score_details,
        "pdf_path": str(pdf_path),
    }


# -----------------------------------------------------------------------------
# F. SCORING ACADÉMIQUE AUTOMATIQUE (5 critères × 2 pts)
# -----------------------------------------------------------------------------

def compute_academic_score(case_data: dict, response: str, verification: dict) -> dict:
    """
    Calcule un score sur 10 points (5 critères × 2 pts) :
        - Exactitude numérique (2 pts) : taux d'exactitude >= 99% -> 2 ; >= 95% -> 1 ; sinon 0.
        - Cohérence (2 pts) : aucune contradiction détectable -> 2 ; mineure -> 1 ; sinon 0.
        - Clarté (2 pts) : structure en paragraphes/numérotée -> 2 ; sinon 1 ou 0.
        - Utilité ingénieur (2 pts) : présence de recommandations -> 2 ; sinon 0-1.
        - Traçabilité (2 pts) : mention explicite des métriques -> 2 ; sinon 0-1.

    NB : il s'agit d'une heuristique automatique. Une évaluation humaine reste recommandée.
    """
    score = {}

    # --- 1. Exactitude numérique ---
    if verification["accuracy_rate"] >= 99.0:
        score["Exactitude numérique"] = 2
    elif verification["accuracy_rate"] >= 95.0:
        score["Exactitude numérique"] = 1
    else:
        score["Exactitude numérique"] = 0

    # --- 2. Cohérence (recherche de contradictions basiques) ---
    response_lower = response.lower()
    contradiction_markers = ["contradiction", "incohérent", "impossible"]
    has_contradiction = any(m in response_lower for m in contradiction_markers)
    # Vérifier que le coût final est bien présenté comme < coût initial si applicable
    m = case_data.get("metrics", {})
    c_init = m.get("cost_initial")
    c_final = m.get("cost_final")
    coherent = True
    if c_init is not None and c_final is not None:
        # Si le LLM affirme que le coût a augmenté alors que c_final < c_init, c'est incohérent
        if c_final < c_init and "augment" in response_lower and "coût" in response_lower:
            coherent = False
    if coherent and not has_contradiction:
        score["Cohérence"] = 2
    elif has_contradiction:
        score["Cohérence"] = 0
    else:
        score["Cohérence"] = 1

    # --- 3. Clarté ---
    # Un rapport clair a au moins 3 paragraphes ou une structure numérotée
    paragraphs = [p for p in response.split("\n\n") if len(p.strip()) > 30]
    has_numbering = bool(re.search(r"(?:^|\n)\s*\d+[\.\)]\s", response))
    if len(paragraphs) >= 3 or has_numbering:
        score["Clarté"] = 2
    elif len(paragraphs) >= 1:
        score["Clarté"] = 1
    else:
        score["Clarté"] = 0

    # --- 4. Utilité ingénieur ---
    # Recherche de mots-clés liés aux recommandations
    rec_markers = ["recommand", "suggèr", "propos", "il est conseillé", "action", "solution"]
    has_recommendation = any(m in response_lower for m in rec_markers)
    if has_recommendation:
        score["Utilité ingénieur"] = 2
    elif "analyse" in response_lower or "cause" in response_lower:
        score["Utilité ingénieur"] = 1
    else:
        score["Utilité ingénieur"] = 0

    # --- 5. Traçabilité ---
    # Vérifier que le rapport cite au moins 3 métriques distinctes parmi celles attendues
    m = case_data.get("metrics", {})
    expected_keys = [
        ("coût final", "cost_final"),
        ("conflit", "conflicts_cochannel"),
        ("temps", "time_seconds"),
        ("itération", "iterations"),
        ("canaux", "used_channels"),
    ]
    cited_count = 0
    for kw, key in expected_keys:
        if key in m and kw in response_lower:
            cited_count += 1
    if cited_count >= 3:
        score["Traçabilité"] = 2
    elif cited_count >= 1:
        score["Traçabilité"] = 1
    else:
        score["Traçabilité"] = 0

    return score