#validate_results.py
"""
Module de validation et de certification statistique autonome du framework BD-CeNN.

Ce script est appele automatiquement a la fin de chaque experience (E1 a E10).
Il effectue un audit complet de coherence sur l'ensemble des fichiers CSV :
    1. Classification intelligente des fichiers (summary, metrics, allocations, raw).
    2. Recalcul independant et certifie de l'energie J(x) et des conflits.
    3. Verification des invariants statistiques et structurels (sans duplication).
    4. Generation d'un rapport PDF academique (ReportLab) avec division
       automatique des seuls tableaux larges (> 11 colonnes) en sous-blocs.

Auteur : Nelson MUKABI NGOMBO -- Polytechnique, Universite de Kinshasa
"""

import os
import re
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

import config
from metrics import (
    compute_cost_cci,
    compute_cost_cci_aci,
    count_conflicts_cci,
    count_conflicts_aci,
    create_channel_interference_matrix,
)

# Configuration de ReportLab pour la generation PDF
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================================
# 1. STRUCTURE DES RESULTATS DE VERIFICATION
# ============================================================================

class VerificationResult:
    """Structure contenant le statut d'evaluation d'un invariant individuel."""
    def __init__(self, code: str, desc: str, success: bool, details: str = ""):
        self.code = code
        self.desc = desc
        self.success = success
        self.details = details


# ============================================================================
# 2. SEPARATION SEMANTIQUE DES FLUX CSV
# ============================================================================

def classify_csv_files(csv_dir: Path) -> dict:
    """
    Classe les fichiers CSV presents dans le dossier d'une experience par type.

    Parameters
    ----------
    csv_dir : Path
        Dossier contenant les resultats de l'experience.

    Returns
    -------
    dict
        Dictionnaire trie par categorie semantique.
    """
    classified = {
        "summary": [],
        "metrics": [],
        "allocations": [],
        "raw": [],
        "trajectory": [],
        "llm_reports": [],  # [FIX-E10] CSV du verificateur LLM, non affiche dans le PDF d'audit
        "other": []
    }

    for csv_path in sorted(csv_dir.glob("*.csv")):
        fname = csv_path.name.lower()

        if "summary" in fname:
            classified["summary"].append(csv_path)
        elif "method_metrics" in fname or "convergence_metrics" in fname:
            classified["metrics"].append(csv_path)
        elif "cell_channels" in fname:
            classified["allocations"].append(csv_path)
        elif "raw" in fname:
            classified["raw"].append(csv_path)
        elif "trajectory" in fname:
            classified["trajectory"].append(csv_path)
        # [FIX-E10] Le CSV llm_fidelity_evaluation.csv NE DOIT PLUS etre
        # inclus dans le rapport d'audit validate_results.py. Ce rapport
        # valide uniquement les metriques du solveur. Le tableau de fidelite
        # LLM figure dans le rapport PDF de chaque cas, produit par
        # llm_assistant.generate_pdf_report.
        elif "llm_fidelity" in fname:
            classified["llm_reports"].append(csv_path)

        # [FIX-E10] CSV dedie aux metriques du solveur pour E10
        elif "solver_metrics" in fname:
            classified["metrics"].append(csv_path)
        else:
            classified["other"].append(csv_path)

    return classified


def _split_wide_dataframe(df: pd.DataFrame, max_cols: int = 11) -> list:
    """
    Decoupe un DataFrame ayant trop de colonnes en plusieurs sous-blocs coherents.
    Les colonnes d'identification sont repetees en entete de chaque bloc.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame d'entree.
    max_cols : int, optionnel
        Seuil de colonnes a partir duquel declencher le decoupage (defaut 11).

    Returns
    -------
    list of tuple (pd.DataFrame, str)
        Liste des sous-blocs et leur suffixe d'identification (a), (b), etc.
    """
    id_candidates = [
        "N", "K", "density_label", "threshold", "method", "Method",
        "scenario", "Scenario", "noise_level", "mod_level", "value",
        "Case", "Scenario (N, K, seed)", "seed", "Cell", "n_seeds",
    ]
    id_cols = [c for c in df.columns if c in id_candidates]
    id_cols = id_cols[:5]

    # [FIX-E7] Ajout du prefixe "change_" pour capturer les groupes
    # mean/std/median/min/max des experiences E7 et E8.
    groups = [
        [c for c in df.columns if c.startswith("cost_") or c.startswith("norm_cost_")],
        [c for c in df.columns if c.startswith("conflicts_")],
        [c for c in df.columns if c.startswith("change_") or c.startswith("changes_")],
        [c for c in df.columns if (c.startswith("used_channels") 
                                   or c.startswith("time_") 
                                   or c.startswith("n_sweeps_") 
                                   or c.startswith("normalized_"))]
    ]

    used = set(id_cols)
    for g in groups:
        used.update(g)
    others = [c for c in df.columns if c not in used]
    if others:
        groups.append(others)

    groups = [g for g in groups if g]

    # Ne jamais decouper un tableau qui tient sous le seuil maximal de colonnes
    if len(df.columns) <= max_cols:
        return [(df, "")]

    blocks = []
    for i, g in enumerate(groups):
        sub_cols = [c for c in id_cols if c in df.columns] + [c for c in g if c in df.columns]
        sub_df = df[sub_cols].copy()
        suffix = f" ({chr(97 + i)})"
        blocks.append((sub_df, suffix))

    return blocks


# ============================================================================
# 3. RECALCUL ET COMPARAISON INDEPENDANTE
# ============================================================================

def recompute_and_compare(df_data: pd.DataFrame, mode: str, K: int,
                           W_dict: dict, cutoff: int = 2) -> list:
    """
    Execute le recalcul de chaque vecteur d'allocation et compare les couts.
    """
    anomalies = []
    M = create_channel_interference_matrix(K) if mode == "adjacent" else None

    for idx, row in df_data.iterrows():
        seed_val = int(row["seed"]) if "seed" in row and pd.notna(row["seed"]) else None
        method_name = row.get("method", row.get("Methode", row.get("Method", f"Ligne_{idx}")))

        x = None
        if "assignment" in row and pd.notna(row["assignment"]):
            try:
                x = np.array(json.loads(row["assignment"]), dtype=int)
            except Exception:
                pass

        if x is None:
            continue

        W = W_dict.get(seed_val)
        if W is None:
            continue

        if mode == "cochannel":
            cost_rec = compute_cost_cci(x, W)
            cci_rec = count_conflicts_cci(x, W)
            aci_rec = 0
        else:
            cost_rec = compute_cost_cci_aci(x, W, M)
            cci_rec = count_conflicts_cci(x, W)
            aci_rec = count_conflicts_aci(x, W, cutoff=cutoff)

        total_rec = cci_rec + aci_rec

        # Verification du cout enregistre
        cost_saved = float(row["cost"]) if "cost" in row and pd.notna(row["cost"]) else None
        if cost_saved is None and "Cout" in row and pd.notna(row["Cout"]):
            cost_saved = float(row["Cout"])

        if cost_saved is not None and abs(cost_rec - cost_saved) > 1e-3:
            anomalies.append(
                f"Ligne {idx} ({method_name}): Ecart de cout detecte "
                f"(Recalcule: {cost_rec:.4f}, Enregistre: {cost_saved:.4f})"
            )

        # Verification conflicts_cci
        if "conflicts_cci" in row and pd.notna(row["conflicts_cci"]):
            cci_saved = int(row["conflicts_cci"])
            if cci_rec != cci_saved:
                anomalies.append(
                    f"Ligne {idx} ({method_name}): Ecart conflicts_cci "
                    f"(Recalcule: {cci_rec}, Enregistre: {cci_saved})"
                )

        # Verification conflicts_aci
        if "conflicts_aci" in row and pd.notna(row["conflicts_aci"]):
            aci_saved = int(row["conflicts_aci"])
            if aci_rec != aci_saved:
                anomalies.append(
                    f"Ligne {idx} ({method_name}): Ecart conflicts_aci "
                    f"(Recalcule: {aci_rec}, Enregistre: {aci_saved})"
                )

        # Verification conflicts_total
        tot_saved = None
        if "conflicts_total" in row and pd.notna(row["conflicts_total"]):
            tot_saved = int(row["conflicts_total"])
        elif "conflicts" in row and pd.notna(row["conflicts"]):
            tot_saved = int(row["conflicts"])

        if tot_saved is not None and total_rec != tot_saved:
            anomalies.append(
                f"Ligne {idx} ({method_name}): Ecart conflicts_total "
                f"(Recalcule: {total_rec}, Enregistre: {tot_saved})"
            )

    return anomalies


# ============================================================================
# 4. COMPOSANTS DE RENDU ET MISE EN PAGE DU PDF
# ============================================================================

# [FIX-E7] Correspondance suffixe statistique -> libelle court.
# Utilise pour construire les en-tetes a 2 niveaux (groupes / sous-en-tetes).
_STAT_SUFFIX_LABELS = {
    "mean": "moy.",
    "std": "é.t.",
    "median": "méd.",
    "min": "min",
    "max": "max",
}

# [FIX-E7] [FIX-E8] Correspondance prefixe -> libelle de groupe.
# Permet de nommer les colonnes fusionnees dans la premiere ligne d'en-tete.
# E8 utilise un override specifique (voir build_pdf_report) pour "Temps d'adaptation",
# "Nb. réaffectations canaux" et "Coût final".
_GROUP_HEADER_LABELS = {
    "cost_rel": "Coût relatif (%)",
    "change": "Taux de changement (%)",
    "changes": "Nb. réaffectations canaux",  # [FIX-E8] libelle enrichi
    "cost": "Coût J(x)",
    "cost_initial": "J_init",
    "cost_final": "J_final",
    "normalized_cost": "Coût normalisé",
    "conflicts_total": "Conflits totaux",
    "conflicts_cci": "Conflits C_CCI",
    "conflicts_aci": "Conflits C_ACI",
    "used_channels": "Canaux utilisés",
    "time": "Temps (ms)",
    "n_sweeps": "Sweeps",
}


def _detect_column_groups(cols, labels_override=None):
    """
    [FIX-E7] Detecte les sequences de colonnes consecutives partageant un
    prefixe statistique commun (ex. cost_rel_mean, cost_rel_std, ...).

    [FIX-E8] Ajout d'un parametre labels_override pour permettre des
    libelles specifiques a certaines experiences (E8 par exemple).

    Parameters
    ----------
    cols : list of str
        Noms de colonnes du DataFrame.
    labels_override : dict or None
        Dictionnaire optionnel pour remplacer certains libelles de groupe.

    Returns
    -------
    groups : list of dict
        Chaque dict contient : 'start', 'end', 'prefix', 'label', 'suffixes'.
    parsed : list of tuple or None
        Resultat du parsing de chaque colonne : (prefix, suffix) ou None.
    """
    labels = dict(_GROUP_HEADER_LABELS)
    if labels_override:
        labels.update(labels_override)

    def _parse(col):
        for suffix in _STAT_SUFFIX_LABELS:
            if col.endswith(f"_{suffix}"):
                return col[:-len(suffix) - 1], suffix
        return None

    parsed = [_parse(c) for c in cols]
    groups = []
    i = 0
    n = len(cols)
    while i < n:
        if parsed[i] is None:
            i += 1
            continue
        prefix, _ = parsed[i]
        j = i
        while j < n and parsed[j] is not None and parsed[j][0] == prefix:
            j += 1
        # Un groupe necessite au moins 2 colonnes consecutives
        if j - i >= 2:
            group_label = labels.get(prefix, prefix)
            groups.append({
                "start": i,
                "end": j - 1,
                "prefix": prefix,
                "label": group_label,
                "suffixes": [parsed[k][1] for k in range(i, j)],
            })
        i = j
    return groups, parsed


def _format_cell_value(val, col_name: str) -> str:
    """Formate une valeur de cellule selon la nomenclature du projet."""
    if pd.isna(val):
        return "-"

    # [FIX-E8] Les colonnes d'identification (mod_level, noise_level)
    # s'affichent comme des entiers sans decimale.
    if col_name in ("mod_level", "noise_level"):
        try:
            return f"{int(round(float(val)))}"
        except Exception:
            return str(val)

    # [FIX-E9] La colonne "value" (Nombre d'essais) s'affiche en entier.
    if col_name == "value":
        try:
            return f"{int(round(float(val)))}"
        except Exception:
            return str(val)

    if isinstance(val, float):
        if "cost" in col_name.lower() or "cout" in col_name.lower():
            return f"{val:.2f}"
        elif "time" in col_name.lower():
            # [FIX-E8] Suppression du suffixe "ms" : l'unite figure dans
            # l'en-tete de groupe ("Temps d'adaptation (ms)").
            # [FIX-E9] Passage a 3 decimales pour respecter le format E9.
            return f"{val * 1000:.3f}"
        elif "changes" in col_name.lower():
            # [FIX-E8] Colonnes "changes_mean", "changes_std", ... en 2 decimales.
            return f"{val:.2f}"
        elif "percent" in col_name.lower() or "rate" in col_name.lower() or "rel" in col_name.lower():
            return f"{val:.1f}%"
        elif "normalized" in col_name.lower() or "norm" in col_name.lower():
            return f"{val:.4f}"
        else:
            return f"{val:.1f}"
    elif isinstance(val, (int, np.integer)):
        return str(int(val))
    else:
        return str(val)


def _abbreviate_column(col: str) -> str:
    """Transforme un en-tete technique en abreviation normalisee."""
    replacements = [
        ("cost_initial_mean", "Moy. J_init"),
        ("cost_initial_std", "E.T. J_init"),
        ("cost_final_mean", "Moy. J_final"),
        ("cost_final_std", "E.T. J_final"),
        ("cost_mean", "Moy. Cout"),
        ("cost_std", "E.T. Cout"),
        ("cost_min", "Min Cout"),
        ("cost_max", "Max Cout"),
        ("cost_median", "Med. Cout"),
        ("conflicts_total_mean", "Moy. C_tot"),
        ("conflicts_total_std", "E.T. C_tot"),
        ("conflicts_total_min", "Min C_tot"),
        ("conflicts_total_max", "Max C_tot"),
        ("conflicts_total_median", "Med. C_tot"),
        ("conflicts_cci_mean", "Moy. C_CCI"),
        ("conflicts_cci_std", "E.T. C_CCI"),
        ("conflicts_aci_mean", "Moy. C_ACI"),
        ("conflicts_aci_std", "E.T. C_ACI"),
        ("used_channels_mean", "Moy. K_used"),
        ("used_channels_std", "E.T. K_used"),
        ("used_channels_min", "Min K"),
        ("used_channels_max", "Max K"),
        ("time_mean", "Moy. t_exec"),
        ("time_std", "E.T. t_exec"),
        ("n_sweeps_mean", "Moy. Sweeps"),
        ("n_sweeps_std", "E.T. Sweeps"),
        ("n_sweeps_min", "Min Sw."),
        ("n_sweeps_max", "Max Sw."),
        ("n_sweeps_median", "Med. Sw."),
        ("normalized_cost_mean", "Moy. J_norm"),
        ("normalized_cost_std", "E.T. J_norm"),
        ("density_label", "Densite"),
        ("threshold", "Seuil"),
        ("method", "Methode"),
        ("Method", "Methode"),
        ("conflicts_cci", "C_CCI"),
        ("conflicts_aci", "C_ACI"),
        ("conflicts_total", "C_tot"),
        ("used_channels", "K_used"),
        ("Cout", "Cout J(x)"),
        ("seed", "Seed"),
        ("noise_level", "Bruit (%)"),
        # [FIX-E8] Libelle enrichi pour mod_level (utilise dans E8)
        ("mod_level", "Taux de modif. (%)"),
        # [FIX-E9] Libelle enrichi pour "value" (utilise dans E9)
        ("value", "Nombre d'essais"),
        # [FIX-E7] Abreviations des colonnes de degradation relative (E7)
        ("cost_rel_mean", "Moy. Degrad."),
        ("cost_rel_std", "E.T. Degrad."),
        ("cost_rel_median", "Med. Degrad."),
        ("cost_rel_min", "Min Degrad."),
        ("cost_rel_max", "Max Degrad."),
        # [FIX-E7] Abreviations des colonnes de taux de changement (E7)
        ("change_mean", "Moy. Reaff."),
        ("change_std", "E.T. Reaff."),
        ("change_median", "Med. Reaff."),
        ("change_min", "Min Reaff."),
        ("change_max", "Max Reaff."),
        ("scenario", "Scenario"),
        ("n_seeds", "n_seeds"),
        ("Scenario (N, K, seed)", "Scenario (N,K,seed)"),
    ]
    result = col
    for old, new in replacements:
        if old in result:
            result = result.replace(old, new)
            break
    return result


# [FIX-E9] Groupes metriques ordonnes tels qu'attendus dans les tableaux E9.
# Chaque entree : (prefixe_colonnes, libelle_affichage)
_E9_METRIC_GROUPS = [
    ("cost",             "Coût"),
    ("conflicts_total",  "Conflit"),
    ("time",             "Temps d'exécution (ms)"),
]


def _render_e9_table(sub_df, method_name, table_counter,
                     style_h2, style_table_header, style_table_cell,
                     table_width, filename):
    """
    [FIX-E9] Construit un tableau a 3 niveaux d'en-tetes specifique a E9 :
        Niveau 1 : nom de la methode (fusion horizontale sur toutes les colonnes
                   de metriques, la colonne "Nombre d'essais" reste a gauche)
        Niveau 2 : groupes metriques (Coût | Conflit | Temps d'exécution (ms))
        Niveau 3 : suffixes statistiques (moy. | e.t. | med. | min | max)

    Parameters
    ----------
    sub_df : pd.DataFrame
        Sous-DataFrame filtre sur une methode donnee (colonne "method" retiree).
    method_name : str
        Nom de la methode ("Greedy" ou "BD-CeNN").
    table_counter : int
        Index sequentiel du tableau (pour les styles uniques).
    style_h2, style_table_header, style_table_cell : ParagraphStyle
        Styles ReportLab heritees du document.
    table_width : float
        Largeur totale disponible en points.
    filename : str
        Nom du fichier source (pour le sous-titre).

    Returns
    -------
    list of flowable
        Liste des flowables (titre + tableau) a ajouter a l'histoire.
    """
    # Colonne d'identification a gauche
    id_cols = [c for c in ["value"] if c in sub_df.columns]

    # Colonnes de donnees ordonnees par groupe metrique
    data_cols = []
    group_spans = []  # (start_idx_absolu, end_idx_absolu, label)
    for prefix, label in _E9_METRIC_GROUPS:
        cols_for_group = [
            f"{prefix}_{s}"
            for s in ["mean", "std", "median", "min", "max"]
            if f"{prefix}_{s}" in sub_df.columns
        ]
        if cols_for_group:
            start = len(id_cols) + len(data_cols)
            data_cols.extend(cols_for_group)
            end = len(id_cols) + len(data_cols) - 1
            group_spans.append((start, end, label))

    all_cols = id_cols + data_cols
    n_cols = len(all_cols)

    # Styles adaptes a la densite de colonnes
    if n_cols > 10:
        cell_style = ParagraphStyle(
            f"CellE9_{table_counter}", parent=style_table_cell,
            fontSize=6.5, leading=8.5
        )
        header_style = ParagraphStyle(
            f"HeaderE9_{table_counter}", parent=style_table_header,
            fontSize=7, leading=9
        )
    else:
        cell_style = style_table_cell
        header_style = style_table_header

    # --- Ligne 1 : "Nombre d'essais" (vertical span) + nom de methode (horiz span)
    row1 = [Paragraph("", header_style) for _ in range(n_cols)]
    if id_cols:
        row1[0] = Paragraph("<b>Nombre d'essais</b>", header_style)
    method_span_start = len(id_cols)
    method_span_end = n_cols - 1
    if method_span_end >= method_span_start:
        row1[method_span_start] = Paragraph(f"<b>{method_name}</b>", header_style)

    # --- Ligne 2 : groupes metriques
    row2 = [Paragraph("", header_style) for _ in range(n_cols)]
    for start, end, label in group_spans:
        row2[start] = Paragraph(f"<b>{label}</b>", header_style)

    # --- Ligne 3 : suffixes statistiques
    row3 = [Paragraph("", header_style) for _ in range(n_cols)]
    for k, col_name in enumerate(all_cols):
        if col_name in id_cols:
            row3[k] = Paragraph("", header_style)
        else:
            suffix = col_name.rsplit("_", 1)[-1]
            row3[k] = Paragraph(
                f"<b>{_STAT_SUFFIX_LABELS.get(suffix, suffix)}</b>",
                header_style
            )

    table_rows = [row1, row2, row3]

    # --- Lignes de donnees
    for _, row in sub_df.iterrows():
        row_p = []
        for col in all_cols:
            val = row[col]
            formatted = _format_cell_value(val, col)
            row_p.append(Paragraph(formatted, cell_style))
        table_rows.append(row_p)

    col_width = table_width / max(1, n_cols)
    t_data = Table(
        table_rows,
        colWidths=[col_width] * n_cols,
        repeatRows=3
    )

    # --- Styles : en-tetes bleus + spans
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 2), colors.HexColor("#1e3a5f")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 3), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # SPAN : Nombre d'essais sur les 3 lignes d'en-tete
    for k in range(len(id_cols)):
        style_cmds.append(("SPAN", (k, 0), (k, 2)))
    # SPAN : nom de methode sur la ligne 1
    if method_span_end > method_span_start:
        style_cmds.append(("SPAN", (method_span_start, 0), (method_span_end, 0)))
    # SPAN : chaque groupe metrique sur la ligne 2
    for start, end, _ in group_spans:
        style_cmds.append(("SPAN", (start, 1), (end, 1)))

    t_data.setStyle(TableStyle(style_cmds))

    # --- Titre + sous-titre
    title_html = (
        f"<b>Tableau {table_counter} --- Tableau de synthese statistique : "
        f"{method_name}</b><br/>"
        f"<i>Fichier source : {filename}</i>"
    )
    return [Paragraph(title_html, style_h2), Spacer(1, 0.15 * cm), t_data,
            Spacer(1, 0.4 * cm)]

def _render_e9_table_for_mode(sub_df, method_name, table_counter, mode,
                              style_h2, style_table_header, style_table_cell,
                              table_width, filename):
    """
    [FIX-E9-AUDIT] Rendu E9 du rapport d'audit aligne sur le regime.
    Utilise _e9_standalone_metric_groups(mode) pour distinguer
    C_CCI, C_ACI et C_tot en regime adjacent, et ajoute les
    colonnes Gain (%) et CostRatio issues du CSV summary.
    """
    metric_groups = _e9_standalone_metric_groups(mode)

    id_cols = [c for c in ["value"] if c in sub_df.columns]

    data_cols = []
    group_spans = []
    for prefix, label, n_s in metric_groups:
        if n_s == 5:
            cols_for_group = [
                f"{prefix}_{s}"
                for s in ["mean", "std", "median", "min", "max"]
                if f"{prefix}_{s}" in sub_df.columns
            ]
        else:
            cols_for_group = [prefix] if prefix in sub_df.columns else []
        if cols_for_group:
            start = len(id_cols) + len(data_cols)
            data_cols.extend(cols_for_group)
            end = len(id_cols) + len(data_cols) - 1
            group_spans.append((start, end, label, n_s))

    all_cols = id_cols + data_cols
    n_cols = len(all_cols)
    if n_cols == 0:
        return []

    # Styles adaptes
    if n_cols > 20:
        cell_style = ParagraphStyle(
            f"CellE9Audit_{table_counter}", parent=style_table_cell,
            fontSize=5.8, leading=7.2
        )
        header_style = ParagraphStyle(
            f"HeaderE9Audit_{table_counter}", parent=style_table_header,
            fontSize=5.8, leading=7.2
        )
    elif n_cols > 12:
        cell_style = ParagraphStyle(
            f"CellE9Audit_{table_counter}", parent=style_table_cell,
            fontSize=6.5, leading=8.5
        )
        header_style = ParagraphStyle(
            f"HeaderE9Audit_{table_counter}", parent=style_table_header,
            fontSize=7, leading=9
        )
    else:
        cell_style = style_table_cell
        header_style = style_table_header

    # Ligne 1 : nom de methode (SPAN horizontal sur data cols)
    row1 = [Paragraph("", header_style) for _ in range(n_cols)]
    if id_cols:
        row1[0] = Paragraph("<b>Nombre d'essais</b>", header_style)
    method_span_start = len(id_cols)
    method_span_end = n_cols - 1
    row1[method_span_start] = Paragraph(f"<b>{method_name}</b>", header_style)

    # Ligne 2 : libelles de groupes
    row2 = [Paragraph("", header_style) for _ in range(n_cols)]
    for start, end, label, _ in group_spans:
        row2[start] = Paragraph(f"<b>{label}</b>", header_style)

    # Ligne 3 : suffixes statistiques (ou vide pour Gain/CostRatio)
    row3 = [Paragraph("", header_style) for _ in range(n_cols)]
    for k, col_name in enumerate(all_cols):
        if col_name in id_cols:
            continue
        # Determine le groupe de cette colonne
        grp = next(
            (g for g in group_spans if g[0] <= k <= g[1]), None
        )
        if grp is None:
            continue
        n_s = grp[3]
        if n_s == 5:
            suffix = col_name.rsplit("_", 1)[-1]
            row3[k] = Paragraph(
                f"<b>{_STAT_SUFFIX_LABELS.get(suffix, suffix)}</b>",
                header_style
            )

    table_rows = [row1, row2, row3]

    for _, row in sub_df.iterrows():
        row_p = []
        for col in all_cols:
            val = row[col]
            formatted = _format_cell_value(val, col)
            row_p.append(Paragraph(formatted, cell_style))
        table_rows.append(row_p)

    col_width = table_width / max(1, n_cols)
    t_data = Table(table_rows, colWidths=[col_width] * n_cols, repeatRows=3)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 2), colors.HexColor("#1e3a5f")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 3), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for k in range(len(id_cols)):
        style_cmds.append(("SPAN", (k, 0), (k, 2)))
    if method_span_end > method_span_start:
        style_cmds.append(("SPAN", (method_span_start, 0),
                           (method_span_end, 0)))
    for start, end, _, n_s in group_spans:
        if n_s > 1:
            style_cmds.append(("SPAN", (start, 1), (end, 1)))

    t_data.setStyle(TableStyle(style_cmds))

    title_html = (
        f"<b>Tableau {table_counter} --- Tableau de synthese statistique : "
        f"{method_name}</b><br/>"
        f"<i>Fichier source : {filename} (regime {mode})</i>"
    )
    return [Paragraph(title_html, style_h2), Spacer(1, 0.15 * cm),
            t_data, Spacer(1, 0.4 * cm)]

# ============================================================================
# [FIX-E2-TABLES] Generateur de PDFs standalone pour les tables E2
# ============================================================================
# Objectif : produire 16 PDFs (4 methodes x 2 scenarios x 2 regimes) ou
# chaque PDF contient UNE SEULE table au format strict impose par le memoire.
# Ces PDFs sont ensuite inclus dans LaTeX via \includepdf (package pdfpages),
# ce qui elimine toute hallucination LLM sur les valeurs numeriques.

# Etiquettes des statistiques dans l'ordre d'affichage impose par le memoire
_E2_STAT_DISPLAY = [
    ("mean",   "Moy."),
    ("std",    "É.T."),
    ("median", "Méd."),
    ("min",    "Min"),
    ("max",    "Max"),
]

# Paliers de densite (cle CSV, seuil, libelle affiche)
_E2_DENSITY_LEVELS = [
    ("Low",    30, "Densité Faible (Threshold=30)"),
    ("Medium", 50, "Densité Moyenne (Threshold=50)"),
    ("High",   70, "Densité Forte (Threshold=70)"),
]


# ============================================================================
# [FIX-E3-TABLES] Ordre d'affichage impose par le memoire pour les tables E3.
# Identique a l'ordre E2 (mean, std, median, min, max).
# ============================================================================

_E3_STAT_DISPLAY = [
    ("mean",   "Moy."),
    ("std",    "É.T."),
    ("median", "Méd."),
    ("min",    "Min"),
    ("max",    "Max"),
]

# 5 niveaux de K explores dans E3 (scenario S4)
_E3_K_VALUES = [2, 3, 4, 6, 8]

# Ordre d'affichage des methodes dans chaque table E3
_E3_METHOD_ORDER = ["Random", "Greedy", "DSATUR", "BD-CeNN"]
# ============================================================================
# [FIX-E4-TABLES] Ordre d'affichage impose par le memoire pour les tables E4.
# Structure : 1 seule table par regime, 4 sections (Random, Greedy, DSATUR,
# BD-CeNN), 3 lignes par section (Faible 30, Moyenne 50, Forte 70).
# Format A4 paysage.
# ============================================================================

_E4_STAT_DISPLAY = [
    ("mean",   "Moy."),
    ("std",    "É.T."),
    ("median", "Méd."),
    ("min",    "Min"),
    ("max",    "Max"),
]

# Paliers de densite (cle CSV, seuil, libelle affiche)
_E4_DENSITY_LEVELS = [
    ("Low",    30, "Faible (30)"),
    ("Medium", 50, "Moyenne (50)"),
    ("High",   70, "Forte (70)"),
]

# Ordre d'affichage des methodes
_E4_METHOD_ORDER = ["Random", "Greedy", "DSATUR", "BD-CeNN"]


def _e4_metric_groups(mode: str):
    """
    [FIX-E4-TABLES] Retourne les groupes de metriques a afficher selon le
    regime. La table E4 est unique par regime (pas de separation
    cout/conflit vs canaux utilises).

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".

    Returns
    -------
    list of (str, str)
        Liste de tuples (prefixe_colonne, libelle_groupe).
    """
    if mode == "adjacent":
        return [
            ("cost",            "Coût"),
            ("conflicts_cci",   "Conflits C_CCI"),
            ("conflicts_aci",   "Conflits C_ACI"),
            ("conflicts_total", "Conflits C_tot"),
            ("used_channels",   "Canaux utilisés"),
            ("time",            "Temps d'exécution (ms)"),
            ("n_sweeps",        "Nombre de sweeps"),
        ]
    else:
        return [
            ("cost",            "Coût"),
            ("conflicts_total", "Conflit"),
            ("used_channels",   "Canaux utilisés"),
            ("time",            "Temps d'exécution (ms)"),
            ("n_sweeps",        "Nombre de sweeps"),
        ]


def _e3_metric_groups(mode: str, table_type: str):
    """
    [FIX-E3-TABLES] Retourne les groupes de metriques a afficher selon le
    mode (cochannel | adjacent) et le type de table (cost_conflict |
    used_channels).

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".
    table_type : str
        "cost_conflict"  : tableau des couts et conflits
        "used_channels"  : tableau des canaux utilises

    Returns
    -------
    list of (str, str)
        Liste de tuples (prefixe_colonne, libelle_groupe).
    """
    if table_type == "used_channels":
        return [("used_channels", "Canaux utilisés")]

    # cost_conflict
    if mode == "adjacent":
        return [
            ("cost",            "Coût"),
            ("conflicts_cci",   "Conflits C_CCI"),
            ("conflicts_aci",   "Conflits C_ACI"),
            ("conflicts_total", "Conflits C_tot"),
        ]
    else:
        # En CCI-only, C_ACI = 0 par construction. On affiche uniquement
        # le cout et le total (qui vaut C_CCI).
        return [
            ("cost",            "Coût"),
            ("conflicts_total", "Conflit"),
        ]

def _e2_metric_groups(mode: str, method_name: str):
    """
    Retourne la liste des blocs de metriques a afficher selon le regime et
    la methode. Chaque entree est un tuple (prefixe_colonne, libelle_groupe).
    """
    if mode == "adjacent":
        groups = [
            ("cost",             "Coût"),
            ("conflicts_cci",    "Conflits C_CCI"),
            ("conflicts_aci",    "Conflits C_ACI"),
            ("conflicts_total",  "Conflits C_tot"),
            ("used_channels",    "Canaux utilisés"),
            ("time",             "Temps (ms)"),
        ]
    else:
        groups = [
            ("cost",             "Coût"),
            ("conflicts_cci",    "Conflit"),
            ("used_channels",    "Canaux utilisés"),
            ("time",             "Temps (ms)"),
        ]

    # Le solveur BD-CeNN expose en plus le nombre de sweeps (nouvelle convention).
    if method_name == "BD-CeNN":
        groups.append(("n_sweeps", "Nombre de sweeps"))

    return groups


def _e2_format_value(val, prefix: str) -> str:
    """Formate une valeur pour affichage francais (virgule decimale)."""
    if pd.isna(val):
        return "-"
    try:
        v = float(val)
    except Exception:
        return str(val)

    if prefix == "time":
        return f"{v:.2f}".replace(".", ",")
    if prefix in ("cost",):
        return f"{v:.2f}".replace(".", ",")
    if prefix == "n_sweeps":
        return f"{v:.1f}".replace(".", ",")
    # Metriques entieres (conflits, canaux)
    return str(int(round(v)))


def _build_e2_standalone_table_pdf(
    method_name: str,
    scenario_name: str,
    N: int,
    mode: str,
    df_summary: pd.DataFrame,
    output_path: Path,
) -> bool:
    """
    [FIX-E2-TABLES] Construit UN SEUL PDF contenant la table de synthese
    d'une methode pour un scenario et un regime donnes.

    La table reproduit exactement la structure des images de reference :
      - Trois sections de densite (Faible / Moyenne / Forte), chacune avec
        un sous-tableau K=4 puis K=6.
      - Colonnes groupees : K | Cout (5 stats) | C_CCI (5) | C_ACI (5) |
        C_tot (5) | Canaux (5) | Temps (5) [| Sweeps (5) si BD-CeNN].
      - Note de bas de table en petites capitales.

    Returns
    -------
    bool
        True si le PDF a ete genere, False sinon.
    """
    if not REPORTLAB_AVAILABLE:
        return False

    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    )
    from reportlab.lib.units import cm

    sub = df_summary[
        (df_summary["method"] == method_name) &
        (df_summary["N"] == N)
    ].copy()
    if sub.empty:
        return False

    metric_groups = _e2_metric_groups(mode, method_name)
    is_adjacent = (mode == "adjacent")
    regime_label = "Co-canal + canal adjacent" if is_adjacent else "Co-canal"

    # Identification des K disponibles pour ce (N, methode)
    K_values = sorted(sub["K"].unique().tolist())
    if not K_values:
        return False

    n_id_col = 1
    n_stat_per_group = len(_E2_STAT_DISPLAY)
    n_groups = len(metric_groups)
    n_cols = n_id_col + n_groups * n_stat_per_group

    # Preparation des styles ReportLab
    styles = getSampleStyleSheet()
    style_caption = ParagraphStyle(
        "E2Caption", parent=styles["Normal"],
        fontSize=10, leading=13, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=8,
    )
    style_section = ParagraphStyle(
        "E2Section", parent=styles["Normal"],
        fontSize=8, leading=10, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    style_hdr_w = ParagraphStyle(
        "E2HdrW", parent=styles["Normal"],
        fontSize=7, leading=8.5, textColor=colors.white,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    style_hdr_b = ParagraphStyle(
        "E2HdrB", parent=styles["Normal"],
        fontSize=7, leading=8.5, textColor=colors.black,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    style_cell = ParagraphStyle(
        "E2Cell", parent=styles["Normal"],
        fontSize=7, leading=8.5, alignment=TA_CENTER,
        fontName="Helvetica",
    )
    style_note = ParagraphStyle(
        "E2Note", parent=styles["Normal"],
        fontSize=7, leading=9, alignment=TA_LEFT,
        fontName="Helvetica-Oblique",
    )

    # Construction du tableau plat
    table_data = []

    for dens_key, dens_th, dens_label in _E2_DENSITY_LEVELS:
        sub_d = sub[sub["threshold"] == dens_th].sort_values("K")
        if sub_d.empty:
            continue

        # Ligne 1 : titre de la section densite (fusion horizontale totale)
        row1 = [Paragraph(f"<b>{dens_label}</b>", style_section)]
        row1 += [""] * (n_cols - 1)
        table_data.append(row1)

        # Ligne 2 : en-tetes de groupes metriques
        row2 = [Paragraph("<b>K</b>", style_hdr_b)]
        for _prefix, label in metric_groups:
            row2.append(Paragraph(f"<b>{label}</b>", style_hdr_b))
            row2 += [""] * (n_stat_per_group - 1)
        table_data.append(row2)

        # Ligne 3 : suffixes statistiques
        row3 = [""]
        for _prefix, _label in metric_groups:
            for _suffix, s_lbl in _E2_STAT_DISPLAY:
                row3.append(Paragraph(f"<b>{s_lbl}</b>", style_hdr_b))
        table_data.append(row3)

        # Lignes de donnees : une par valeur de K
        for K in K_values:
            sub_k = sub_d[sub_d["K"] == K]
            row_k = [Paragraph(f"<b>{int(K)}</b>", style_cell)]
            if sub_k.empty:
                for _ in range(n_groups * n_stat_per_group):
                    row_k.append(Paragraph("-", style_cell))
            else:
                r = sub_k.iloc[0]
                for prefix, _label in metric_groups:
                    for suffix, _ in _E2_STAT_DISPLAY:
                        col_name = f"{prefix}_{suffix}"
                        val = r.get(col_name, None)
                        row_k.append(
                            Paragraph(_e2_format_value(val, prefix), style_cell)
                        )
            table_data.append(row_k)

    if not table_data:
        return False

    # Largeurs des colonnes
    page_w = landscape(A4)[0]
    available_w = page_w - 2.0 * cm
    k_w = 0.9 * cm
    other_w = (available_w - k_w) / max(1, (n_cols - 1))
    col_widths = [k_w] + [other_w] * (n_cols - 1)

    tbl = Table(table_data, colWidths=col_widths, repeatRows=0)

    # Commandes de style (grille + SPAN sur en-tetes de groupes)
    style_cmds = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]

    # Calcul des indices de lignes pour les SPAN
    row_idx = 0
    for dens_key, dens_th, dens_label in _E2_DENSITY_LEVELS:
        sub_d = sub[sub["threshold"] == dens_th]
        if sub_d.empty:
            continue
        # SPAN horizontal sur le titre de section
        style_cmds.append(("SPAN", (0, row_idx), (n_cols - 1, row_idx)))
        row_idx += 1
        # SPAN vertical sur la cellule "K" (couvre les 2 lignes d'en-tete)
        style_cmds.append(("SPAN", (0, row_idx), (0, row_idx + 1)))
        # SPAN horizontal sur chaque groupe metrique (2 lignes d'en-tete)
        col_cursor = 1
        for _prefix, _label in metric_groups:
            style_cmds.append((
                "SPAN",
                (col_cursor, row_idx),
                (col_cursor + n_stat_per_group - 1, row_idx),
            ))
            col_cursor += n_stat_per_group
        row_idx += 2
        # Lignes de donnees
        row_idx += len(K_values)

    tbl.setStyle(TableStyle(style_cmds))

    # Legende descriptive (la numerotation "Table 15" etc. est geree par LaTeX)
    caption_txt = (
        f"Résultats de la méthode <b>{method_name}</b> ({regime_label}) "
        f"selon K et la densité du réseau du scénario {scenario_name} "
        f"avec N={N}"
    )

    # Note de bas de table
    note_parts = [
        "<i>Note.</i> Moy. = Moyenne, É.T. = Écart-type, "
        "Méd. = Médiane, Min. = Minimal, Max. = Maximal"
    ]
    if is_adjacent:
        note_parts.append(
            "C<sub>CCI</sub> = conflits co-canal, "
            "C<sub>ACI</sub> = conflits adjacents, "
            "C<sub>tot</sub> = C<sub>CCI</sub> + C<sub>ACI</sub>"
        )
    note_txt = ", ".join(note_parts) + "."

    story = [
        Paragraph(caption_txt, style_caption),
        tbl,
        Spacer(1, 0.3 * cm),
        Paragraph(note_txt, style_note),
    ]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=1.0 * cm,
        rightMargin=1.0 * cm,
        topMargin=1.0 * cm,
        bottomMargin=0.8 * cm,
    )
    doc.build(story)
    return True

def _build_e3_standalone_table_pdf(
    table_type: str,
    mode: str,
    df_summary: pd.DataFrame,
    output_path: Path,
) -> bool:
    """
    [FIX-E3-TABLES] Construit UN SEUL PDF contenant la table de synthese
    de l'experience E3 (scenario S4, N=50, K variable).

    La table reproduit la structure des images de reference :
      - Un en-tete de caption en haut.
      - Quatre sections (Random, Greedy, DSATUR, BD-CeNN).
      - Pour chaque section, un sous-tableau K = 2, 3, 4, 6, 8.
      - Colonnes groupees : K | groupe1 (5 stats) | groupe2 (5 stats) | ...
      - Note de bas de table.

    Le format de page est A4 portrait, conformement a la consigne.

    Parameters
    ----------
    table_type : str
        "cost_conflict" ou "used_channels".
    mode : str
        "cochannel" ou "adjacent".
    df_summary : pd.DataFrame
        Contenu de results/csv/E3/{mode}/summary_metrics.csv.
    output_path : Path
        Chemin de sortie du PDF.

    Returns
    -------
    bool
        True si le PDF a ete genere, False sinon.
    """
    if not REPORTLAB_AVAILABLE:
        return False

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    )
    from reportlab.lib.units import cm

    metric_groups = _e3_metric_groups(mode, table_type)
    is_adjacent = (mode == "adjacent")
    regime_label = "Co-canal + canal adjacent" if is_adjacent else "Co-canal"

    # Comptage des colonnes (1 pour K + 5 par groupe de metriques)
    n_stat_per_group = len(_E3_STAT_DISPLAY)
    n_groups = len(metric_groups)
    n_cols = 1 + n_groups * n_stat_per_group

    # Preparation des styles ReportLab
    styles = getSampleStyleSheet()
    style_caption = ParagraphStyle(
        "E3Caption", parent=styles["Normal"],
        fontSize=11, leading=14, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=10,
    )
    style_section = ParagraphStyle(
        "E3Section", parent=styles["Normal"],
        fontSize=9, leading=11, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    style_hdr = ParagraphStyle(
        "E3Hdr", parent=styles["Normal"],
        fontSize=7, leading=8.5, textColor=colors.black,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    style_cell = ParagraphStyle(
        "E3Cell", parent=styles["Normal"],
        fontSize=7, leading=8.5, alignment=TA_CENTER,
        fontName="Helvetica",
    )
    style_note = ParagraphStyle(
        "E3Note", parent=styles["Normal"],
        fontSize=7, leading=9, alignment=TA_LEFT,
        fontName="Helvetica-Oblique",
    )

    # Reduction adaptative de la police si la table est large
    # (utile pour la version adjacente a 21 colonnes en A4 portrait).
    if n_cols > 15:
        cell_style = ParagraphStyle(
            "E3CellTight", parent=style_cell,
            fontSize=6, leading=7.5,
        )
        hdr_style = ParagraphStyle(
            "E3HdrTight", parent=style_hdr,
            fontSize=5.8, leading=7.2,
        )
    else:
        cell_style = style_cell
        hdr_style = style_hdr

    # Construction du tableau plat
    table_data = []

    for method in _E3_METHOD_ORDER:
        sub_m = df_summary[df_summary["method"] == method]
        if sub_m.empty:
            continue

        # Ligne 1 : titre de section methode (fusion horizontale totale)
        row_sec = [Paragraph(f"<b>{method}</b>", style_section)]
        row_sec += [""] * (n_cols - 1)
        table_data.append(row_sec)

        # Ligne 2 : en-tetes des groupes de metriques
        row_hdr1 = [Paragraph("<b>K</b>", hdr_style)]
        for _prefix, label in metric_groups:
            row_hdr1.append(Paragraph(f"<b>{label}</b>", hdr_style))
            row_hdr1 += [""] * (n_stat_per_group - 1)
        table_data.append(row_hdr1)

        # Ligne 3 : suffixes statistiques
        row_hdr2 = [""]
        for _prefix, _label in metric_groups:
            for _suffix, s_lbl in _E3_STAT_DISPLAY:
                row_hdr2.append(Paragraph(f"<b>{s_lbl}</b>", hdr_style))
        table_data.append(row_hdr2)

        # Lignes de donnees : une par valeur de K
        for K in _E3_K_VALUES:
            sub_k = sub_m[sub_m["K"] == K]
            row_k = [Paragraph(f"<b>{int(K)}</b>", cell_style)]
            if sub_k.empty:
                for _ in range(n_groups * n_stat_per_group):
                    row_k.append(Paragraph("-", cell_style))
            else:
                r = sub_k.iloc[0]
                for prefix, _label in metric_groups:
                    for suffix, _ in _E3_STAT_DISPLAY:
                        col_name = f"{prefix}_{suffix}"
                        val = r.get(col_name, None)
                        row_k.append(
                            Paragraph(_e2_format_value(val, prefix), cell_style)
                        )
            table_data.append(row_k)

    if not table_data:
        return False

    # Largeurs des colonnes (A4 portrait)
    page_w = A4[0]  # A4 portrait : 595.27 pt de large
    available_w = page_w - 2.0 * cm
    k_w = 0.8 * cm
    other_w = (available_w - k_w) / max(1, (n_cols - 1))
    col_widths = [k_w] + [other_w] * (n_cols - 1)

    tbl = Table(table_data, colWidths=col_widths, repeatRows=0)

    # Commandes de style
    style_cmds = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]

    # Calcul des SPANs (fusion section + fusion en-tete de groupe)
    row_idx = 0
    for method in _E3_METHOD_ORDER:
        sub_m = df_summary[df_summary["method"] == method]
        if sub_m.empty:
            continue

        # SPAN horizontal sur la ligne de section methode
        style_cmds.append(("SPAN", (0, row_idx), (n_cols - 1, row_idx)))
        row_idx += 1

        # SPAN vertical sur la cellule "K" (couvre 2 lignes d'en-tete)
        style_cmds.append(("SPAN", (0, row_idx), (0, row_idx + 1)))

        # SPAN horizontal sur chaque groupe metrique (2 lignes d'en-tete)
        col_cursor = 1
        for _prefix, _label in metric_groups:
            style_cmds.append((
                "SPAN",
                (col_cursor, row_idx),
                (col_cursor + n_stat_per_group - 1, row_idx),
            ))
            col_cursor += n_stat_per_group

        row_idx += 2  # 2 lignes d'en-tete
        row_idx += len(_E3_K_VALUES)  # lignes de donnees

    tbl.setStyle(TableStyle(style_cmds))

    # Construction du caption (le numero "Table X" est gere par LaTeX)
    if table_type == "used_channels":
        caption_txt = (
            f"Nombre de canaux utilisés ({regime_label}) "
            f"selon K et la méthode"
        )
    else:
        caption_txt = (
            f"Résultats comparatifs des 4 méthodes ({regime_label}) "
            f"selon le nombre de canaux K"
        )

    # Note de bas de table
    note_parts = [
        "<i>Note.</i> Moy. = Moyenne, É.T. = Écart-type, "
        "Méd. = Médiane, Min. = Minimal, Max. = Maximal"
    ]
    if is_adjacent and table_type == "cost_conflict":
        note_parts.append(
            "C<sub>CCI</sub> = conflits co-canal, "
            "C<sub>ACI</sub> = conflits adjacents, "
            "C<sub>tot</sub> = C<sub>CCI</sub> + C<sub>ACI</sub>"
        )
    note_txt = ", ".join(note_parts) + "."

    story = [
        Paragraph(caption_txt, style_caption),
        tbl,
        Spacer(1, 0.3 * cm),
        Paragraph(note_txt, style_note),
    ]

    # Page A4 PORTRAIT (le user l'a explicitement demande pour E3)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,  # A4 par defaut en portrait dans ReportLab
        leftMargin=1.0 * cm,
        rightMargin=1.0 * cm,
        topMargin=1.0 * cm,
        bottomMargin=0.8 * cm,
    )
    doc.build(story)
    return True

def _build_e4_standalone_table_pdf(
    mode: str,
    df_summary: pd.DataFrame,
    output_path: Path,
) -> bool:
    """
    [FIX-E4-TABLES] Construit UN SEUL PDF contenant la table de synthese
    de l'experience E4 (scenario S3, N=50, K=6, densite variable).

    Structure conforme a l'image de reference :
      - Caption en haut.
      - En-tete a 2 niveaux : ligne 1 = groupes de metriques (fusionnes
        horizontalement), ligne 2 = suffixes statistiques.
      - Quatre sections (Random, Greedy, DSATUR, BD-CeNN).
      - Pour chaque section, trois lignes (Faible 30, Moyenne 50, Forte 70).
      - Note de bas de table.

    Format : A4 paysage.

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".
    df_summary : pd.DataFrame
        Contenu de results/csv/E4/{mode}/summary_metrics.csv.
    output_path : Path
        Chemin de sortie du PDF.

    Returns
    -------
    bool
        True si le PDF a ete genere, False sinon.
    """
    if not REPORTLAB_AVAILABLE:
        return False

    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    )
    from reportlab.lib.units import cm

    metric_groups = _e4_metric_groups(mode)
    is_adjacent = (mode == "adjacent")
    regime_label = "Co-canal + canal adjacent" if is_adjacent else "Co-canal"

    # Comptage des colonnes : 1 pour Densite + 5 par groupe
    n_stat_per_group = len(_E4_STAT_DISPLAY)
    n_groups = len(metric_groups)
    n_cols = 1 + n_groups * n_stat_per_group

    # Preparation des styles ReportLab
    styles = getSampleStyleSheet()
    style_caption = ParagraphStyle(
        "E4Caption", parent=styles["Normal"],
        fontSize=11, leading=14, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=10,
    )
    style_section = ParagraphStyle(
        "E4Section", parent=styles["Normal"],
        fontSize=9, leading=11, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    style_note = ParagraphStyle(
        "E4Note", parent=styles["Normal"],
        fontSize=7.5, leading=9.5, alignment=TA_LEFT,
        fontName="Helvetica-Oblique",
    )

    # Reduction adaptative de la police selon la densite de colonnes.
    # La version adjacente a 36 colonnes necessite une police tres petite.
    if n_cols > 30:
        hdr_style = ParagraphStyle(
            "E4HdrTight", parent=styles["Normal"],
            fontSize=5.5, leading=7, textColor=colors.black,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        )
        cell_style = ParagraphStyle(
            "E4CellTight", parent=styles["Normal"],
            fontSize=5.5, leading=7, alignment=TA_CENTER,
            fontName="Helvetica",
        )
    elif n_cols > 20:
        hdr_style = ParagraphStyle(
            "E4HdrMed", parent=styles["Normal"],
            fontSize=6.2, leading=8, textColor=colors.black,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        )
        cell_style = ParagraphStyle(
            "E4CellMed", parent=styles["Normal"],
            fontSize=6.2, leading=8, alignment=TA_CENTER,
            fontName="Helvetica",
        )
    else:
        hdr_style = ParagraphStyle(
            "E4HdrStd", parent=styles["Normal"],
            fontSize=7, leading=8.5, textColor=colors.black,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        )
        cell_style = ParagraphStyle(
            "E4CellStd", parent=styles["Normal"],
            fontSize=7, leading=8.5, alignment=TA_CENTER,
            fontName="Helvetica",
        )

    # Helper de formatage local : "—" pour les cellules non applicables
    def _fmt(val, prefix):
        if pd.isna(val):
            return "—"
        try:
            v = float(val)
        except Exception:
            return str(val)
        if prefix == "time":
            return f"{v:.2f}".replace(".", ",")
        if prefix == "cost":
            return f"{v:.2f}".replace(".", ",")
        if prefix == "n_sweeps":
            return f"{v:.1f}".replace(".", ",")
        return str(int(round(v)))

    # Construction du tableau plat
    table_data = []

    # --- Ligne 1 : en-tetes de groupes (avec "Densité" fusionne verticalement)
    row1 = [Paragraph("<b>Densité</b>", hdr_style)]
    for _prefix, label in metric_groups:
        row1.append(Paragraph(f"<b>{label}</b>", hdr_style))
        row1 += [""] * (n_stat_per_group - 1)
    table_data.append(row1)

    # --- Ligne 2 : suffixes statistiques
    row2 = [""]
    for _prefix, _label in metric_groups:
        for _suffix, s_lbl in _E4_STAT_DISPLAY:
            row2.append(Paragraph(f"<b>{s_lbl}</b>", hdr_style))
    table_data.append(row2)

    # --- Corps : une section par methode, 3 lignes par section
    for method in _E4_METHOD_ORDER:
        sub_m = df_summary[df_summary["method"] == method]
        if sub_m.empty:
            continue

        # Ligne de section (fusion horizontale sur toutes les colonnes)
        row_sec = [Paragraph(f"<b>{method}</b>", style_section)]
        row_sec += [""] * (n_cols - 1)
        table_data.append(row_sec)

        # Trois lignes de donnees : une par densite
        for _dens_key, dens_th, dens_label in _E4_DENSITY_LEVELS:
            sub_d = sub_m[sub_m["threshold"] == dens_th]
            row_d = [Paragraph(dens_label, cell_style)]
            if sub_d.empty:
                for _ in range(n_groups * n_stat_per_group):
                    row_d.append(Paragraph("—", cell_style))
            else:
                r = sub_d.iloc[0]
                for prefix, _label in metric_groups:
                    for suffix, _ in _E4_STAT_DISPLAY:
                        col_name = f"{prefix}_{suffix}"
                        val = r.get(col_name, None)
                        row_d.append(Paragraph(_fmt(val, prefix), cell_style))
            table_data.append(row_d)

    if not table_data:
        return False

    # Largeurs des colonnes (A4 paysage)
    page_w = landscape(A4)[0]  # ~842 pt
    available_w = page_w - 1.6 * cm
    dens_w = 2.4 * cm
    other_w = (available_w - dens_w) / max(1, (n_cols - 1))
    col_widths = [dens_w] + [other_w] * (n_cols - 1)

    tbl = Table(table_data, colWidths=col_widths, repeatRows=0)

    # --- Commandes de style
    style_cmds = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
    ]

    # SPAN vertical sur la cellule "Densité" (couvre les 2 lignes d'en-tete)
    style_cmds.append(("SPAN", (0, 0), (0, 1)))

    # SPAN horizontal sur chaque groupe de metriques (ligne 1)
    col_cursor = 1
    for _prefix, _label in metric_groups:
        style_cmds.append((
            "SPAN",
            (col_cursor, 0),
            (col_cursor + n_stat_per_group - 1, 0),
        ))
        col_cursor += n_stat_per_group

    # SPAN horizontal sur chaque ligne de section methode
    row_idx = 2  # on a deja 2 lignes d'en-tete
    for method in _E4_METHOD_ORDER:
        sub_m = df_summary[df_summary["method"] == method]
        if sub_m.empty:
            continue
        style_cmds.append(("SPAN", (0, row_idx), (n_cols - 1, row_idx)))
        row_idx += 1  # ligne de section
        row_idx += 3  # 3 lignes de densite

    tbl.setStyle(TableStyle(style_cmds))

    # --- Caption
    caption_txt = (
        f"Résultats comparatifs selon la densité du réseau ({regime_label})"
    )

    # --- Note de bas de table
    note_parts = [
        "<i>Note.</i> Moy. = Moyenne, É.T. = Écart-Type, Méd. = Médiane. "
        "Le symbole « — » indique une métrique non applicable à la méthode."
    ]
    if is_adjacent:
        note_parts.append(
            "C<sub>CCI</sub> = conflits co-canal, "
            "C<sub>ACI</sub> = conflits adjacents, "
            "C<sub>tot</sub> = C<sub>CCI</sub> + C<sub>ACI</sub>."
        )
    note_txt = " ".join(note_parts)

    story = [
        Paragraph(caption_txt, style_caption),
        tbl,
        Spacer(1, 0.3 * cm),
        Paragraph(note_txt, style_note),
    ]

    # --- Document A4 paysage
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=0.8 * cm,
        rightMargin=0.8 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
    )
    doc.build(story)
    return True

def _build_e6_standalone_table_pdf(
    mode: str,
    df_summary: pd.DataFrame,
    output_path: Path,
) -> bool:
    """
    [FIX-E6-TABLES] Construit UN SEUL PDF contenant la table de synthese
    de l'experience E6 (scalabilite, K=8, threshold=50, N variable).

    Structure conforme a l'image de reference :
      - Caption en haut.
      - En-tete a 2 niveaux : ligne 1 = groupes de metriques (fusionnes
        horizontalement), ligne 2 = suffixes statistiques.
      - Quatre sections (Random, Greedy, DSATUR, BD-CeNN).
      - Pour chaque section, cinq lignes (N = 20, 30, 50, 100, 200).
      - Note de bas de table.

    Format : A4 paysage.

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".
    df_summary : pd.DataFrame
        Contenu de results/csv/E6/{mode}/summary_metrics.csv.
    output_path : Path
        Chemin de sortie du PDF.

    Returns
    -------
    bool
        True si le PDF a ete genere, False sinon.
    """
    if not REPORTLAB_AVAILABLE:
        return False

    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    )
    from reportlab.lib.units import cm

    metric_groups = _e6_metric_groups(mode)
    is_adjacent = (mode == "adjacent")
    regime_label = "Co-canal + canal adjacent" if is_adjacent else "Co-canal"

    # Comptage des colonnes : 1 pour N + 5 par groupe
    n_stat_per_group = len(_E6_STAT_DISPLAY)
    n_groups = len(metric_groups)
    n_cols = 1 + n_groups * n_stat_per_group

    # Preparation des styles ReportLab
    styles = getSampleStyleSheet()
    style_caption = ParagraphStyle(
        "E6Caption", parent=styles["Normal"],
        fontSize=11, leading=14, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=10,
    )
    style_section = ParagraphStyle(
        "E6Section", parent=styles["Normal"],
        fontSize=9, leading=11, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    style_note = ParagraphStyle(
        "E6Note", parent=styles["Normal"],
        fontSize=7.5, leading=9.5, alignment=TA_LEFT,
        fontName="Helvetica-Oblique",
    )

    # Reduction adaptative de la police selon la densite de colonnes.
    # La version adjacente a 36 colonnes necessite une police tres petite.
    if n_cols > 30:
        hdr_style = ParagraphStyle(
            "E6HdrTight", parent=styles["Normal"],
            fontSize=5.5, leading=7, textColor=colors.black,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        )
        cell_style = ParagraphStyle(
            "E6CellTight", parent=styles["Normal"],
            fontSize=5.5, leading=7, alignment=TA_CENTER,
            fontName="Helvetica",
        )
    elif n_cols > 20:
        hdr_style = ParagraphStyle(
            "E6HdrMed", parent=styles["Normal"],
            fontSize=6.2, leading=8, textColor=colors.black,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        )
        cell_style = ParagraphStyle(
            "E6CellMed", parent=styles["Normal"],
            fontSize=6.2, leading=8, alignment=TA_CENTER,
            fontName="Helvetica",
        )
    else:
        hdr_style = ParagraphStyle(
            "E6HdrStd", parent=styles["Normal"],
            fontSize=7, leading=8.5, textColor=colors.black,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        )
        cell_style = ParagraphStyle(
            "E6CellStd", parent=styles["Normal"],
            fontSize=7, leading=8.5, alignment=TA_CENTER,
            fontName="Helvetica",
        )

    # Helper de formatage local : "—" pour les cellules non applicables
    def _fmt(val, prefix):
        if pd.isna(val):
            return "—"
        try:
            v = float(val)
        except Exception:
            return str(val)
        if prefix == "time":
            # Conversion en ms pour uniformiser l'affichage (E6 stocke en s)
            return f"{v * 1000:.2f}".replace(".", ",")
        if prefix == "cost":
            return f"{v:.2f}".replace(".", ",")
        if prefix == "n_sweeps":
            return f"{v:.1f}".replace(".", ",")
        return str(int(round(v)))

    # Construction du tableau plat
    table_data = []

    # --- Ligne 1 : en-tetes de groupes (avec "N" fusionne verticalement)
    row1 = [Paragraph("<b>N</b>", hdr_style)]
    for _prefix, label in metric_groups:
        row1.append(Paragraph(f"<b>{label}</b>", hdr_style))
        row1 += [""] * (n_stat_per_group - 1)
    table_data.append(row1)

    # --- Ligne 2 : suffixes statistiques
    row2 = [""]
    for _prefix, _label in metric_groups:
        for _suffix, s_lbl in _E6_STAT_DISPLAY:
            row2.append(Paragraph(f"<b>{s_lbl}</b>", hdr_style))
    table_data.append(row2)

    # --- Corps : une section par methode, 5 lignes par section
    for method in _E6_METHOD_ORDER:
        sub_m = df_summary[df_summary["method"] == method]
        if sub_m.empty:
            continue

        # Ligne de section (fusion horizontale sur toutes les colonnes)
        row_sec = [Paragraph(f"<b>{method}</b>", style_section)]
        row_sec += [""] * (n_cols - 1)
        table_data.append(row_sec)

        # Cinq lignes de donnees : une par valeur de N
        for N_val in _E6_N_VALUES:
            sub_n = sub_m[sub_m["N"] == N_val]
            row_n = [Paragraph(str(N_val), cell_style)]
            if sub_n.empty:
                for _ in range(n_groups * n_stat_per_group):
                    row_n.append(Paragraph("—", cell_style))
            else:
                r = sub_n.iloc[0]
                for prefix, _label in metric_groups:
                    for suffix, _ in _E6_STAT_DISPLAY:
                        col_name = f"{prefix}_{suffix}"
                        val = r.get(col_name, None)
                        row_n.append(Paragraph(_fmt(val, prefix), cell_style))
            table_data.append(row_n)

    if not table_data:
        return False

    # Largeurs des colonnes (A4 paysage)
    page_w = landscape(A4)[0]  # ~842 pt
    available_w = page_w - 1.6 * cm
    n_w = 1.4 * cm
    other_w = (available_w - n_w) / max(1, (n_cols - 1))
    col_widths = [n_w] + [other_w] * (n_cols - 1)

    tbl = Table(table_data, colWidths=col_widths, repeatRows=0)

    # --- Commandes de style
    style_cmds = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
    ]

    # SPAN vertical sur la cellule "N" (couvre les 2 lignes d'en-tete)
    style_cmds.append(("SPAN", (0, 0), (0, 1)))

    # SPAN horizontal sur chaque groupe de metriques (ligne 1)
    col_cursor = 1
    for _prefix, _label in metric_groups:
        style_cmds.append((
            "SPAN",
            (col_cursor, 0),
            (col_cursor + n_stat_per_group - 1, 0),
        ))
        col_cursor += n_stat_per_group

    # SPAN horizontal sur chaque ligne de section methode
    row_idx = 2  # on a deja 2 lignes d'en-tete
    for method in _E6_METHOD_ORDER:
        sub_m = df_summary[df_summary["method"] == method]
        if sub_m.empty:
            continue
        style_cmds.append(("SPAN", (0, row_idx), (n_cols - 1, row_idx)))
        row_idx += 1  # ligne de section
        row_idx += 5  # 5 lignes de N

    tbl.setStyle(TableStyle(style_cmds))

    # --- Caption
    caption_txt = (
        f"Résultats comparatifs selon la scalabilité du réseau ({regime_label})"
    )

    # --- Note de bas de table
    note_parts = [
        "<i>Note.</i> moy. = moyenne, é.t. = écart-type, méd. = médiane. "
        "Le symbole « — » indique une métrique non applicable à la méthode."
    ]
    if is_adjacent:
        note_parts.append(
            "C<sub>CCI</sub> = conflits co-canal, "
            "C<sub>ACI</sub> = conflits adjacents, "
            "C<sub>tot</sub> = C<sub>CCI</sub> + C<sub>ACI</sub>."
        )
    note_txt = " ".join(note_parts)

    story = [
        Paragraph(caption_txt, style_caption),
        tbl,
        Spacer(1, 0.3 * cm),
        Paragraph(note_txt, style_note),
    ]

    # --- Document A4 paysage
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=0.8 * cm,
        rightMargin=0.8 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
    )
    doc.build(story)
    return True

def _build_e8_standalone_table_pdf(
    mode: str,
    df_summary: pd.DataFrame,
    output_path: Path,
) -> bool:
    """
    [FIX-E8-TABLES] Construit UN SEUL PDF contenant la table de synthese
    de l'experience E8 (reseau dynamique, scenario S7, N=45, K=5).

    Structure conforme a l'image de reference :
      - Caption en haut (Table X --- Tableau comparatif...).
      - Deux sections : "Réoptimisation à chaud (Warm start)" puis
        "Réoptimisation à froid (Cold start)".
      - Chaque section contient :
          * Ligne 1 : nom du mode (fusion horizontale sur 15 colonnes).
          * Ligne 2 : 3 groupes metriques x 5 stats (moy., e.t., med., min, max).
          * 3 lignes de donnees : mod_level 5, 10, 20.
      - Note de bas de table.

    Format : A4 paysage.

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".
    df_summary : pd.DataFrame
        Contenu de results/csv/E8/{mode}/summary_metrics.csv.
    output_path : Path
        Chemin de sortie du PDF.

    Returns
    -------
    bool
        True si le PDF a ete genere, False sinon.
    """
    if not REPORTLAB_AVAILABLE:
        return False

    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    )
    from reportlab.lib.units import cm

    metric_groups = _e8_metric_groups()
    is_adjacent = (mode == "adjacent")
    regime_label = "Co-canal + canal adjacent" if is_adjacent else "Co-canal"

    # Comptage des colonnes : 1 pour mod_level + 5 par groupe de metriques
    n_stat_per_group = len(_E8_STAT_DISPLAY)
    n_groups = len(metric_groups)
    n_data_cols = n_groups * n_stat_per_group
    n_cols = 1 + n_data_cols  # 16 colonnes au total

    # Preparation des styles ReportLab
    styles = getSampleStyleSheet()
    style_caption = ParagraphStyle(
        "E8Caption", parent=styles["Normal"],
        fontSize=10, leading=13, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=10,
    )
    style_mode_hdr = ParagraphStyle(
        "E8ModeHdr", parent=styles["Normal"],
        fontSize=9, leading=11, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    style_hdr = ParagraphStyle(
        "E8Hdr", parent=styles["Normal"],
        fontSize=7.5, leading=9.5, textColor=colors.black,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    style_cell = ParagraphStyle(
        "E8Cell", parent=styles["Normal"],
        fontSize=7.5, leading=9.5, alignment=TA_CENTER,
        fontName="Helvetica",
    )
    style_note = ParagraphStyle(
        "E8Note", parent=styles["Normal"],
        fontSize=7.5, leading=9.5, alignment=TA_LEFT,
        fontName="Helvetica-Oblique",
    )

    # Helper de formatage local. Les temps sont stockes en secondes dans le CSV
    # mais affiches en millisecondes (conformement a l'image de reference).
    def _fmt(val, prefix):
        if pd.isna(val):
            return "—"
        try:
            v = float(val)
        except Exception:
            return str(val)
        if prefix == "time":
            return f"{v * 1000:.2f}".replace(".", ",")
        if prefix == "cost":
            return f"{v:.2f}".replace(".", ",")
        if prefix == "changes":
            return f"{v:.2f}".replace(".", ",")
        return str(val)

    # Construction du tableau plat
    table_data = []

    for mode_key in _E8_MODE_ORDER:
        sub_m = df_summary[df_summary["mode"] == mode_key]
        if sub_m.empty:
            continue

        mode_label = _E8_MODE_LABELS[mode_key]

        # --- Ligne 1 : titre du mode (fusion horizontale sur les 15 colonnes
        # de donnees, la cellule "Taux de modif." reste pour la ligne d'en-tete)
        row_mode = [Paragraph("", style_hdr)]
        row_mode += [Paragraph(f"<b>{mode_label}</b>", style_mode_hdr)]
        row_mode += [""] * (n_data_cols - 1)
        table_data.append(row_mode)

        # --- Ligne 2 : en-tetes de groupes de metriques
        row_hdr1 = [Paragraph("<b>Taux de modif.<br/>(%)</b>", style_hdr)]
        for _prefix, label in metric_groups:
            row_hdr1.append(Paragraph(f"<b>{label}</b>", style_hdr))
            row_hdr1 += [""] * (n_stat_per_group - 1)
        table_data.append(row_hdr1)

        # --- Ligne 3 : suffixes statistiques
        row_hdr2 = [""]
        for _prefix, _label in metric_groups:
            for _suffix, s_lbl in _E8_STAT_DISPLAY:
                row_hdr2.append(Paragraph(f"<b>{s_lbl}</b>", style_hdr))
        table_data.append(row_hdr2)

        # --- Lignes de donnees : une par mod_level
        for mod_lvl in _E8_MOD_LEVELS:
            sub_lvl = sub_m[sub_m["mod_level"] == mod_lvl]
            row_lvl = [Paragraph(str(mod_lvl), style_cell)]
            if sub_lvl.empty:
                for _ in range(n_data_cols):
                    row_lvl.append(Paragraph("—", style_cell))
            else:
                r = sub_lvl.iloc[0]
                for prefix, _label in metric_groups:
                    for suffix, _ in _E8_STAT_DISPLAY:
                        col_name = f"{prefix}_{suffix}"
                        val = r.get(col_name, None)
                        row_lvl.append(Paragraph(_fmt(val, prefix), style_cell))
            table_data.append(row_lvl)

    if not table_data:
        return False

    # Largeurs des colonnes (A4 paysage)
    page_w = landscape(A4)[0]  # ~842 pt
    available_w = page_w - 1.6 * cm
    mod_w = 1.6 * cm
    other_w = (available_w - mod_w) / max(1, n_data_cols)
    col_widths = [mod_w] + [other_w] * n_data_cols

    tbl = Table(table_data, colWidths=col_widths, repeatRows=0)

    # --- Commandes de style
    style_cmds = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
    ]

    # On parcourt chaque mode pour poser les SPAN :
    #   - SPAN horizontal sur la ligne de titre du mode (colonnes 1 a 15)
    #   - SPAN vertical sur "Taux de modif. (%)" (couvre 2 lignes d'en-tete)
    #   - SPAN horizontal sur chaque groupe metrique (ligne 2)
    row_idx = 0
    for mode_key in _E8_MODE_ORDER:
        sub_m = df_summary[df_summary["mode"] == mode_key]
        if sub_m.empty:
            continue

        # Ligne 1 : SPAN horizontal sur le titre du mode (de la colonne 1 a n_cols-1)
        style_cmds.append(("SPAN", (1, row_idx), (n_cols - 1, row_idx)))
        # On laisse la premiere cellule (col 0) vide pour cette ligne
        row_idx += 1

        # Ligne 2 et 3 : SPAN vertical sur la cellule "Taux de modif."
        style_cmds.append(("SPAN", (0, row_idx), (0, row_idx + 1)))

        # SPAN horizontal sur chaque groupe metrique
        col_cursor = 1
        for _prefix, _label in metric_groups:
            style_cmds.append((
                "SPAN",
                (col_cursor, row_idx),
                (col_cursor + n_stat_per_group - 1, row_idx),
            ))
            col_cursor += n_stat_per_group

        row_idx += 2  # 2 lignes d'en-tete
        row_idx += len(_E8_MOD_LEVELS)  # 3 lignes de donnees

    tbl.setStyle(TableStyle(style_cmds))

    # --- Caption
    caption_txt = (
        f"Tableau comparatif du mode Warm start et Cold start du BD-CeNN "
        f"pour le scénario S7 (N = 45, K = 5, Area = 200, Threshold = 30) "
        f"--- Régime {regime_label}"
    )

    # --- Note de bas de table
    note_txt = (
        "<i>Note.</i> moy. = moyenne, é.t. = écart-type, méd. = médiane. "
        "Les temps ont été convertis en millisecondes (ms)."
    )

    story = [
        Paragraph(caption_txt, style_caption),
        tbl,
        Spacer(1, 0.3 * cm),
        Paragraph(note_txt, style_note),
    ]

    # --- Document A4 paysage
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=0.8 * cm,
        rightMargin=0.8 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
    )
    doc.build(story)
    return True

def _build_e9_standalone_table_pdf(
    mode: str,
    df_summary: pd.DataFrame,
    output_path: Path,
) -> bool:
    """
    [FIX-E9-TABLES] Construit UN SEUL PDF contenant la table comparative
    Greedy vs BD-CeNN selon le nombre d'essais (scenario S3).

    Structure conforme a l'image de reference :
      - Caption en haut.
      - Section Greedy (haut) puis section BD-CeNN (bas).
      - Chaque section : ligne 1 = nom de la methode (fusion horizontale),
        ligne 2 = groupes de metriques, ligne 3 = suffixes statistiques.
      - 4 lignes de donnees : R = 1, 5, 10, 20.
      - Colonnes Gain (%) et CostRatio en fin de bloc, pour materialiser
        le compromis qualite/cout demande par le rapport d'evaluation.

    Format : A4 paysage.
    """
    if not REPORTLAB_AVAILABLE:
        return False

    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    )
    from reportlab.lib.units import cm

    metric_groups = _e9_standalone_metric_groups(mode)
    is_adjacent = (mode == "adjacent")
    regime_label = "Co-canal + canal adjacent" if is_adjacent else "Co-canal"

    # Comptage des colonnes : 1 pour "Nombre d'essais" + somme des tailles
    n_data_cols = sum(g[2] for g in metric_groups)
    n_cols = 1 + n_data_cols

    # Styles
    styles = getSampleStyleSheet()
    style_caption = ParagraphStyle(
        "E9Caption", parent=styles["Normal"],
        fontSize=11, leading=14, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=10,
    )
    style_mode_hdr = ParagraphStyle(
        "E9ModeHdr", parent=styles["Normal"],
        fontSize=9, leading=11, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    style_hdr = ParagraphStyle(
        "E9Hdr", parent=styles["Normal"],
        fontSize=7, leading=8.5, textColor=colors.black,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    style_cell = ParagraphStyle(
        "E9Cell", parent=styles["Normal"],
        fontSize=7, leading=8.5, alignment=TA_CENTER,
        fontName="Helvetica",
    )
    style_note = ParagraphStyle(
        "E9Note", parent=styles["Normal"],
        fontSize=7, leading=9, alignment=TA_LEFT,
        fontName="Helvetica-Oblique",
    )

    # Reduction adaptative de la police si la table est tres large
    if n_cols > 25:
        cell_style = ParagraphStyle(
            "E9CellTight", parent=style_cell,
            fontSize=5.8, leading=7.2,
        )
        hdr_style = ParagraphStyle(
            "E9HdrTight", parent=style_hdr,
            fontSize=5.8, leading=7.2,
        )
    else:
        cell_style = style_cell
        hdr_style = style_hdr

    # Helper de formatage
    def _fmt(val, prefix):
        if pd.isna(val):
            return "—"
        try:
            v = float(val)
        except Exception:
            return str(val)
        if prefix == "time":
            return f"{v * 1000:.2f}".replace(".", ",")
        if prefix == "cost":
            return f"{v:.2f}".replace(".", ",")
        if prefix == "gain_pct":
            return f"{v:.1f}".replace(".", ",")
        if prefix == "cost_ratio":
            return f"{v:.2f}".replace(".", ",")
        return str(int(round(v)))

    # Construction du tableau
    table_data = []

    for method in _E9_METHOD_ORDER_STANDALONE:
        sub_m = df_summary[df_summary["method"] == method]
        if sub_m.empty:
            continue

        # --- Ligne 1 : nom de la methode (SPAN horizontal sur data cols)
        row_method = [Paragraph("", hdr_style)]
        row_method += [Paragraph(f"<b>{method}</b>", style_mode_hdr)]
        row_method += [""] * (n_data_cols - 1)
        table_data.append(row_method)

        # --- Ligne 2 : groupes de metriques
        row_hdr1 = [Paragraph("<b>Nombre d'essais</b>", hdr_style)]
        for _prefix, label, n_s in metric_groups:
            row_hdr1.append(Paragraph(f"<b>{label}</b>", hdr_style))
            row_hdr1 += [""] * (n_s - 1)
        table_data.append(row_hdr1)

        # --- Ligne 3 : suffixes statistiques (5 stats pour les groupes
        # classiques, cellule vide pour Gain et CostRatio)
        row_hdr2 = [""]
        for _prefix, _label, n_s in metric_groups:
            if n_s == 5:
                for _suffix, s_lbl in _E9_STANDALONE_STAT_DISPLAY:
                    row_hdr2.append(Paragraph(f"<b>{s_lbl}</b>", hdr_style))
            else:
                row_hdr2.append("")  # cellule unique, pas de sous-stat
        table_data.append(row_hdr2)

        # --- Lignes de donnees
        for essai in _E9_ESSAIS_VALUES:
            sub_e = sub_m[sub_m["value"] == essai]
            row = [Paragraph(str(essai), cell_style)]
            if sub_e.empty:
                for _ in range(n_data_cols):
                    row.append(Paragraph("—", cell_style))
            else:
                r = sub_e.iloc[0]
                for prefix, _label, n_s in metric_groups:
                    if n_s == 5:
                        for suffix, _ in _E9_STANDALONE_STAT_DISPLAY:
                            col_name = f"{prefix}_{suffix}"
                            val = r.get(col_name, None)
                            row.append(Paragraph(_fmt(val, prefix), cell_style))
                    else:
                        val = r.get(prefix, None)
                        row.append(Paragraph(_fmt(val, prefix), cell_style))
            table_data.append(row)

    if not table_data:
        return False

    # Largeurs des colonnes (A4 paysage)
    page_w = landscape(A4)[0]
    available_w = page_w - 1.6 * cm
    nom_w = 1.8 * cm
    other_w = (available_w - nom_w) / max(1, n_data_cols)
    col_widths = [nom_w] + [other_w] * n_data_cols

    tbl = Table(table_data, colWidths=col_widths, repeatRows=0)

    # Commandes de style
    style_cmds = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
    ]

    # SPANs : 3 lignes d'en-tete par methode
    row_idx = 0
    for method in _E9_METHOD_ORDER_STANDALONE:
        sub_m = df_summary[df_summary["method"] == method]
        if sub_m.empty:
            continue

        # SPAN ligne 1 : nom de la methode sur les colonnes data
        style_cmds.append(("SPAN", (1, row_idx), (n_cols - 1, row_idx)))
        row_idx += 1

        # SPAN vertical sur "Nombre d'essais"
        style_cmds.append(("SPAN", (0, row_idx), (0, row_idx + 1)))

        # SPAN horizontal sur chaque groupe metrique (ligne 2)
        col_cursor = 1
        for _prefix, _label, n_s in metric_groups:
            if n_s > 1:
                style_cmds.append((
                    "SPAN",
                    (col_cursor, row_idx),
                    (col_cursor + n_s - 1, row_idx),
                ))
            col_cursor += n_s

        row_idx += 2  # 2 lignes d'en-tete apres la ligne methode
        row_idx += len(_E9_ESSAIS_VALUES)  # 4 lignes de donnees

    tbl.setStyle(TableStyle(style_cmds))

    # Caption
    caption_txt = (
        f"Tableau comparatif des méthodes Greedy et BD-CeNN selon le nombre "
        f"d'essais ({regime_label})"
    )

    # Note
    note_parts = [
        "<i>Note.</i> moy. = moyenne, é.t. = écart-type, méd. = médiane. "
        "Les temps ont été convertis en millisecondes (ms)."
    ]
    if is_adjacent:
        note_parts.append(
            "C<sub>CCI</sub> = conflits co-canal, "
            "C<sub>ACI</sub> = conflits adjacents, "
            "C<sub>tot</sub> = C<sub>CCI</sub> + C<sub>ACI</sub>. "
            "Gain(R) = (J<sub>1</sub> - J<sub>R</sub>) / J<sub>1</sub> × 100. "
            "CostRatio(R) = T<sub>R</sub> / T<sub>1</sub>."
        )
    else:
        note_parts.append(
            "Gain(R) = (J<sub>1</sub> - J<sub>R</sub>) / J<sub>1</sub> × 100. "
            "CostRatio(R) = T<sub>R</sub> / T<sub>1</sub>."
        )
    note_txt = " ".join(note_parts)

    story = [
        Paragraph(caption_txt, style_caption),
        tbl,
        Spacer(1, 0.3 * cm),
        Paragraph(note_txt, style_note),
    ]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=0.8 * cm,
        rightMargin=0.8 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
    )
    doc.build(story)
    return True

def _generate_e2_standalone_tables(mode: str):
    """
    [FIX-E2-TABLES] Genere les 8 PDFs standalone pour E2 sous un regime
    donne (4 methodes x 2 scenarios = 8 fichiers).

    Les PDFs sont places dans :
        results/validation/E2/{mode}/tables/
    avec la nomenclature :
        table_{methode}_{scenario}_{mode}.pdf
    """
    if not REPORTLAB_AVAILABLE:
        print("[WARNING] ReportLab indisponible : tables E2 standalone ignorees.")
        return

    csv_dir = config.CSV_DIR / "E2" / mode
    summary_path = csv_dir / "summary_metrics.csv"
    if not summary_path.exists():
        print(f"[INFO] Pas de summary_metrics.csv pour E2/{mode} : tables E2 ignorees.")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"[ERROR] Lecture {summary_path} impossible : {e}")
        return

    out_dir = config.VALIDATION_DIR / "E2" / mode / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["Random", "Greedy", "DSATUR", "BD-CeNN"]
    scenarios = [("S2", 30), ("S3", 50)]

    n_ok = 0
    for scenario_name, N in scenarios:
        for method in methods:
            fname = f"table_{method.lower().replace('-', '_')}_{scenario_name.lower()}_{mode}.pdf"
            out_path = out_dir / fname
            try:
                ok = _build_e2_standalone_table_pdf(
                    method_name=method,
                    scenario_name=scenario_name,
                    N=N,
                    mode=mode,
                    df_summary=df,
                    output_path=out_path,
                )
                if ok:
                    n_ok += 1
                    print(f"[SUCCESS] Table E2 standalone : {out_path.name}")
            except Exception as e:
                print(f"[ERROR] Echec generation {fname} : {e}")

    print(f"[INFO] E2/{mode} : {n_ok}/8 tables standalone generees dans {out_dir}")

def _generate_e3_standalone_tables(mode: str):
    """
    [FIX-E3-TABLES] Genere les 2 PDFs standalone pour E3 sous un regime
    donne : un pour le cout + conflits, un pour les canaux utilises.

    Les PDFs sont places dans :
        results/validation/E3/{mode}/tables/
    avec la nomenclature :
        table_e3_cost_conflict_{mode}.pdf
        table_e3_used_channels_{mode}.pdf

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".
    """
    if not REPORTLAB_AVAILABLE:
        print("[WARNING] ReportLab indisponible : tables E3 standalone ignorees.")
        return

    csv_dir = config.CSV_DIR / "E3" / mode
    summary_path = csv_dir / "summary_metrics.csv"
    if not summary_path.exists():
        print(f"[INFO] Pas de summary_metrics.csv pour E3/{mode} : tables E3 ignorees.")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"[ERROR] Lecture {summary_path} impossible : {e}")
        return

    out_dir = config.VALIDATION_DIR / "E3" / mode / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    table_types = ["cost_conflict", "used_channels"]
    n_ok = 0
    for table_type in table_types:
        fname = f"table_e3_{table_type}_{mode}.pdf"
        out_path = out_dir / fname
        try:
            ok = _build_e3_standalone_table_pdf(
                table_type=table_type,
                mode=mode,
                df_summary=df,
                output_path=out_path,
            )
            if ok:
                n_ok += 1
                print(f"[SUCCESS] Table E3 standalone : {out_path.name}")
        except Exception as e:
            print(f"[ERROR] Echec generation {fname} : {e}")

    print(f"[INFO] E3/{mode} : {n_ok}/2 tables standalone generees dans {out_dir}")

def _generate_e4_standalone_tables(mode: str):
    """
    [FIX-E4-TABLES] Genere le PDF standalone unique pour E4 sous un regime
    donne.

    Le PDF est place dans :
        results/validation/E4/{mode}/tables/
    avec la nomenclature :
        table_e4_{mode}.pdf

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".
    """
    if not REPORTLAB_AVAILABLE:
        print("[WARNING] ReportLab indisponible : table E4 standalone ignoree.")
        return

    csv_dir = config.CSV_DIR / "E4" / mode
    summary_path = csv_dir / "summary_metrics.csv"
    if not summary_path.exists():
        print(f"[INFO] Pas de summary_metrics.csv pour E4/{mode} : table E4 ignoree.")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"[ERROR] Lecture {summary_path} impossible : {e}")
        return

    out_dir = config.VALIDATION_DIR / "E4" / mode / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"table_e4_{mode}.pdf"
    try:
        ok = _build_e4_standalone_table_pdf(
            mode=mode,
            df_summary=df,
            output_path=out_path,
        )
        if ok:
            print(f"[SUCCESS] Table E4 standalone : {out_path.name}")
        else:
            print(f"[WARNING] Table E4 vide pour {mode}.")
    except Exception as e:
        print(f"[ERROR] Echec generation table E4/{mode} : {e}")

def _generate_e6_standalone_tables(mode: str):
    """
    [FIX-E6-TABLES] Genere le PDF standalone unique pour E6 sous un regime
    donne.

    Le PDF est place dans :
        results/validation/E6/{mode}/tables/
    avec la nomenclature :
        table_e6_{mode}.pdf

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".
    """
    if not REPORTLAB_AVAILABLE:
        print("[WARNING] ReportLab indisponible : table E6 standalone ignoree.")
        return

    csv_dir = config.CSV_DIR / "E6" / mode
    summary_path = csv_dir / "summary_metrics.csv"
    if not summary_path.exists():
        print(f"[INFO] Pas de summary_metrics.csv pour E6/{mode} : table E6 ignoree.")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"[ERROR] Lecture {summary_path} impossible : {e}")
        return

    out_dir = config.VALIDATION_DIR / "E6" / mode / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"table_e6_{mode}.pdf"
    try:
        ok = _build_e6_standalone_table_pdf(
            mode=mode,
            df_summary=df,
            output_path=out_path,
        )
        if ok:
            print(f"[SUCCESS] Table E6 standalone : {out_path.name}")
        else:
            print(f"[WARNING] Table E6 vide pour {mode}.")
    except Exception as e:
        print(f"[ERROR] Echec generation table E6/{mode} : {e}")

def _generate_e8_standalone_tables(mode: str):
    """
    [FIX-E8-TABLES] Genere le PDF standalone unique pour E8 sous un regime
    donne.

    Le PDF est place dans :
        results/validation/E8/{mode}/tables/
    avec la nomenclature :
        table_e8_{mode}.pdf

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".
    """
    if not REPORTLAB_AVAILABLE:
        print("[WARNING] ReportLab indisponible : table E8 standalone ignoree.")
        return

    csv_dir = config.CSV_DIR / "E8" / mode
    summary_path = csv_dir / "summary_metrics.csv"
    if not summary_path.exists():
        print(f"[INFO] Pas de summary_metrics.csv pour E8/{mode} : table E8 ignoree.")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"[ERROR] Lecture {summary_path} impossible : {e}")
        return

    out_dir = config.VALIDATION_DIR / "E8" / mode / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"table_e8_{mode}.pdf"
    try:
        ok = _build_e8_standalone_table_pdf(
            mode=mode,
            df_summary=df,
            output_path=out_path,
        )
        if ok:
            print(f"[SUCCESS] Table E8 standalone : {out_path.name}")
        else:
            print(f"[WARNING] Table E8 vide pour {mode}.")
    except Exception as e:
        print(f"[ERROR] Echec generation table E8/{mode} : {e}")

def _generate_e9_standalone_tables(mode: str):
    """
    [FIX-E9-TABLES] Genere le PDF standalone unique pour E9 sous un regime.
    """
    if not REPORTLAB_AVAILABLE:
        print("[WARNING] ReportLab indisponible : table E9 standalone ignoree.")
        return

    csv_dir = config.CSV_DIR / "E9" / mode
    summary_path = csv_dir / "summary_metrics.csv"
    if not summary_path.exists():
        print(f"[INFO] Pas de summary_metrics.csv pour E9/{mode} : table E9 ignoree.")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"[ERROR] Lecture {summary_path} impossible : {e}")
        return

    out_dir = config.VALIDATION_DIR / "E9" / mode / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"table_e9_{mode}.pdf"
    try:
        ok = _build_e9_standalone_table_pdf(
            mode=mode,
            df_summary=df,
            output_path=out_path,
        )
        if ok:
            print(f"[SUCCESS] Table E9 standalone : {out_path.name}")
        else:
            print(f"[WARNING] Table E9 vide pour {mode}.")
    except Exception as e:
        print(f"[ERROR] Echec generation table E9/{mode} : {e}")

# ============================================================================
# [FIX-E6-TABLES] Ordre d'affichage impose par le memoire pour les tables E6.
# Structure : 1 seule table par regime, 4 sections (Random, Greedy, DSATUR,
# BD-CeNN), 5 lignes par section (N = 20, 30, 50, 100, 200).
# Format A4 paysage.
# ============================================================================

_E6_STAT_DISPLAY = [
    ("mean",   "Moy."),
    ("std",    "É.T."),
    ("median", "Méd."),
    ("min",    "Min"),
    ("max",    "Max"),
]

# 5 valeurs de N explorees dans E6
_E6_N_VALUES = [20, 30, 50, 100, 200]

# Ordre d'affichage des methodes
_E6_METHOD_ORDER = ["Random", "Greedy", "DSATUR", "BD-CeNN"]


def _e6_metric_groups(mode: str):
    """
    [FIX-E6-TABLES] Retourne les groupes de metriques a afficher selon le
    regime. La table E6 est unique par regime.

    Parameters
    ----------
    mode : str
        "cochannel" ou "adjacent".

    Returns
    -------
    list of (str, str)
        Liste de tuples (prefixe_colonne, libelle_groupe).
    """
    if mode == "adjacent":
        return [
            ("cost",            "Coût"),
            ("conflicts_cci",   "Conflits C_CCI"),
            ("conflicts_aci",   "Conflits C_ACI"),
            ("conflicts_total", "Conflits C_tot"),
            ("used_channels",   "Canaux utilisés"),
            ("time",            "Temps d'exécution (ms)"),
            ("n_sweeps",        "Nombre de sweeps"),
        ]
    else:
        return [
            ("cost",            "Coût"),
            ("conflicts_total", "Conflit"),
            ("used_channels",   "Canaux utilisés"),
            ("time",            "Temps d'exécution (ms)"),
            ("n_sweeps",        "Nombre de sweeps"),
        ]

# ============================================================================
# [FIX-E8-TABLES] Ordre d'affichage impose par le memoire pour les tables E8.
# Structure : 1 seule table par regime, avec deux sections (Warm start et
# Cold start), 3 groupes de metriques (temps d'adaptation, nombre de
# reaffectations canaux, cout final) x 5 stats (moy., e.t., med., min, max).
# Format A4 paysage.
# ============================================================================

_E8_STAT_DISPLAY = [
    ("mean",   "moy."),
    ("std",    "é.t."),
    ("median", "méd."),
    ("min",    "min"),
    ("max",    "max"),
]

# 3 niveaux de modification explores dans E8 (scenario S7)
_E8_MOD_LEVELS = [5, 10, 20]

# Ordre d'affichage des modes (warm avant cold)
_E8_MODE_ORDER = ["warm", "cold"]

# Libelles d'affichage des modes
_E8_MODE_LABELS = {
    "warm": "Réoptimisation à chaud (Warm start)",
    "cold": "Réoptimisation à froid (Cold start)",
}


def _e8_metric_groups():
    """
    [FIX-E8-TABLES] Retourne les groupes de metriques a afficher pour E8.
    Chaque entree est un tuple (prefixe_colonne, libelle_groupe).

    Returns
    -------
    list of (str, str)
        Liste de tuples (prefixe_colonne, libelle_groupe).
    """
    return [
        ("time",    "Temps d'adaptation (ms)"),
        ("changes", "Nb. réaffectations canaux"),
        ("cost",    "Coût final"),
    ]

# ============================================================================
# [FIX-E9-TABLES] Generateur de PDF standalone pour E9.
# Structure : Tableau comparatif Greedy et BD-CeNN selon le nombre d'essais.
# Deux sections (Greedy, BD-CeNN), chacune avec 4 lignes (1, 5, 10, 20).
# Format A4 paysage.
# ============================================================================

_E9_STANDALONE_STAT_DISPLAY = [
    ("mean",   "moy."),
    ("std",    "é.t."),
    ("median", "méd."),
    ("min",    "min"),
    ("max",    "max"),
]

# 4 niveaux d'essais explores dans E9
_E9_ESSAIS_VALUES = [1, 5, 10, 20]

# Ordre d'affichage des methodes
_E9_METHOD_ORDER_STANDALONE = ["Greedy", "BD-CeNN"]


def _e9_standalone_metric_groups(mode: str):
    """
    [FIX-E9-TABLES] Retourne les groupes de metriques E9 selon le regime.
    Chaque entree est un tuple (prefixe_colonne, libelle_groupe, n_stat).
    Le troisieme element indique le nombre de colonnes du groupe (5 pour
    les stats classiques, 1 pour les metriques scalaires Gain et CostRatio).
    """
    if mode == "adjacent":
        return [
            ("cost",            "Coût",                5),
            ("conflicts_cci",   "C_CCI",               5),
            ("conflicts_aci",   "C_ACI",               5),
            ("conflicts_total", "C_tot",               5),
            ("time",            "Temps d'exécution (ms)", 5),
            ("gain_pct",        "Gain (%)",            1),
            ("cost_ratio",      "CostRatio",           1),
        ]
    else:
        return [
            ("cost",            "Coût",                5),
            ("conflicts_total", "Conflit",             5),
            ("time",            "Temps d'exécution (ms)", 5),
            ("gain_pct",        "Gain (%)",            1),
            ("cost_ratio",      "CostRatio",           1),
        ]

def build_pdf_report(
    experiment_id: str,
    mode: str,
    verifications: list,
    anomalies: list,
    all_tables: dict,
    output_path: Path
):
    """
    Genere le document d'audit PDF ReportLab.
    """
    if not REPORTLAB_AVAILABLE:
        print("[WARNING] ReportLab non installe. Le PDF d'audit n'a pas pu etre genere.")
        return

    max_cols = max([len(df.columns) for df, _ in all_tables.values()]) if all_tables else 0
    use_landscape = max_cols > 8 or experiment_id == "E10"
    target_pagesize = landscape(A4) if use_landscape else A4

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=target_pagesize,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm
    )

    styles = getSampleStyleSheet()

    style_univ = ParagraphStyle(
        "UnivKin", parent=styles["Normal"],
        fontSize=10, leading=13, textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    style_title = ParagraphStyle(
        "TitleCustom", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#1e3a5f"),
        alignment=TA_CENTER, spaceAfter=12, spaceBefore=20,
        fontName="Helvetica-Bold"
    )
    style_h1 = ParagraphStyle(
        "H1Custom", parent=styles["Heading1"],
        fontSize=13, textColor=colors.HexColor("#1e3a5f"),
        spaceAfter=8, spaceBefore=14, fontName="Helvetica-Bold"
    )
    style_h2 = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"],
        fontSize=11, textColor=colors.HexColor("#2a5285"),
        spaceAfter=6, spaceBefore=10, fontName="Helvetica-Bold"
    )
    style_body = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"],
        fontSize=9.5, leading=13, alignment=TA_JUSTIFY,
        fontName="Helvetica"
    )
    style_table_header = ParagraphStyle(
        "TableHeader", parent=styles["Normal"],
        fontSize=8, leading=10, textColor=colors.white,
        fontName="Helvetica-Bold", alignment=TA_CENTER
    )
    style_table_cell = ParagraphStyle(
        "TableCell", parent=styles["Normal"],
        fontSize=7.5, leading=9, fontName="Helvetica", alignment=TA_CENTER
    )
    style_small = ParagraphStyle(
        "SmallCustom", parent=styles["BodyText"],
        fontSize=8, leading=10, textColor=colors.HexColor("#64748b"),
    )
    style_meta_cell = ParagraphStyle(
        "MetaCell", parent=styles["Normal"],
        fontSize=9.5, leading=12, fontName="Helvetica", alignment=TA_LEFT
    )
    style_meta_label = ParagraphStyle(
        "MetaLabel", parent=style_meta_cell, fontName="Helvetica-Bold"
    )

    table_width = 24.0 * cm if use_landscape else 16.5 * cm

    story = []

    # -------------------------------------------------------------------------
    # PAGE 1 : Page de couverture
    # -------------------------------------------------------------------------
    story.append(Paragraph("UNIVERSITE DE KINSHASA", style_univ))
    story.append(Paragraph("FACULTE POLYTECHNIQUE", style_univ))
    story.append(Paragraph("DEPARTEMENT DE GENIE ELECTRIQUE ET INFORMATIQUE", style_univ))
    story.append(Spacer(1, 2.5 * cm))

    story.append(Paragraph("RAPPORT DE VALIDATION EXPERIMENTALE", style_title))
    mode_label = "CCI-only (Co-Canal)" if mode == "cochannel" else "CCI+ACI (Adjacent)"
    story.append(Paragraph(f"Experience {experiment_id} --- Mode : {mode_label}", style_univ))
    story.append(Spacer(1, 2.5 * cm))

    verdict_text = "CONFORME" if len(anomalies) == 0 else "NON CONFORME"
    verdict_color = "#15803d" if len(anomalies) == 0 else "#b91c1c"

    meta_data = [
        [Paragraph("Auteur", style_meta_label),
         Paragraph("Nelson MUKABI NGOMBO", style_meta_cell)],
        [Paragraph("Date de generation", style_meta_label),
         Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style_meta_cell)],
        [Paragraph("Version specification", style_meta_label),
         Paragraph("1.0 (Gelee)", style_meta_cell)],
        [Paragraph("Verdict global", style_meta_label),
         Paragraph(f"<font color='{verdict_color}'><b>{verdict_text}</b></font>", style_meta_cell)],
        [Paragraph("Anomalies detectees", style_meta_label),
         Paragraph(str(len(anomalies)), style_meta_cell)],
        [Paragraph("Fichiers audites", style_meta_label),
         Paragraph(str(len(all_tables)), style_meta_cell)],
    ]

    t_meta = Table(meta_data, colWidths=[6.0 * cm, table_width - 6.0 * cm])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_meta)

    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(
        "<i>Genere automatiquement par validate_results.py --- Version 1.0</i>",
        style_univ
    ))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 2 : Synthese executive
    # -------------------------------------------------------------------------
    story.append(Paragraph("Synthese executive", style_h1))
    story.append(Spacer(1, 0.2 * cm))

    total_inv = len(verifications)
    valid_inv = sum(1 for v in verifications if v.success)

    exec_text = (
        f"Ce document consigne l'audit d'integrite numerique de l'experience "
        f"<b>{experiment_id}</b> sous le regime d'interference <b>{mode_label}</b>.<br/><br/>"
        f"L'audit a valide <b>{total_inv}</b> invariants statistiques et numeriques "
        f"sur <b>{len(all_tables)}</b> fichiers CSV pour garantir la reproductibilite "
        f"des resultats du memoire."
    )
    story.append(Paragraph(exec_text, style_body))
    story.append(Spacer(1, 0.4 * cm))

    stats_data = [
        ["Indicateur de conformite", "Valeur"],
        ["Nombre d'invariants testes", str(total_inv)],
        ["Invariants valides", str(valid_inv)],
        ["Invariants en echec", str(total_inv - valid_inv)],
        ["Nombre total d'anomalies detectees", str(len(anomalies))],
        ["Nombre de fichiers audites", str(len(all_tables))],
    ]
    t_stats = Table(stats_data, colWidths=[10.0 * cm, table_width - 10.0 * cm])
    t_stats.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 0.6 * cm))

    if len(anomalies) == 0:
        rec_title = "RECOMMANDATION : INSERER SANS RESERVES"
        rec_color = "#15803d"
        rec_text = (
            "L'ensemble des invariants de coherence et des recalculs de cout a ete valide. "
            "Les resultats sont certifies conformes a la specification mathematique v1.0."
        )
    else:
        rec_title = "RECOMMANDATION : CORRECTIONS REQUISES"
        rec_color = "#b91c1c"
        rec_text = (
            f"L'audit a identifie {len(anomalies)} anomalie(s). "
            "Veuillez corriger les ecarts signales avant d'inserer ces donnees dans le memoire."
        )

    story.append(Paragraph(f"<b><font color='{rec_color}'>{rec_title}</font></b>", style_h2))
    story.append(Paragraph(rec_text, style_body))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 3 : Detail des invariants
    # -------------------------------------------------------------------------
    story.append(Paragraph("Detail des verifications d'invariants", style_h1))
    story.append(Spacer(1, 0.2 * cm))

    inv_table_data = [["Code", "Invariant teste", "Statut"]]
    for v in verifications:
        stat_char = "Valide" if v.success else "Echec"
        inv_table_data.append([v.code, v.desc, stat_char])

    t_inv = Table(inv_table_data, colWidths=[2.5 * cm, table_width - 5.5 * cm, 3.0 * cm])
    t_inv.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_inv)

    if anomalies:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Anomalies relevees :", style_h2))
        for anomaly in anomalies[:30]:
            story.append(Paragraph(f"<font color='#b91c1c'>-</font> {anomaly}", style_body))
        if len(anomalies) > 30:
            story.append(Paragraph(
                f"<i>... et {len(anomalies) - 30} autres anomalies non affichees.</i>",
                style_small
            ))

    # -------------------------------------------------------------------------
    # PAGES SUIVANTES : TABLEAUX DE DONNEES CERTIFIES
    # -------------------------------------------------------------------------
    if all_tables:
        story.append(PageBreak())
        story.append(Paragraph("Tableaux de donnees certifies", style_h1))
        story.append(Spacer(1, 0.2 * cm))

        legend_text = (
            "<i>Nomenclature : Moy. = Moyenne, E.T. = Ecart-type, "
            "C_CCI = Conflits co-canal, C_ACI = Conflits adjacents, "
            "C_tot = Conflits totaux, t_exec = Temps d'execution (en ms), "
            "Sweeps = Balayages complets, K_used = Canaux utilises.</i>"
        )
        story.append(Paragraph(legend_text, style_small))
        story.append(Spacer(1, 0.4 * cm))

        table_counter = 0

        for filename, (df, csv_type) in all_tables.items():
            # [FIX-E9] Detection du motif specifique a E9 : colonnes "value" et
            # "method" avec methodes dans {"BD-CeNN", "Greedy"}. On rend alors
            # un tableau a 3 niveaux d'en-tetes PAR methode, sans passer par
            # _split_wide_dataframe (qui fragmenterait inutilement le tableau).
            is_e9_pattern = (
                csv_type == "summary"
                and "value" in df.columns
                and "method" in df.columns
                and set(df["method"].dropna().unique()).issubset({"BD-CeNN", "Greedy"})
            )

            if is_e9_pattern:
                for method_name in ["Greedy", "BD-CeNN"]:
                    method_df = (
                        df[df["method"] == method_name]
                        .drop(columns=["method"])
                        .reset_index(drop=True)
                    )
                    if method_df.empty:
                        continue
                    table_counter += 1
                    flowables = _render_e9_table_for_mode( 
                        sub_df=method_df,
                        method_name=method_name,
                        table_counter=table_counter,
                        mode=mode,                           
                        style_h2=style_h2,
                        style_table_header=style_table_header,
                        style_table_cell=style_table_cell,
                        table_width=table_width,
                        filename=filename,
                    )
                    story.extend(flowables)
                continue

            # [FIX-E8] Detection d'une colonne "mode" (warm/cold) : si presente,
            # le tableau source est scinde en deux sous-tableaux distincts,
            # l'un pour le warm start, l'autre pour le cold start.
            mode_splits = None
            if "mode" in df.columns:
                unique_modes = set(df["mode"].dropna().unique())
                if unique_modes and unique_modes <= {"warm", "cold"}:
                    mode_splits = []
                    for mode_val in ["warm", "cold"]:
                        mode_df = (df[df["mode"] == mode_val]
                                   .drop(columns=["mode"])
                                   .reset_index(drop=True))
                        if mode_val == "warm":
                            mode_label_split = "Réoptimisation à chaud (Warm start)"
                        else:
                            mode_label_split = "Réoptimisation à froid (Cold start)"
                        mode_splits.append((mode_df, mode_label_split))

            dfs_to_render = mode_splits if mode_splits else [(df, None)]

            for split_df, split_label in dfs_to_render:
                # [FIX-E8] Pour les tableaux E8 scindes, on desactive la division
                # supplementaire en sous-blocs : la totalite des colonnes tient
                # sur une seule ligne (mode paysage A4).
                block_max_cols = 25 if mode_splits else 11
                blocks = _split_wide_dataframe(split_df, max_cols=block_max_cols)

                for block_idx, (sub_df, suffix) in enumerate(blocks):
                    table_counter += 1

                    type_labels = {
                        "summary": "Tableau de synthese statistique",
                        "metrics": "Tableau de metriques par methode",
                        "allocations": "Grille d'allocation cellule-canal",
                        "raw_summary": "Resume statistique des donnees brutes",
                        "trajectory_summary": "Resume des trajectoires de convergence",
                    }
                    type_label = type_labels.get(csv_type, "Tableau de donnees")

                    # [FIX-E8] Le titre integre le label warm/cold si applicable.
                    if split_label:
                        title_html = (
                            f"<b>Tableau {table_counter} --- {type_label} : "
                            f"{split_label}{suffix}</b><br/>"
                            f"<i>Fichier source : {filename}</i>"
                        )
                    else:
                        title_html = (
                            f"<b>Tableau {table_counter} --- {type_label}{suffix}</b><br/>"
                            f"<i>Fichier source : {filename}</i>"
                        )

                    story.append(Paragraph(title_html, style_h2))
                    story.append(Spacer(1, 0.15 * cm))

                    cols = list(sub_df.columns)
                    n_cols = len(cols)

                    renamed_cols = [_abbreviate_column(col) for col in cols]

                    # [FIX-E8] Labels de groupe specifiques pour E8.
                    custom_labels = None
                    if mode_splits:
                        custom_labels = {
                            "time": "Temps d'adaptation (ms)",
                            "changes": "Nb. réaffectations canaux",
                            "cost": "Coût final",
                        }

                    # [FIX-E7] Detection des groupes statistiques pour construire
                    # un en-tete a 2 niveaux (groupe fusionne + sous-en-tetes).
                    groups, parsed_cols = _detect_column_groups(
                        cols, labels_override=custom_labels
                    )

                    table_rows = []

                    if n_cols > 8:
                        cell_style = ParagraphStyle(
                            f"CellSmall_{table_counter}", parent=style_table_cell,
                            fontSize=6.5, leading=8
                        )
                        header_style = ParagraphStyle(
                            f"HeaderSmall_{table_counter}", parent=style_table_header,
                            fontSize=6.5, leading=8
                        )
                    else:
                        cell_style = style_table_cell
                        header_style = style_table_header

                    # [FIX-E7] Construction de l'en-tete a 2 niveaux si groupes detectes
                    if groups:
                        header_top = [""] * n_cols
                        header_bot = [""] * n_cols
                        for g in groups:
                            header_top[g["start"]] = g["label"]
                            for k in range(g["start"], g["end"] + 1):
                                header_bot[k] = _STAT_SUFFIX_LABELS[parsed_cols[k][1]]
                        # Colonnes standalone : on reutilise l'abreviation
                        for k in range(n_cols):
                            if parsed_cols[k] is None:
                                header_top[k] = renamed_cols[k]

                        table_rows.append(
                            [Paragraph(f"<b>{c}</b>", header_style) for c in header_top]
                        )
                        table_rows.append(
                            [Paragraph(f"<b>{c}</b>", header_style) for c in header_bot]
                        )
                        n_header_rows = 2
                    else:
                        header_p = [Paragraph(f"<b>{col}</b>", header_style) for col in renamed_cols]
                        table_rows.append(header_p)
                        n_header_rows = 1

                    # Limite max de 200 lignes pour summary (aucune troncature sur E2)
                    max_rows = 200 if csv_type == "summary" else (60 if csv_type == "metrics" else 30)
                    display_df = sub_df.head(max_rows)

                    for _, row in display_df.iterrows():
                        row_p = []
                        for col_name in cols:
                            val = row[col_name]
                            formatted = _format_cell_value(val, col_name)
                            row_p.append(Paragraph(formatted, cell_style))
                        table_rows.append(row_p)

                    col_width = table_width / max(1, n_cols)
                    # En-tete repete automatiquement en cas de passage sur plusieurs pages
                    t_data = Table(
                        table_rows,
                        colWidths=[col_width] * n_cols,
                        repeatRows=n_header_rows
                    )

                    # [FIX-E7] ROWBACKGROUNDS demarre apres l'en-tete (1 ou 2 lignes)
                    # et on ajoute les commandes SPAN pour fusionner les cellules de groupe.
                    style_cmds = [
                        ("BACKGROUND", (0, 0), (-1, n_header_rows - 1),
                         colors.HexColor("#1e3a5f")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ("ROWBACKGROUNDS", (0, n_header_rows), (-1, -1),
                         [colors.white, colors.HexColor("#f8fafc")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                    for g in groups:
                        style_cmds.append(("SPAN", (g["start"], 0), (g["end"], 0)))

                    t_data.setStyle(TableStyle(style_cmds))

                    story.append(t_data)

                    if len(sub_df) > max_rows:
                        story.append(Paragraph(
                            f"<i>({len(sub_df) - max_rows} lignes supplementaires non affichees)</i>",
                            style_small
                        ))

                    story.append(Spacer(1, 0.4 * cm))

    # =========================================================================
    # DERNIERE PAGE : CLOTURE
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Conclusion et signature", style_h1))
    story.append(Spacer(1, 0.3 * cm))

    conclusion_text = (
        f"Le present rapport certifie l'integrite numerique des resultats de "
        f"l'experience <b>{experiment_id}</b> sous le regime <b>{mode_label}</b>.<br/><br/>"
        f"<b>Verdict final : <font color='{verdict_color}'>{verdict_text}</font></b><br/><br/>"
    )
    if anomalies:
        conclusion_text += (
            f"<b>{len(anomalies)}</b> anomalie(s) ont ete detectees. "
            f"Les corrections doivent etre appliquees avant toute insertion "
            f"dans le manuscrit du memoire."
        )
    else:
        conclusion_text += (
            "Aucune anomalie n'a ete detectee. Les resultats peuvent etre "
            "integres dans le manuscrit sans reserve."
        )
    story.append(Paragraph(conclusion_text, style_body))

    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(
        "<i>Rapport genere automatiquement par validate_results.py --- Version 1.0<br/>"
        "Universite de Kinshasa, Faculte Polytechnique</i>",
        style_univ
    ))

    doc.build(story)


# ============================================================================
# 5. ORCHESTRATEUR PRINCIPAL D'AUDIT
# ============================================================================

def validate_experiment(experiment_id: str, modes: list = None):
    """
    Decouvre les fichiers de sortie d'une experience et execute la certification.

    Traitement adaptatif par type de CSV :
        - summary_metrics.csv, method_metrics.csv, convergence_metrics.csv,
          llm_fidelity_evaluation.csv : reproduction integrale
        - cell_channels.csv : formatage en grille d'allocation
        - trajectory_*.csv : verification des invariants uniquement
          (les tableaux [Points cles] sont volontairement exclus du PDF)
    """
    if modes is None:
        modes = ["cochannel", "adjacent"]

    for mode in modes:
        csv_dir = config.CSV_DIR / experiment_id / mode
        if not csv_dir.exists():
            continue

        print(f"[AUDIT] Initialisation du protocole de conformite pour {experiment_id} ({mode})...")

        verifications = []
        anomalies = []
        all_tables = {}

        # Classification intelligente des fichiers
        classified = classify_csv_files(csv_dir)

        # -------------------------------------------------------------------
        # A. TRAITEMENT DES FICHIERS DE SYNTHESE (summary_metrics.csv)
        # -------------------------------------------------------------------
        for csv_path in classified["summary"]:
            try:
                df = pd.read_csv(csv_path)
                all_tables[csv_path.name] = (df, "summary")

                # [V1] min <= median <= max
                for prefix in ["cost", "conflicts_total", "conflicts_cci", "conflicts_aci"]:
                    min_col = f"{prefix}_min"
                    max_col = f"{prefix}_max"
                    med_col = f"{prefix}_median"

                    if all(c in df.columns for c in [min_col, med_col, max_col]):
                        v1_ok = True
                        for idx, row in df.iterrows():
                            if (pd.notna(row[min_col]) and pd.notna(row[med_col])
                                    and pd.notna(row[max_col])):
                                if not (row[min_col] <= row[med_col] <= row[max_col]):
                                    v1_ok = False
                                    anomalies.append(
                                        f"{csv_path.name} ligne {idx}: [V1] "
                                        f"Echec inegalite min ({row[min_col]}) "
                                        f"<= med ({row[med_col]}) <= max ({row[max_col]})"
                                    )
                        verifications.append(VerificationResult(
                            "V1",
                            f"Inegalite min <= mediane <= max ({csv_path.name} - {prefix})",
                            v1_ok
                        ))

                # [V2] std >= 0
                for col in df.columns:
                    if col.endswith("_std"):
                        valid_vals = df[col].dropna()
                        v2_ok = (valid_vals >= 0).all() if len(valid_vals) > 0 else True
                        if not v2_ok:
                            anomalies.append(f"{csv_path.name}: [V2] Ecart-type negatif dans {col}")
                        verifications.append(VerificationResult(
                            "V2", f"Ecart-type >= 0 ({csv_path.name} - {col})", v2_ok
                        ))

                # [V3] std == 0 => min == max
                for prefix in ["cost", "conflicts_total"]:
                    std_col = f"{prefix}_std"
                    min_col = f"{prefix}_min"
                    max_col = f"{prefix}_max"
                    if all(c in df.columns for c in [std_col, min_col, max_col]):
                        v3_ok = True
                        for idx, row in df.iterrows():
                            if (pd.notna(row[std_col]) and pd.notna(row[min_col])
                                    and pd.notna(row[max_col])):
                                if row[std_col] == 0 and row[min_col] != row[max_col]:
                                    v3_ok = False
                                    anomalies.append(
                                        f"{csv_path.name} ligne {idx}: [V3] "
                                        f"std=0 mais min != max pour {prefix}"
                                    )
                        verifications.append(VerificationResult(
                            "V3",
                            f"Coherence std==0 => min==max ({csv_path.name} - {prefix})",
                            v3_ok
                        ))

            except Exception as e:
                anomalies.append(f"Erreur d'analyse sur {csv_path.name}: {e}")

        # -------------------------------------------------------------------
        # B. TRAITEMENT DES FICHIERS DE METRIQUES (method_metrics.csv, etc.)
        # -------------------------------------------------------------------
        for csv_path in classified["metrics"]:
            try:
                df = pd.read_csv(csv_path)
                all_tables[csv_path.name] = (df, "metrics")

                fname = csv_path.name

                # [V1bis] Cout strictement positif ou nul
                cost_col = None
                if "Cout" in df.columns:
                    cost_col = "Cout"
                elif "cost" in df.columns:
                    cost_col = "cost"
                elif "cost_mean" in df.columns:
                    cost_col = "cost_mean"

                if cost_col:
                    v1b_ok = (df[cost_col].dropna() >= 0).all()
                    if not v1b_ok:
                        anomalies.append(f"{fname}: [V1bis] Cout negatif detecte")
                    verifications.append(VerificationResult(
                        "V1bis", f"Cout >= 0 ({fname})", v1b_ok
                    ))

                # [V7] Coherence conflicts_total = conflicts_cci + conflicts_aci
                if all(c in df.columns for c in
                       ["conflicts_cci", "conflicts_aci", "conflicts_total"]):
                    v7_ok = True
                    for idx, row in df.iterrows():
                        if all(pd.notna(row[c]) for c in
                               ["conflicts_cci", "conflicts_aci", "conflicts_total"]):
                            expected = int(row["conflicts_cci"]) + int(row["conflicts_aci"])
                            if int(row["conflicts_total"]) != expected:
                                v7_ok = False
                                anomalies.append(
                                    f"{fname} ligne {idx}: [V7] "
                                    f"conflicts_total ({row['conflicts_total']}) != "
                                    f"cci ({row['conflicts_cci']}) + aci ({row['conflicts_aci']})"
                                )
                    verifications.append(VerificationResult(
                        "V7", f"Coherence C_tot = C_CCI + C_ACI ({fname})", v7_ok
                    ))

                # [V8bis] Une seule seed attendue sur E1
                if experiment_id == "E1" and "seed" in df.columns:
                    unique_seeds = df["seed"].dropna().unique()
                    v8b_ok = len(unique_seeds) == 1
                    if not v8b_ok:
                        anomalies.append(f"{fname}: [V8bis] Seeds multiples sur E1 : {unique_seeds}")
                    verifications.append(VerificationResult(
                        "V8bis", f"Seed unique sur E1 ({fname})", v8b_ok
                    ))

                # [V9bis] En mode cochannel, ACI doit etre nul
                if mode == "cochannel" and "conflicts_aci" in df.columns:
                    v9b_ok = (df["conflicts_aci"].dropna() == 0).all()
                    if not v9b_ok:
                        anomalies.append(f"{fname}: [V9bis] En mode CCI-only, conflicts_aci doit etre 0")
                    verifications.append(VerificationResult(
                        "V9bis", f"ACI nul en mode cochannel ({fname})", v9b_ok
                    ))

                # [V10] used_channels <= K
                if "used_channels" in df.columns:
                    max_used = df["used_channels"].max()
                    verifications.append(VerificationResult(
                        "V10", f"Canaux utilises max = {int(max_used)} ({fname})", True
                    ))

                # =====================================================
                # [FIX-E10] Invariant specifique E10 : coherence de la
                # decomposition C_tot = C_CCI + C_ACI pour le CSV
                # solver_metrics_e10.csv. Ce CSV contient 20 lignes
                # (une par cas) au lieu d'une agregation, donc la
                # verification ligne a ligne est directe.
                # =====================================================
                if "solver_metrics" in fname:
                    if all(c in df.columns for c in
                           ["conflicts_cci", "conflicts_aci", "conflicts_total"]):
                        v_e10_ok = True
                        for idx, row in df.iterrows():
                            expected = int(row["conflicts_cci"]) + int(row["conflicts_aci"])
                            if int(row["conflicts_total"]) != expected:
                                v_e10_ok = False
                                anomalies.append(
                                    f"{fname} ligne {idx}: [V10-E10] "
                                    f"C_tot ({row['conflicts_total']}) != "
                                    f"C_CCI + C_ACI ({expected})"
                                )
                        verifications.append(VerificationResult(
                            "V10-E10",
                            f"Coherence C_tot = C_CCI + C_ACI pour E10 ({fname})",
                            v_e10_ok,
                        ))

                    # [FIX-E10] Invariant supplementaire : sweeps >= 0
                    if "sweeps" in df.columns:
                        v_sweeps_ok = (df["sweeps"].dropna() >= 0).all()
                        verifications.append(VerificationResult(
                            "V10-SWEEPS",
                            f"Sweeps >= 0 ({fname})",
                            v_sweeps_ok,
                        ))

            except Exception as e:
                anomalies.append(f"Erreur d'analyse sur {csv_path.name}: {e}")

                                # [FIX-E10] Invariant specifique E10 : coherence de la
                # decomposition C_tot = C_CCI + C_ACI pour le CSV solver_metrics_e10.
                # Ce CSV contient 20 lignes (une par cas) au lieu d'une
                # agregation, donc la verification ligne a ligne est directe.
                if "solver_metrics" in fname:
                    if all(c in df.columns for c in
                           ["conflicts_cci", "conflicts_aci", "conflicts_total"]):
                        v_e10_ok = True
                        for idx, row in df.iterrows():
                            expected = int(row["conflicts_cci"]) + int(row["conflicts_aci"])
                            if int(row["conflicts_total"]) != expected:
                                v_e10_ok = False
                                anomalies.append(
                                    f"{fname} ligne {idx}: [V10-E10] "
                                    f"C_tot ({row['conflicts_total']}) != "
                                    f"C_CCI + C_ACI ({expected})"
                                )
                        verifications.append(VerificationResult(
                            "V10-E10",
                            f"Coherence C_tot = C_CCI + C_ACI pour E10 ({fname})",
                            v_e10_ok,
                        ))

                    # [FIX-E10] Invariant supplementaire : sweeps >= 0
                    if "sweeps" in df.columns:
                        v_sweeps_ok = (df["sweeps"].dropna() >= 0).all()
                        verifications.append(VerificationResult(
                            "V10-SWEEPS",
                            f"Sweeps >= 0 ({fname})",
                            v_sweeps_ok,
                        ))

        # -------------------------------------------------------------------
        # C. TRAITEMENT DES FICHIERS D'ALLOCATIONS (cell_channels.csv)
        # -------------------------------------------------------------------
        for csv_path in classified["allocations"]:
            try:
                df = pd.read_csv(csv_path)
                all_tables[csv_path.name] = (df, "allocations")

                fname = csv_path.name

                # [V11] Tous les canaux doivent etre >= 0
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in ("Cell", "seed"):
                        continue
                    v11_ok = (df[col].dropna() >= 0).all()
                    if not v11_ok:
                        anomalies.append(f"{fname}: [V11] Canal negatif detecte dans {col}")
                    verifications.append(VerificationResult(
                        "V11", f"Canaux >= 0 ({fname} - {col})", v11_ok
                    ))

                # [V12] Coherence de l'index de cellule
                if "Cell" in df.columns:
                    expected_cells = list(range(len(df)))
                    actual_cells = df["Cell"].tolist()
                    v12_ok = expected_cells == actual_cells
                    if not v12_ok:
                        anomalies.append(f"{fname}: [V12] Index de cellules non consecutif")
                    verifications.append(VerificationResult(
                        "V12", f"Index de cellules coherent ({fname})", v12_ok
                    ))

            except Exception as e:
                anomalies.append(f"Erreur d'analyse sur {csv_path.name}: {e}")

        # -------------------------------------------------------------------
        # D. TRAITEMENT DES FICHIERS BRUTS (raw_metrics.csv)
        # -------------------------------------------------------------------
        for csv_path in classified["raw"]:
            try:
                df = pd.read_csv(csv_path)
                fname = csv_path.name

                # [V8] Uniformite des seeds inter-methodes
                if "seed" in df.columns and "method" in df.columns:
                    seeds_by_method = df.groupby("method")["seed"].apply(set)
                    methods_list = list(seeds_by_method.index)
                    if len(methods_list) > 1:
                        ref_s = seeds_by_method[methods_list[0]]
                        v8_ok = all(seeds_by_method[m] == ref_s for m in methods_list)
                        if not v8_ok:
                            anomalies.append(f"{fname}: [V8] Graines non uniformes entre methodes")
                        verifications.append(VerificationResult(
                            "V8", f"Graines identiques inter-methodes ({fname})", v8_ok
                        ))

                # [V13] Nombre de seeds >= 30 pour les experiences agregees
                if "seed" in df.columns and experiment_id not in ("E1",):
                    n_seeds = df["seed"].nunique()
                    v13_ok = n_seeds >= 30
                    if not v13_ok:
                        anomalies.append(f"{fname}: [V13] Nombre de seeds insuffisant : {n_seeds} < 30")
                    verifications.append(VerificationResult(
                        "V13", f"Nombre de seeds >= 30 ({fname}) : {n_seeds} seeds", v13_ok
                    ))

                # [V14] Coherence des couts (>= 0)
                if "cost" in df.columns:
                    v14_ok = (df["cost"].dropna() >= 0).all()
                    if not v14_ok:
                        anomalies.append(f"{fname}: [V14] Couts negatifs detectes")
                    verifications.append(VerificationResult(
                        "V14", f"Couts >= 0 ({fname})", v14_ok
                    ))

            except Exception as e:
                anomalies.append(f"Erreur d'analyse sur {csv_path.name}: {e}")

        # -------------------------------------------------------------------
        # E. TRAITEMENT DES FICHIERS DE TRAJECTOIRE (trajectory_*.csv)
        # -------------------------------------------------------------------
        # NOTE : les invariants V15 et V16 sont verifies, mais AUCUN tableau
        # de type [Points cles] n'est ajoute a all_tables. Les trajectoires
        # ne sont donc plus reproduites dans le PDF d'audit.
        # -------------------------------------------------------------------
        for csv_path in classified["trajectory"]:
            try:
                df = pd.read_csv(csv_path)
                fname = csv_path.name

                # [V15] Coherence des trajectoires (mean_cost monotone decroissant)
                if "mean_cost" in df.columns and len(df) > 1:
                    first_cost = df["mean_cost"].iloc[0]
                    last_cost = df["mean_cost"].iloc[-1]
                    v15_ok = last_cost <= first_cost
                    if not v15_ok:
                        anomalies.append(f"{fname}: [V15] Trajectoire non decroissante")
                    verifications.append(VerificationResult(
                        "V15", f"Trajectoire decroissante ({fname})", v15_ok
                    ))

                # [V16] std_cost >= 0
                if "std_cost" in df.columns:
                    v16_ok = (df["std_cost"].dropna() >= 0).all()
                    if not v16_ok:
                        anomalies.append(f"{fname}: [V16] std_cost negatif detecte")
                    verifications.append(VerificationResult(
                        "V16", f"std_cost >= 0 ({fname})", v16_ok
                    ))

            except Exception as e:
                anomalies.append(f"Erreur d'analyse sur {csv_path.name}: {e}")

        # -------------------------------------------------------------------
        # F. TRAITEMENT DES AUTRES FICHIERS
        # -------------------------------------------------------------------
        for csv_path in classified["other"]:
            try:
                df = pd.read_csv(csv_path)
                all_tables[csv_path.name] = (df, "metrics")
                verifications.append(VerificationResult(
                    "V17", f"Fichier detecte et charge ({csv_path.name})", True
                ))
            except Exception as e:
                anomalies.append(f"Erreur d'analyse sur {csv_path.name}: {e}")

        # Verification globale de l'invariant total (fallback)
        if not verifications:
            verifications.append(VerificationResult("EMPTY", "Detection de fichiers de donnees", True))

        # Generation du PDF d'audit
        validation_sub_dir = config.VALIDATION_DIR / experiment_id / mode
        validation_sub_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = validation_sub_dir / f"validation_report_{experiment_id}_{mode}.pdf"

        try:
            build_pdf_report(
                experiment_id=experiment_id,
                mode=mode,
                verifications=verifications,
                anomalies=anomalies,
                all_tables=all_tables,
                output_path=pdf_path
            )
            print(f"[SUCCESS] Rapport de conformite genere : {pdf_path}")
        except Exception as e:
            print(f"[ERROR] Echec de generation du PDF pour {experiment_id} ({mode}) : {e}")

        # [FIX-E2-TABLES] Generation des 8 tables standalone pour E2
        # (utilisees via \includepdf dans le memoire LaTeX pour eviter
        # toute hallucination LLM sur les valeurs numeriques).
        if experiment_id == "E2":
            _generate_e2_standalone_tables(mode)

        # [FIX-E3-TABLES] Generation des 2 tables standalone pour E3
        # (A4 portrait, avec colonnes C_CCI / C_ACI / C_tot en mode adjacent).
        elif experiment_id == "E3":
            _generate_e3_standalone_tables(mode)

        # [FIX-E4-TABLES] Generation de la table standalone unique pour E4
        # (A4 paysage, structure par methode et par densite, colonnes
        # C_CCI / C_ACI / C_tot en mode adjacent).
        elif experiment_id == "E4":
            _generate_e4_standalone_tables(mode)


        # [FIX-E6-TABLES] Generation de la table standalone unique pour E6
        # (A4 paysage, structure par methode et par valeur de N, colonnes
        # C_CCI / C_ACI / C_tot en mode adjacent).
        elif experiment_id == "E6":
            _generate_e6_standalone_tables(mode)

        # [FIX-E8-TABLES] Generation de la table standalone unique pour E8
        # (A4 paysage, structure warm/cold avec 3 groupes de metriques).
        elif experiment_id == "E8":
            _generate_e8_standalone_tables(mode)

        # [FIX-E9-TABLES] Generation de la table standalone unique pour E9
        # (A4 paysage, structure Greedy/BD-CeNN avec colonnes C_CCI / C_ACI /
        # C_tot en mode adjacent et metriques Gain / CostRatio).
        elif experiment_id == "E9":
            _generate_e9_standalone_tables(mode)
# ============================================================================
# 6. POINT D'ENTREE AUTONOME
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python validate_results.py <experiment_id> [mode1 mode2 ...]")
        sys.exit(1)

    exp_id = sys.argv[1]
    modes_arg = sys.argv[2:] if len(sys.argv) > 2 else None

    print(f"[MANUAL AUDIT] Relance de l'audit pour {exp_id}...")
    validate_experiment(experiment_id=exp_id, modes=modes_arg)
    print("[DONE] Audit termine.")