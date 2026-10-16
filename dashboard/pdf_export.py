"""
LLM PDF report generator for the interactive dashboard.
Reproduces exactly the E10 audit PDF format from llm_assistant.py:
    - Prompt sent to LLM
    - Raw LLM response
    - Automatic numerical audit table
    - Corrected response (if hallucinations detected)
    - Certification badge

One PDF per interference mode (CCI-only and CCI+ACI).
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_llm_pdf(
    case_data: dict,
    prompt: str,
    raw_response: str,
    verification_initial: dict,
    corrected_response: str,
    verification_final: dict,
    output_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    verification_struct: Optional[dict] = None,
    semantic: Optional[dict] = None,
) -> Optional[bytes]:
    """
    Generates a professional PDF audit report for a single interference mode.
    Mirrors the exact structure of E10's generate_pdf_report().

    Args:
        case_data: Case data dict (same structure as E10)
        prompt: Full prompt text sent to the LLM
        raw_response: Initial LLM response text
        verification_initial: Verification dict from verify_numbers() on raw response
        corrected_response: Corrected response after regeneration (may equal raw)
        verification_final: Verification dict on final response
        output_path: Optional file path (if None, returns bytes)
        model_name: LLM model identifier for header

    Returns:
        bytes: PDF content if output_path is None, else None (writes to disk)
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
        return None

    # -------------------------------------------------------------------
    # Styles setup (identical to E10)
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # Document setup
    # -------------------------------------------------------------------
    if output_path is None:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=1.8*cm, bottomMargin=1.8*cm,
            title=f"Rapport LLM - {case_data.get('case_id')}",
        )
    else:
        doc = SimpleDocTemplate(
            str(output_path), pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=1.8*cm, bottomMargin=1.8*cm,
            title=f"Rapport LLM - {case_data.get('case_id')}",
        )

    story = []

    model_label = case_data.get("model_label", "CCI-only")
    conflicts_label = _conflicts_label(model_label)

    # -------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------
    story.append(Paragraph(
        f"Rapport d'audit LLM - Attribution de canaux ({model_label})",
        style_title
    ))
    story.append(Spacer(1, 0.3*cm))

    # -------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------
    from llm_assistant import get_llm_inference_params, numeric_certification

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
        ["Mode d'interference", model_label],
        ["Contexte", case_data.get("context", "-")],
        ["Version du code", run_meta.get("code_version", "-")],
        ["Modele LLM", model_name if model_name else "Non specifie"],
        ["Inference", f"temperature = {llm_params['temperature']}, top_p = {llm_params['top_p']}, "
                      f"top_k = {llm_params['top_k']}, max_output_tokens = "
                      f"{llm_params['max_output_tokens']}"],
        ["Directive", llm_params["directive_version"]],
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

    # -------------------------------------------------------------------
    # Certification badge
    # -------------------------------------------------------------------
    # Statut numérique combiné (whitelist + quadruplets), puis statut sémantique
    numeric = numeric_certification(verification_final, verification_struct)
    if numeric["status"] == "NON_CERTIFIE":
        badge = (
            f'<font color="#b30000"><b>AUDIT NUMERIQUE : NON CERTIFIE</b></font> - '
            f'{numeric["invented_count"]} valeur(s) hors whitelist, '
            f'{numeric["invalid_quadruplets"]} quadruplet(s) mal attribue(s).'
        )
    elif numeric["status"] == "AVERTISSEMENT":
        badge = ('<font color="#b45309"><b>AUDIT NUMERIQUE : AVERTISSEMENT</b></font> - '
                 'aucune valeur numerique a verifier.')
    else:
        badge = (
            f'<font color="#006600"><b>AUDIT NUMERIQUE : CERTIFIE</b></font> '
            f'(taux d\'exactitude : {numeric["accuracy_rate"]:.1f}%, '
            f'{numeric["checked_quadruplets"]} quadruplet(s) conforme(s)).'
        )
    story.append(Paragraph(badge, style_body))
    if semantic:
        story.append(Paragraph(
            f'<b>AUDIT SEMANTIQUE : {semantic["status"]}</b> - score {semantic["score"]:.1f}% '
            f'({semantic["version"]}, non bloquant).', style_body))
    story.append(Spacer(1, 0.4*cm))

    # -------------------------------------------------------------------
    # 1. Simulation data
    # -------------------------------------------------------------------
    story.append(Paragraph("1. Donnees de simulation utilisees", style_h1))
    m = case_data.get("metrics", {})
    sim_data = [
        ["Metrique", "Valeur"],
        ["Cout initial J_0", _fmt(m.get("cost_initial"))],
        ["Cout final J(x)", _fmt(m.get("cost_final"))],
        *_conflict_rows(model_label, m),
        ["Temps d'execution (ms)", _fmt_us(m.get("time_seconds"))],
        ["Sweeps (BD-CeNN)", _fmt(m.get("sweeps"))],
        ["Canaux utilises", _fmt(m.get("used_channels"))],
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

    # -------------------------------------------------------------------
    # 2. Prompt sent to LLM
    # -------------------------------------------------------------------
    story.append(Paragraph("2. Prompt envoye au LLM", style_h1))
    prompt_escaped = prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(prompt_escaped.replace("\n", "<br/>"), style_mono))
    story.append(Spacer(1, 0.4*cm))

    # -------------------------------------------------------------------
    # 3. Raw LLM response
    # -------------------------------------------------------------------
    story.append(Paragraph("3. Reponse brute du LLM (avant audit)", style_h1))
    raw_escaped = raw_response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(raw_escaped.replace("\n", "<br/>"), style_body))
    story.append(Spacer(1, 0.4*cm))

    # -------------------------------------------------------------------
    # 4. Automatic numerical audit
    # -------------------------------------------------------------------
    story.append(Paragraph("4. Audit numerique automatique", style_h1))
    audit_data = [
        ["Indicateur", "Valeur"],
        ["Nombres extraits de la reponse", str(len(verification_initial["all_numbers"]))],
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

    if verification_initial["invented_numbers"]:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Valeurs detectees comme non conformes :", style_h2))
        for val, raw in verification_initial["invented_numbers"]:
            story.append(Paragraph(f"- <b>{raw}</b> (interprete comme {val})", style_body))

    story.append(Spacer(1, 0.4*cm))

    # -------------------------------------------------------------------
    # 5. Corrected response (if hallucination was detected)
    # -------------------------------------------------------------------
    if verification_initial["has_hallucination"]:
        story.append(Paragraph("5. Reponse corrigee apres regeneration sous contrainte", style_h1))
        corr_escaped = corrected_response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(corr_escaped.replace("\n", "<br/>"), style_body))

        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Audit de la reponse corrigee :", style_h2))
        audit2 = [
            ["Nombres conformes apres correction", str(len(verification_final["valid_numbers"]))],
            ["Nombres inventes apres correction", str(len(verification_final["invented_numbers"]))],
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

    # -------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------
    footer_text = (
        "<i>Rapport genere automatiquement dans le cadre du memoire "
        "'BD-CeNN + LLM' - Nelson Ngombo.<br/>"
        f"Mode d'interference : <b>{model_label}</b>. "
        f"Modele LLM utilise : <b>{model_name if model_name else 'Non specifie'}</b>."
        "</i>"
    )
    story.append(Paragraph(footer_text, style_body))

    # -------------------------------------------------------------------
    # Build document
    # -------------------------------------------------------------------
    doc.build(story)

    if output_path is None:
        buffer.seek(0)
        return buffer.getvalue()
    return None


# ---------------------------------------------------------------------------
# Helper functions (mirror llm_assistant.py utilities)
# ---------------------------------------------------------------------------

def _fmt(value, ndigits=2):
    """Formats a numeric value for display."""
    import numpy as np
    try:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{value:.{ndigits}f}"
    except Exception:
        pass
    return str(value)


def _conflicts_label(model_label: str) -> str:
    """Libellé officiel du compteur principal selon le régime."""
    if model_label == "CCI+ACI":
        return "C_tot"
    return "C_CCI"


def _conflict_rows(model_label: str, m: dict) -> list:
    """Lignes de compteurs discrets, distinctes du coût J(x)."""
    rows = [["C_CCI", _fmt(m.get("conflicts_cci"))]]
    if model_label == "CCI+ACI":
        rows.append(["C_ACI", _fmt(m.get("conflicts_aci"))])
        rows.append(["C_tot", _fmt(m.get("conflicts"))])
    return rows


def _fmt_us(seconds) -> str:
    """Durée stockée en secondes, restituée en millisecondes (4 décimales)."""
    try:
        return f"{float(seconds) * 1e3:.4f}"
    except (TypeError, ValueError):
        return str(seconds)