"""
Tab 5: LLM Analysis Report with Independent Regex-based Audit.
"""

import streamlit as st
import numpy as np

from dashboard import session_manager as sm
from dashboard import toast
from dashboard.chart_factory import MODE_LABELS
from dashboard.components import (
    section_title,
    empty_state,
    info_banner,
    mode_selector,
    divider_with_label,
    render_error_box,
)


def _build_case_data(record, mode: str) -> dict:
    """Assembles the case_data payload for llm_assistant.build_prompt."""
    from graph_model import InterferenceGraph

    topology = record.topology
    bd_result = record.solver_results.get("BD-CeNN", {}).get(mode)
    if bd_result is None:
        return None

    baselines = {}
    for name in ["Random", "Greedy", "DSATUR"]:
        bres = record.solver_results.get(name, {}).get(mode)
        if bres:
            baselines[name] = {
                "cost": float(bres.cost),
                "conflicts": int(bres.n_conflicts),
                "time": float(bres.wall_time_seconds),
            }

    graph = InterferenceGraph(topology)
    if bd_result.initial_assignment is not None:
        cost_init = graph.compute_cost(bd_result.initial_assignment, mode)
    else:
        cost_init = 0.0

    conflict_edges = graph.get_conflict_edges_cci(bd_result.assignment)
    if mode == "cci_aci":
        conflict_edges += graph.get_conflict_edges_adjacent(bd_result.assignment)
    conflicting_cells = sorted(set(c for edge in conflict_edges for c in edge))

    allocations = {}
    for name, modes_dict in record.solver_results.items():
        if mode in modes_dict:
            allocations[name] = [int(c) for c in modes_dict[mode].assignment]

    cell_positions = {
        int(i): [float(topology.positions[i, 0]), float(topology.positions[i, 1])]
        for i in range(topology.N)
    }

    edges_all = []
    for i in range(topology.N):
        for j in range(i + 1, topology.N):
            if topology.W[i, j] > 0:
                edges_all.append((int(i), int(j), int(topology.W[i, j])))
    edges_all.sort(key=lambda e: -e[2])
    total_edges = len(edges_all)
    topology_edges = edges_all[:80]

    N = topology.N
    cell_degree = np.zeros(N, dtype=int)
    cell_strength = np.zeros(N, dtype=int)
    for (i, j, w) in edges_all:
        cell_degree[i] += 1
        cell_degree[j] += 1
        cell_strength[i] += w
        cell_strength[j] += w
    top_idx = np.argsort(-cell_strength)[:10]
    cell_summary = [
        {"cell": int(idx), "degree": int(cell_degree[idx]), "strength": int(cell_strength[idx])}
        for idx in top_idx
    ]

    topology_meta = {
        "seed": int(topology.seed),
        "threshold": float(topology.threshold),
        "N": int(topology.N),
        "K": int(topology.K),
    }

    model_label = "CCI-only" if mode == "cci" else "CCI+ACI"

    return {
        "case_id": f"DASHBOARD_{record.experiment_id}",
        "scenario": topology.scenario_name or "CUSTOM",
        "N": topology.N,
        "K": topology.K,
        "seed": topology.seed,
        "context": f"Analyse interactive depuis le dashboard [{model_label}]",
        "model_label": model_label,
        "metrics": {
            "cost_initial": float(cost_init),
            "cost_final": float(bd_result.cost),
            "conflicts": int(bd_result.n_conflicts),
            "time_seconds": float(bd_result.wall_time_seconds),
            "iterations": int(bd_result.best_iteration),
            "used_channels": int(bd_result.used_channels),
        },
        "baselines": baselines,
        "conflicting_cells": conflicting_cells,
        "allocations": allocations,
        "cell_positions": cell_positions,
        "topology_edges": topology_edges,
        "total_edges": total_edges,
        "cell_summary": cell_summary,
        "topology_meta": topology_meta,
    }


def _generate_llm_report(mode: str):
    """Invokes the LLM and audits the result."""
    record = sm.get("experiment_record")
    if record is None:
        toast.error("Aucun resultat disponible.")
        return

    case_data = _build_case_data(record, mode)
    if case_data is None:
        toast.error(f"Aucun resultat BD-CeNN pour le mode {MODE_LABELS[mode]}.")
        return

    sm.set_value(f"llm_case_data_{mode}", case_data)

    try:
        from llm_assistant import build_prompt, ask_llm
        prompt = build_prompt(case_data)

        with st.spinner(f"Interrogation du LLM pour {MODE_LABELS[mode]}..."):
            response = ask_llm(prompt)

        sm.set_value(f"llm_response_{mode}", response)

        verifier = sm.get("verifier")
        audit = verifier.audit(response, case_data)
        sm.set_value(f"audit_report_{mode}", audit)

        if audit.status == "CERTIFIED":
            toast.success(f"Rapport genere et certifie ({MODE_LABELS[mode]})")
        elif audit.status == "FAILED":
            toast.warning(f"Rapport genere mais audit echec ({audit.invented_count} valeurs suspectes)")
        else:
            toast.info(f"Rapport genere (audit WARNING : aucune valeur numerique)")

    except Exception as e:
        import traceback
        render_error_box(type(e).__name__, str(e), traceback.format_exc())


def _render_audit_badge(audit_report):
    """Renders a stylized certification badge."""
    if audit_report is None:
        info_banner("Aucun audit realise.", kind="info")
        return

    if audit_report.status == "CERTIFIED":
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
                        border-left: 4px solid #059669; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #065f46;">
                    RAPPORT CERTIFIE
                </div>
                <div style="color: #047857; font-size: 0.9rem; margin-top: 0.3rem;">
                    Toutes les {audit_report.total_numbers} valeurs numeriques citees
                    correspondent aux donnees certifiees.<br>
                    Taux d'exactitude : {audit_report.accuracy_rate:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif audit_report.status == "WARNING":
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                        border-left: 4px solid #d97706; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #92400e;">
                    AVERTISSEMENT
                </div>
                <div style="color: #b45309; font-size: 0.9rem; margin-top: 0.3rem;">
                    Aucune valeur numerique extraite de la reponse. Impossible de
                    certifier ou de rejeter le contenu.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                        border-left: 4px solid #dc2626; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #991b1b;">
                    RAPPORT NON CERTIFIE
                </div>
                <div style="color: #b91c1c; font-size: 0.9rem; margin-top: 0.3rem;">
                    {audit_report.invented_count} valeur(s) numerique(s) ne correspondent
                    pas aux donnees certifiees.<br>
                    Taux d'exactitude : {audit_report.accuracy_rate:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_verification_table(audit_report):
    """Displays the per-number verification table."""
    if audit_report is None or not audit_report.verifications:
        return

    import pandas as pd
    rows = []
    for v in audit_report.verifications:
        rows.append({
            "Extrait": v.extracted_raw,
            "Valeur": round(v.extracted_value, 4),
            "Statut": "OK" if v.matched else "REJETE",
            "Reference": round(v.matched_reference, 4) if v.matched else "-",
        })
    df = pd.DataFrame(rows)

    def _style_row(row):
        color = "#d4edda" if row["Statut"] == "OK" else "#f8d7da"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(df.style.apply(_style_row, axis=1),
                 use_container_width=True, height=300)


def render():
    """Main tab entrypoint."""
    st.header("Rapport LLM et audit independant")

    record = sm.get("experiment_record")
    if record is None:
        empty_state(
            icon_text="[?]",
            title="Aucun resultat a analyser",
            description=(
                "Lancez d'abord une simulation dans l'onglet 'Convergence' pour "
                "produire des metriques que le LLM pourra analyser."
            ),
        )
        return

    st.caption(
        "Le LLM produit une analyse qualitative en francais des resultats de "
        "simulation. Un auditeur regex independant verifie ensuite chaque valeur "
        "numerique citee dans la reponse contre les donnees certifiees."
    )

    # LLM availability check
    from llm_assistant import is_llm_available, get_active_model, get_initialization_error
    if not is_llm_available():
        info_banner(
            "Le LLM n'est pas initialise. Le rapport ne pourra pas etre genere. "
            f"Cause : {get_initialization_error() or 'inconnue'}",
            kind="warning",
        )
    else:
        st.caption(f"Modele LLM actif : `{get_active_model()}`")

    # Mode selector
    available_modes = list(record.solver_results.get("BD-CeNN", {}).keys())
    if not available_modes:
        info_banner("Aucun resultat BD-CeNN disponible.", kind="warning")
        return

    mode = mode_selector(
        session_key="llm_view_mode",
        available_modes=available_modes,
        label="Mode d'interference a analyser",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(
            "Generer l'analyse LLM",
            type="primary",
            use_container_width=True,
            disabled=(not is_llm_available()),
            key="btn_gen_llm",
        ):
            _generate_llm_report(mode)
            st.rerun()

    st.divider()

    response = sm.get(f"llm_response_{mode}")
    audit_report = sm.get(f"audit_report_{mode}")

    if response is None:
        empty_state(
            icon_text="[>]",
            title=f"Aucune analyse generee pour {MODE_LABELS[mode]}",
            description=(
                "Cliquez sur 'Generer l'analyse LLM' ci-dessus pour produire "
                "un rapport qualitatif en francais."
            ),
        )
        return

    # Two-column layout
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        section_title(f"Analyse LLM ({MODE_LABELS[mode]})")
        if response.startswith("[ERREUR LLM]") or response.startswith("[LLM_ERROR]"):
            st.error(response)
        else:
            st.markdown(response)

    with col_right:
        section_title("Certification de l'audit")
        _render_audit_badge(audit_report)

        if audit_report:
            with st.expander("Verification detaillee des nombres"):
                _render_verification_table(audit_report)

            with st.expander("Statistiques de l'audit"):
                st.markdown(f"**Nombres extraits** : {audit_report.total_numbers}")
                st.markdown(f"**Nombres conformes** : {audit_report.valid_count}")
                st.markdown(f"**Divergences** : {audit_report.invented_count}")
                st.markdown(f"**Tolerance** : {audit_report.tolerance * 100:.1f}%")
                st.markdown(f"**Horodatage** : {audit_report.timestamp}")