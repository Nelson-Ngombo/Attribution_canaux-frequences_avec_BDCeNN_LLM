"""
Tab: Export experiment results.

Provides download options for:
    - CSV metrics (combined, CCI-only, CCI+ACI)
    - JSON full record
    - LLM audit PDF reports (one per interference mode, format identical to E10)
    - LLM raw text reports
    - Audit JSON
    - Complete ZIP bundle

Strict separation of CCI and CCI+ACI outputs.
"""

import json
import io
import zipfile
import numpy as np

import streamlit as st

from dashboard import session_manager as sm
from dashboard import toast
from dashboard.chart_factory import (
    build_metrics_dataframe,
    create_convergence_plot,
    create_comparison_bar_chart,
    create_delta_aci_chart,
    MODE_LABELS,
)
from dashboard.components import (
    section_title,
    empty_state,
    info_banner,
    render_scenario_summary,
)
from dashboard.pdf_export import generate_llm_pdf
from export_manager import (
    export_record_json,
    export_metrics_csv,
    export_metrics_split_csv,
    export_convergence_csv,
    export_figure_html,
    build_zip_bundle,
    get_output_dirs,
)


# ---------------------------------------------------------------------------
# Case data reconstruction (mirrors tab_llm_report.py)
# ---------------------------------------------------------------------------

def _build_case_data(record, mode: str) -> dict:
    """
    Rebuilds the case_data payload used originally to query the LLM.
    Required for regenerating the PDF with identical structure to E10.
    """
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
        {"cell": int(idx), "degree": int(cell_degree[idx]),
         "strength": int(cell_strength[idx])}
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


def _generate_llm_pdf_bytes(record, mode: str):
    """
    Generates the LLM PDF for a given mode.
    Returns bytes if successful, None otherwise.
    """
    response = sm.get(f"llm_response_{mode}")
    audit = sm.get(f"audit_report_{mode}")

    if not response or response.startswith("[ERREUR"):
        return None

    case_data = _build_case_data(record, mode)
    if case_data is None:
        return None

    # Rebuild prompt exactly as it was sent
    try:
        from llm_assistant import build_prompt, get_active_model
        prompt = build_prompt(case_data)
        model_name = get_active_model().replace("models/", "") if get_active_model() else None
    except Exception:
        prompt = "(prompt indisponible)"
        model_name = None

    # Build verification dict from AuditReport
    verification_initial = _audit_to_verification_dict(audit)
    verification_final = verification_initial  # No re-correction here
    corrected_response = response

    try:
        pdf_bytes = generate_llm_pdf(
            case_data=case_data,
            prompt=prompt,
            raw_response=response,
            verification_initial=verification_initial,
            corrected_response=corrected_response,
            verification_final=verification_final,
            output_path=None,
            model_name=model_name,
        )
        return pdf_bytes
    except Exception as e:
        toast.error(f"Erreur generation PDF {MODE_LABELS[mode]}: {e}")
        return None


def _audit_to_verification_dict(audit) -> dict:
    """Converts an AuditReport dataclass into the dict format expected by generate_llm_pdf."""
    if audit is None:
        return {
            "all_numbers": [],
            "allowed_numbers": set(),
            "valid_numbers": [],
            "invented_numbers": [],
            "has_hallucination": False,
            "accuracy_rate": 100.0,
        }

    valid = [(v.extracted_value, v.extracted_raw) for v in audit.verifications if v.matched]
    invented = [(v.extracted_value, v.extracted_raw) for v in audit.verifications if not v.matched]
    all_numbers = [(v.extracted_value, v.extracted_raw) for v in audit.verifications]

    return {
        "all_numbers": all_numbers,
        "allowed_numbers": set(),
        "valid_numbers": valid,
        "invented_numbers": invented,
        "has_hallucination": audit.invented_count > 0,
        "accuracy_rate": audit.accuracy_rate,
    }


# ---------------------------------------------------------------------------
# Figures collection (for ZIP)
# ---------------------------------------------------------------------------

def _figures_for_bundle(record):
    """Collects Plotly figures to embed in the ZIP bundle as HTML."""
    figures = []
    try:
        figures.append(("cost_comparison",
                        create_comparison_bar_chart(record, metric="cost")))
        figures.append(("conflicts_comparison",
                        create_comparison_bar_chart(record, metric="n_conflicts")))
        figures.append(("runtime_comparison",
                        create_comparison_bar_chart(record, metric="wall_time_seconds")))

        delta_fig = create_delta_aci_chart(record)
        if delta_fig is not None:
            figures.append(("delta_aci", delta_fig))

        bd_results = record.solver_results.get("BD-CeNN", {})
        for mode, result in bd_results.items():
            figures.append((f"convergence_{mode}",
                            create_convergence_plot(result, mode)))
    except Exception as e:
        toast.warning(f"Certaines figures n'ont pas pu etre generees: {e}")

    return figures


# ---------------------------------------------------------------------------
# MAIN RENDERER
# ---------------------------------------------------------------------------

def render():
    """Main tab entrypoint."""
    st.header("Exportation des resultats")

    record = sm.get("experiment_record")
    if record is None:
        empty_state(
            icon_text="[+]",
            title="Aucun resultat a exporter",
            description=(
                "Lancez d'abord une simulation dans l'onglet 'Convergence' "
                "pour produire des resultats exportables."
            ),
        )
        return

    st.caption(
        "Telechargez les resultats dans plusieurs formats. Les fichiers sont "
        "egalement sauvegardes automatiquement dans "
        "`outputs/interactive_runs/{cochannel|adjacent}/` a la fin de chaque simulation."
    )

    topology = sm.get("topology")
    if topology is not None:
        render_scenario_summary(topology)

    st.divider()

    # ------------------------------------------------------------------
    # Preview table (kept for context)
    # ------------------------------------------------------------------
    section_title("Apercu du tableau des metriques")
    try:
        df = build_metrics_dataframe(record)
        st.dataframe(df, use_container_width=True, height=300)
    except Exception as e:
        toast.error(f"Erreur d'affichage du tableau: {e}")
        return

    st.divider()

    # ------------------------------------------------------------------
    # Individual downloads
    # ------------------------------------------------------------------
    section_title("Telechargements individuels")

    st.caption(
        "Formats disponibles : CSV des metriques (combine ou par mode), "
        "JSON complet du run, rapports LLM (texte et PDF audit format E10), audit JSON."
    )

    # Row 1: JSON, CSV combined, CSV CCI-only
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        try:
            json_bytes = json.dumps(record.to_dict(), indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                label="JSON complet",
                data=json_bytes,
                file_name=f"experiment_{record.experiment_id}.json",
                mime="application/json",
                use_container_width=True,
                key="dl_json_full",
            )
        except Exception as e:
            st.button("JSON complet", disabled=True, use_container_width=True)
            st.caption(f"Erreur: {e}")

    with dl_col2:
        try:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="CSV metriques (combine)",
                data=csv_bytes,
                file_name=f"metrics_{record.experiment_id}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv_combined",
            )
        except Exception:
            st.button("CSV metriques (combine)", disabled=True, use_container_width=True)

    with dl_col3:
        cci_df = df[df["Mode"] == "CCI-only"]
        if not cci_df.empty:
            st.download_button(
                label="CSV metriques (CCI-only)",
                data=cci_df.to_csv(index=False).encode("utf-8"),
                file_name=f"metrics_{record.experiment_id}_cci.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv_cci",
            )
        else:
            st.button("CSV metriques (CCI-only)", disabled=True, use_container_width=True,
                      help="Aucun resultat en mode CCI-only")

    # Row 2: CSV CCI+ACI, LLM TXT, Audit JSON
    dl_col4, dl_col5, dl_col6 = st.columns(3)

    with dl_col4:
        aci_df = df[df["Mode"] == "CCI+ACI"]
        if not aci_df.empty:
            st.download_button(
                label="CSV metriques (CCI+ACI)",
                data=aci_df.to_csv(index=False).encode("utf-8"),
                file_name=f"metrics_{record.experiment_id}_cci_aci.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv_aci",
            )
        else:
            st.button("CSV metriques (CCI+ACI)", disabled=True, use_container_width=True,
                      help="Aucun resultat en mode CCI+ACI")

    with dl_col5:
        llm_cci = sm.get("llm_response_cci")
        llm_aci = sm.get("llm_response_cci_aci")
        combined_llm = ""
        if llm_cci:
            combined_llm += f"=== CCI-only ===\n\n{llm_cci}\n\n"
        if llm_aci:
            combined_llm += f"=== CCI+ACI ===\n\n{llm_aci}\n\n"
        if combined_llm:
            st.download_button(
                label="Rapport LLM (TXT)",
                data=combined_llm.encode("utf-8"),
                file_name=f"llm_report_{record.experiment_id}.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_llm_txt",
            )
        else:
            st.button("Rapport LLM (TXT)", disabled=True, use_container_width=True,
                      help="Generez d'abord un rapport LLM dans l'onglet Rapport LLM")

    with dl_col6:
        audit_cci = sm.get("audit_report_cci")
        audit_aci = sm.get("audit_report_cci_aci")
        combined_audit = {}
        if audit_cci:
            combined_audit["cci"] = audit_cci.to_dict()
        if audit_aci:
            combined_audit["cci_aci"] = audit_aci.to_dict()
        if combined_audit:
            audit_bytes = json.dumps(combined_audit, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                label="Audit (JSON)",
                data=audit_bytes,
                file_name=f"audit_{record.experiment_id}.json",
                mime="application/json",
                use_container_width=True,
                key="dl_audit_json",
            )
        else:
            st.button("Audit (JSON)", disabled=True, use_container_width=True,
                      help="Generez d'abord un rapport LLM avec audit")

    # Row 3: LLM PDF audits (one per mode, format E10)
    st.markdown("")
    st.caption(
        "**Rapports PDF LLM** (format identique a l'experience E10) : "
        "prompt envoye, reponse brute, audit numerique automatique, "
        "reponse corrigee si hallucination detectee, badge de certification."
    )

    dl_col7, dl_col8 = st.columns(2)

    with dl_col7:
        llm_cci = sm.get("llm_response_cci")
        if llm_cci and not llm_cci.startswith("[ERREUR"):
            if st.button(
                "Generer rapport PDF (CCI-only)",
                use_container_width=True,
                key="btn_gen_pdf_cci",
            ):
                with st.spinner("Generation du PDF CCI-only..."):
                    pdf_bytes = _generate_llm_pdf_bytes(record, "cci")
                if pdf_bytes is not None:
                    st.session_state["_pdf_cci_bytes"] = pdf_bytes
                    toast.success("PDF CCI-only pret")
                else:
                    toast.error("Echec de la generation")

            pdf_cci_bytes = st.session_state.get("_pdf_cci_bytes")
            if pdf_cci_bytes:
                st.download_button(
                    label="Telecharger PDF (CCI-only)",
                    data=pdf_cci_bytes,
                    file_name=f"rapport_llm_{record.experiment_id}_cci.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_pdf_cci",
                )
        else:
            st.button("Generer rapport PDF (CCI-only)", disabled=True, use_container_width=True,
                      help="Aucun rapport LLM CCI-only disponible")

    with dl_col8:
        llm_aci = sm.get("llm_response_cci_aci")
        if llm_aci and not llm_aci.startswith("[ERREUR"):
            if st.button(
                "Generer rapport PDF (CCI+ACI)",
                use_container_width=True,
                key="btn_gen_pdf_aci",
            ):
                with st.spinner("Generation du PDF CCI+ACI..."):
                    pdf_bytes = _generate_llm_pdf_bytes(record, "cci_aci")
                if pdf_bytes is not None:
                    st.session_state["_pdf_aci_bytes"] = pdf_bytes
                    toast.success("PDF CCI+ACI pret")
                else:
                    toast.error("Echec de la generation")

            pdf_aci_bytes = st.session_state.get("_pdf_aci_bytes")
            if pdf_aci_bytes:
                st.download_button(
                    label="Telecharger PDF (CCI+ACI)",
                    data=pdf_aci_bytes,
                    file_name=f"rapport_llm_{record.experiment_id}_cci_aci.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_pdf_aci",
                )
        else:
            st.button("Generer rapport PDF (CCI+ACI)", disabled=True, use_container_width=True,
                      help="Aucun rapport LLM CCI+ACI disponible")

    st.divider()

    # ------------------------------------------------------------------
    # Complete ZIP bundle
    # ------------------------------------------------------------------
    section_title("Archive complete (ZIP)")

    st.caption(
        "Regroupe tous les artefacts dans une archive unique : JSON, CSVs par mode, "
        "trajectoires de convergence, rapports LLM (TXT), audit JSON, figures HTML "
        "interactives, et PDF LLM au format E10 pour chaque mode."
    )

    include_llm = st.checkbox(
        "Inclure les rapports LLM (TXT + PDF)",
        value=sm.get("export_include_llm", True),
        key="chk_bundle_llm",
    )
    include_figs = st.checkbox(
        "Inclure les figures HTML interactives",
        value=sm.get("export_include_figures", True),
        key="chk_bundle_figs",
    )
    sm.set_value("export_include_llm", include_llm)
    sm.set_value("export_include_figures", include_figs)

    if st.button("Preparer l'archive ZIP", type="primary", key="btn_prepare_zip"):
        with st.spinner("Construction de l'archive..."):
            try:
                figures = _figures_for_bundle(record) if include_figs else None
                zip_bytes = build_zip_bundle(
                    record,
                    figures=figures,
                    include_llm_report=include_llm,
                )

                # Append LLM PDFs to ZIP
                if include_llm:
                    buffer = io.BytesIO(zip_bytes)
                    with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                        for mode_key in ["cci", "cci_aci"]:
                            pdf_bytes = _generate_llm_pdf_bytes(record, mode_key)
                            if pdf_bytes is not None:
                                fname = f"rapport_llm_{mode_key}.pdf"
                                zf.writestr(fname, pdf_bytes)
                    buffer.seek(0)
                    zip_bytes = buffer.getvalue()

                st.download_button(
                    label="Telecharger l'archive ZIP",
                    data=zip_bytes,
                    file_name=f"experiment_{record.experiment_id}_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_zip_final",
                )
                toast.success("Archive prete pour telechargement")
            except Exception as e:
                import traceback
                toast.error(f"Erreur de construction: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    st.divider()

    # ------------------------------------------------------------------
    # Save to disk
    # ------------------------------------------------------------------
    section_title("Sauvegarde manuelle vers le disque")

    st.caption(
        "Persiste tous les artefacts dans `results/dashboard_runs/` avec "
        "separation stricte cochannel / adjacent. Cette action est distincte "
        "de la sauvegarde automatique qui se produit apres chaque simulation."
    )

    if st.button("Sauvegarder tout sur disque", key="btn_save_disk"):
        try:
            saved_paths = []

            json_path = export_record_json(record)
            saved_paths.append(str(json_path))

            csv_paths = export_metrics_split_csv(record)
            for p in csv_paths.values():
                saved_paths.append(str(p))

            conv_path = export_convergence_csv(record)
            if conv_path:
                saved_paths.append(str(conv_path))

            for mode in ["cci", "cci_aci"]:
                response = sm.get(f"llm_response_{mode}")
                if response:
                    dirs = get_output_dirs(mode)
                    llm_path = dirs["reports"] / f"llm_report_{record.experiment_id}.txt"
                    llm_path.write_text(response, encoding="utf-8")
                    saved_paths.append(str(llm_path))

                audit = sm.get(f"audit_report_{mode}")
                if audit:
                    dirs = get_output_dirs(mode)
                    audit_path = dirs["reports"] / f"audit_{record.experiment_id}.json"
                    audit_path.write_text(
                        json.dumps(audit.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    saved_paths.append(str(audit_path))

                # Save PDF audit for this mode
                pdf_bytes = _generate_llm_pdf_bytes(record, mode)
                if pdf_bytes is not None:
                    dirs = get_output_dirs(mode)
                    pdf_path = dirs["reports"] / f"rapport_llm_{record.experiment_id}_{mode}.pdf"
                    pdf_path.write_bytes(pdf_bytes)
                    saved_paths.append(str(pdf_path))

            figures = _figures_for_bundle(record)
            for fname, fig in figures:
                mode_key = "cci_aci" if "cci_aci" in fname else "cci"
                dirs = get_output_dirs(mode_key)
                fig_path = dirs["figures"] / f"{fname}_{record.experiment_id}.html"
                export_figure_html(fig, fig_path)
                saved_paths.append(str(fig_path))

            toast.success(f"{len(saved_paths)} fichiers sauvegardes sur disque")

            with st.expander("Chemins des fichiers sauvegardes"):
                for p in saved_paths:
                    st.code(p)

        except Exception as e:
            import traceback
            toast.error(f"Echec de la sauvegarde: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())