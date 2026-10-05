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


def _Figures_for_bundle(record):
    figures = []
    try:
        figures.append(("cost_comparison", create_comparison_bar_chart(record, metric="cost")))
        figures.append(("conflicts_cci_comparison", create_comparison_bar_chart(record, metric="conflicts_cci")))
        figures.append(("conflicts_aci_comparison", create_comparison_bar_chart(record, metric="conflicts_aci")))
        figures.append(("conflicts_total_comparison", create_comparison_bar_chart(record, metric="conflicts_total")))
        figures.append(("runtime_comparison", create_comparison_bar_chart(record, metric="wall_time_seconds")))

        delta_fig = create_delta_aci_chart(record)
        if delta_fig is not None:
            figures.append(("delta_aci", delta_fig))

        bd_results = record.solver_results.get("BD-CeNN", {})
        for mode, result in bd_results.items():
            figures.append((f"convergence_{mode}", create_convergence_plot(result, mode)))
    except Exception as e:
        toast.warning(f"Echec de preparation de certaines figures : {e}")

    return figures


def _generate_llm_pdf_bytes(record, mode: str):
    response = sm.get(f"llm_response_{mode}")
    audit = sm.get(f"audit_report_{mode}")

    if not response or response.startswith("[ERREUR"):
        return None

    case_data = _build_case_data(record, mode)
    if case_data is None:
        return None

    try:
        from llm_assistant import build_prompt, get_active_model
        prompt = build_prompt(case_data)
        model_name = get_active_model().replace("models/", "") if get_active_model() else None
    except Exception:
        prompt = "(prompt indisponible)"
        model_name = None

    # Extraction du rapport d'audit au format dictionnaire
    verification_initial = _audit_to_verification_dict(audit)

    try:
        pdf_bytes = generate_llm_pdf(
            case_data=case_data,
            prompt=prompt,
            raw_response=response,
            verification_initial=verification_initial,
            corrected_response=response,
            verification_final=verification_initial,
            output_path=None,
            model_name=model_name,
        )
        return pdf_bytes
    except Exception as e:
        toast.error(f"Echec de gènèration du PDF ({MODE_LABELS[mode]}) : {e}")
        return None


def _audit_to_verification_dict(audit) -> dict:
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


def _build_case_data(record, mode: str) -> dict:
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
                "conflicts_cci": int(bres.n_conflicts_cci),
                "conflicts_aci": int(bres.n_conflicts_aci),
                "conflicts_total": int(bres.n_conflicts_total),
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
            "conflicts": int(bd_result.n_conflicts_total),
            "conflicts_cci": int(bd_result.n_conflicts_cci),
            "conflicts_aci": int(bd_result.n_conflicts_aci),
            "time_seconds": float(bd_result.wall_time_seconds),
            "iterations": int(bd_result.n_sweeps),
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


def render():
    """Point d'entree de l'onglet."""
    st.header("Exportation des resultats")

    record = sm.get("experiment_record")
    if record is None:
        empty_state(
            icon_text="[+]",
            title="Aucun resultat a exporter",
            description="Lancez d'abord la simulation dans l'onglet 'Convergence'.",
        )
        return

    st.caption(
        "Ce module centralise les fonctions de téléchargement de l'experience courante. "
        "Les formats respectent scrupuleusement le decoupage CCI et ACI."
    )

    topology = sm.get("topology")
    if topology is not None:
        render_scenario_summary(topology)

    st.divider()

    # 1. Aperçu
    section_title("Tableau recapitulatif de controle")
    try:
        df = build_metrics_dataframe(record)
        st.dataframe(df, use_container_width=True, height=300)
    except Exception as e:
        toast.error(f"Echec d'affichage du tableau : {e}")
        return

    st.divider()

    # 2. Téléchargements individuels
    section_title("Telechargements individuels")

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
            st.button("CSV metriques (CCI-only)", disabled=True, use_container_width=True)

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
            st.button("CSV metriques (CCI+ACI)", disabled=True, use_container_width=True)

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
            st.button("Rapport LLM (TXT)", disabled=True, use_container_width=True)

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
            st.button("Audit (JSON)", disabled=True, use_container_width=True)

    # 3. Rapports PDF LLM d'Audit (Style E10, Chantier B+C)
    st.markdown("")
    st.caption(
        "**Rapports d'Audit PDF formatte (Style E10)** : intègre les questions du prompt, "
        "la rēponse brute, le badge d'audit du vèrifieur et les corrections le cas echeant."
    )

    dl_col7, dl_col8 = st.columns(2)

    with dl_col7:
        llm_cci = sm.get("llm_response_cci")
        if llm_cci and not llm_cci.startswith("[ERREUR"):
            if st.button("Generer PDF d'Audit (CCI-only)", use_container_width=True, key="btn_gen_pdf_cci"):
                with st.spinner("Generation en cours..."):
                    pdf_bytes = _generate_llm_pdf_bytes(record, "cci")
                if pdf_bytes is not None:
                    st.session_state["_pdf_cci_bytes"] = pdf_bytes
                    toast.success("Rapport PDF CCI-only genere.")
                else:
                    toast.error("Echec de la gènèration.")

            pdf_cci_bytes = st.session_state.get("_pdf_cci_bytes")
            if pdf_cci_bytes:
                st.download_button(
                    label="Telecharger l'Audit PDF (CCI-only)",
                    data=pdf_cci_bytes,
                    file_name=f"rapport_audit_llm_{record.experiment_id}_cci.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_pdf_cci",
                )
        else:
            st.button("Generer PDF d'Audit (CCI-only)", disabled=True, use_container_width=True)

    with dl_col8:
        llm_aci = sm.get("llm_response_cci_aci")
        if llm_aci and not llm_aci.startswith("[ERREUR"):
            if st.button("Generer PDF d'Audit (CCI+ACI)", use_container_width=True, key="btn_gen_pdf_aci"):
                with st.spinner("Generation en cours..."):
                    pdf_bytes = _generate_llm_pdf_bytes(record, "cci_aci")
                if pdf_bytes is not None:
                    st.session_state["_pdf_aci_bytes"] = pdf_bytes
                    toast.success("Rapport PDF CCI+ACI genere.")
                else:
                    toast.error("Echec de la gènèration.")

            pdf_aci_bytes = st.session_state.get("_pdf_aci_bytes")
            if pdf_aci_bytes:
                st.download_button(
                    label="Telecharger l'Audit PDF (CCI+ACI)",
                    data=pdf_aci_bytes,
                    file_name=f"rapport_audit_llm_{record.experiment_id}_cci_aci.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_pdf_aci",
                )
        else:
            st.button("Generer PDF d'Audit (CCI+ACI)", disabled=True, use_container_width=True)

    st.divider()

    # 4. ZIP Bundle complet
    section_title("Archive ZIP de synthese")

    st.caption(
        "Assemble tous les livrables du run au sein d'une seule archive compresse."
    )

    include_llm = st.checkbox(
        "Inclure les rapports d'Audit et texte du LLM",
        value=sm.get("export_include_llm", True),
        key="chk_bundle_llm",
    )
    include_figs = st.checkbox(
        "Inclure les graphes de synthese HTML interactifs",
        value=sm.get("export_include_figures", True),
        key="chk_bundle_figs",
    )
    sm.set_value("export_include_llm", include_llm)
    sm.set_value("export_include_figures", include_figs)

    if st.button("Compiler l'archive ZIP", type="primary", key="btn_prepare_zip"):
        with st.spinner("Compression des donnees..."):
            try:
                figures = _Figures_for_bundle(record) if include_figs else None
                zip_bytes = build_zip_bundle(
                    record,
                    figures=figures,
                    include_llm_report=include_llm,
                )

                if include_llm:
                    buffer = io.BytesIO(zip_bytes)
                    with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                        for mode_key in ["cci", "cci_aci"]:
                            pdf_bytes = _generate_llm_pdf_bytes(record, mode_key)
                            if pdf_bytes is not None:
                                zf.writestr(f"rapport_audit_llm_{mode_key}.pdf", pdf_bytes)
                    buffer.seek(0)
                    zip_bytes = buffer.getvalue()

                st.download_button(
                    label="Telecharger le ZIP de synthese",
                    data=zip_bytes,
                    file_name=f"run_bundle_{record.experiment_id}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_zip_final",
                )
                toast.success("L'archive ZIP est prete.")
            except Exception as e:
                import traceback
                toast.error(f"Echec de compilation ZIP : {e}")
                with st.expander("Details techniques"):
                    st.code(traceback.format_exc())

    st.divider()

    # 5. Sauvegarde disque manuelle
    section_title("Ecrire la session sur le stockage physique")

    st.caption(
        "Exporte et classe la session dans l'arborescence `results/dashboard_runs/`."
    )

    if st.button("Sauvegarder la session", key="btn_save_disk"):
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

                pdf_bytes = _generate_llm_pdf_bytes(record, mode)
                if pdf_bytes is not None:
                    dirs = get_output_dirs(mode)
                    pdf_path = dirs["reports"] / f"rapport_audit_llm_{record.experiment_id}_{mode}.pdf"
                    pdf_path.write_bytes(pdf_bytes)
                    saved_paths.append(str(pdf_path))

            figures = _Figures_for_bundle(record)
            for fname, fig in figures:
                mode_key = "cci_aci" if "cci_aci" in fname else "cci"
                dirs = get_output_dirs(mode_key)
                fig_path = dirs["figures"] / f"{fname}_{record.experiment_id}.html"
                export_figure_html(fig, fig_path)
                saved_paths.append(str(fig_path))

            toast.success(f"Session archivee : {len(saved_paths)} fichiers ecrits.")

            with st.expander("Consulter les chemins de sauvegarde"):
                for p in saved_paths:
                    st.code(p)

        except Exception as e:
            import traceback
            toast.error(f"Echec d'ecriture sur le disque : {e}")
            with st.expander("Details techniques"):
                st.code(traceback.format_exc())