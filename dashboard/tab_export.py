"""
Tab 7: Export experiment results.
Provides download buttons for CSV, JSON, HTML figures, and a complete ZIP bundle
with strict separation of CCI and CCI+ACI outputs.
"""

import json
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
    divider_with_label,
)
from export_manager import (
    export_record_json,
    export_metrics_csv,
    export_metrics_split_csv,
    export_convergence_csv,
    export_figure_html,
    build_zip_bundle,
    get_output_dirs,
)


def _figures_for_bundle(record):
    """Collects Plotly figures to embed into the ZIP bundle."""
    figures = []
    try:
        figures.append(("cost_comparison", create_comparison_bar_chart(record, metric="cost")))
        figures.append(("conflicts_comparison", create_comparison_bar_chart(record, metric="n_conflicts")))
        figures.append(("runtime_comparison", create_comparison_bar_chart(record, metric="wall_time_seconds")))

        delta_fig = create_delta_aci_chart(record)
        if delta_fig is not None:
            figures.append(("delta_aci", delta_fig))

        bd_results = record.solver_results.get("BD-CeNN", {})
        for mode, result in bd_results.items():
            figures.append((f"convergence_{mode}", create_convergence_plot(result, mode)))
    except Exception as e:
        toast.warning(f"Certaines figures n'ont pas pu etre generees: {e}")

    return figures


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
        "egalement sauvegardes automatiquement sur disque dans "
        "`outputs/interactive_runs/{cochannel|adjacent}/` a la fin de chaque simulation."
    )

    topology = sm.get("topology")
    if topology is not None:
        render_scenario_summary(topology)

    st.divider()

    # ------------------------------------------------------------------
    # Summary of available data
    # ------------------------------------------------------------------
    section_title("Resume de l'experience")

    from dashboard.components import kpi_row
    kpi_row([
        {"label": "ID experience", "value": record.experiment_id},
        {"label": "Solveurs", "value": str(len(record.solver_results))},
        {"label": "Modes evalues", "value": str(len(record.solver_results.get("BD-CeNN", {})))},
        {"label": "Delta_ACI",
         "value": (f"+{record.delta_aci_percent:.2f}%"
                   if record.delta_aci_percent is not None else "-")},
    ])

    st.divider()

    # ------------------------------------------------------------------
    # Preview
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
    # Individual download buttons
    # ------------------------------------------------------------------
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
            st.caption(f"Erreur: {e}")

    with dl_col2:
        try:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Metriques CSV (combine)",
                data=csv_bytes,
                file_name=f"metrics_{record.experiment_id}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv_combined",
            )
        except Exception as e:
            st.button("Metriques CSV (combine)", disabled=True, use_container_width=True)

    with dl_col3:
        cci_df = df[df["Mode"] == "CCI-only"]
        if not cci_df.empty:
            st.download_button(
                label="Metriques CSV (CCI-only)",
                data=cci_df.to_csv(index=False).encode("utf-8"),
                file_name=f"metrics_{record.experiment_id}_cci.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv_cci",
            )
        else:
            st.button("Metriques CSV (CCI-only)", disabled=True, use_container_width=True,
                      help="Aucun resultat en mode CCI-only")

    dl_col4, dl_col5, dl_col6 = st.columns(3)

    with dl_col4:
        aci_df = df[df["Mode"] == "CCI+ACI"]
        if not aci_df.empty:
            st.download_button(
                label="Metriques CSV (CCI+ACI)",
                data=aci_df.to_csv(index=False).encode("utf-8"),
                file_name=f"metrics_{record.experiment_id}_cci_aci.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv_aci",
            )
        else:
            st.button("Metriques CSV (CCI+ACI)", disabled=True, use_container_width=True,
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

    st.divider()

    # ------------------------------------------------------------------
    # Complete ZIP bundle
    # ------------------------------------------------------------------
    section_title("Archive complete (ZIP)")

    st.caption(
        "Regroupe tous les artefacts (JSON, CSVs par mode, trajectoires de "
        "convergence, rapports LLM, audit, figures HTML interactives) dans "
        "une archive unique."
    )

    include_llm = st.checkbox(
        "Inclure les rapports LLM",
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
        "de la sauvegarde automatique qui se produit apres chaque simulation "
        "(qui utilise `outputs/interactive_runs/`)."
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