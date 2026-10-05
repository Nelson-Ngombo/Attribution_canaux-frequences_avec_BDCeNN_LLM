"""
Tab 4: Global Solver Comparison with Grouped Bar Charts.
"""

import streamlit as st

from dashboard import session_manager as sm
from dashboard.chart_factory import (
    build_metrics_dataframe,
    create_comparison_bar_chart,
    create_delta_aci_chart,
    create_radar_chart,
    MODE_LABELS,
)
from dashboard.components import (
    section_title,
    empty_state,
    info_banner,
)


def render():
    """Point d'entree de l'onglet."""
    st.header("Comparaison globale : BD-CeNN vs Heuristiques")

    record = sm.get("experiment_record")
    if record is None:
        empty_state(
            icon_text="[+]",
            title="Aucun run disponible",
            description="Exécutez le solveur dans l'onglet 'Convergence' pour comparer les performances.",
        )
        return

    st.caption(
        "Ce module compare le solveur BD-CeNN aux trois heuristiques classiques (Random, Greedy, DSATUR). "
        "Le comportement est segmente pour analyser precisement l'impact des contraintes spectrales."
    )

    # 1. Tableau récapitulatif
    section_title("Tableau recapitulatif de performance")
    df = build_metrics_dataframe(record)
    st.dataframe(df, use_container_width=True, height=350)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Exporter la table de comparaison (CSV)",
        data=csv_bytes,
        file_name=f"metrics_comparison_{record.experiment_id}.csv",
        mime="text/csv",
    )

    st.divider()

    # 2. Histogrammes comparatifs
    section_title("Graphiques de mètrics comparees")

    metric = st.selectbox(
        "Metrique d'analyse",
        options=["cost", "conflicts_cci", "conflicts_aci", "conflicts_total", "used_channels", "wall_time_seconds", "n_sweeps"],
        format_func=lambda m: {
            "cost": "Cout global J(x)",
            "conflicts_cci": "Conflits co-canal C_CCI",
            "conflicts_aci": "Conflits adjacents C_ACI",
            "conflicts_total": "Conflits totaux C_total",
            "used_channels": "Canaux distincts utilises",
            "wall_time_seconds": "Temps d'execution t_exec (s)",
            "n_sweeps": "Nombre de balayages (sweeps)",
        }[m],
        key="cmp_metric_select",
    )
    fig = create_comparison_bar_chart(record, metric=metric)
    st.plotly_chart(fig, use_container_width=True, key=f"cmp_{metric}_{record.experiment_id}")

    st.divider()

    # 3. Graphique Delta_ACI
    section_title("Analyse de la degradation adjacente (Delta_ACI)")
    fig_delta = create_delta_aci_chart(record)
    if fig_delta is not None:
        st.plotly_chart(fig_delta, use_container_width=True, key=f"delta_{record.experiment_id}")
        info_banner(
            "Le Delta_ACI mesure l'impact d'une evaluation sous contrainte adjacente sur le "
            "cout obtenu sous contrainte co-canal simple.",
            kind="info",
        )
    else:
        info_banner(
            "Pour calculer le Delta_ACI, vous devez cocher les deux modes "
            "d'interference lors de la configuration.",
            kind="warning",
        )

    st.divider()

    # 4. Profils radar
    section_title("Profils comportementaux (Radar normalise)")

    available_modes = list(record.metrics.get("BD-CeNN", {}).keys())
    if not available_modes:
        info_banner("Aucune mètric n'est disponible pour tracer le radar.", kind="warning")
        return

    radar_tabs = st.tabs([MODE_LABELS[m] for m in available_modes])
    for tab, mode in zip(radar_tabs, available_modes):
        with tab:
            fig_radar = create_radar_chart(record, mode)
            if fig_radar is not None:
                st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{mode}_{record.experiment_id}")
            else:
                info_banner(f"Donnees insuffisantes pour tracer le profil {MODE_LABELS[mode]}.", kind="warning")