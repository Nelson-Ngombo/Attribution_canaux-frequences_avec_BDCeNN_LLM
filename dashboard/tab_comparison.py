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
    divider_with_label,
)


def render():
    """Main tab entrypoint."""
    st.header("Comparaison globale : BD-CeNN vs Baselines")

    record = sm.get("experiment_record")
    if record is None:
        empty_state(
            icon_text="[+]",
            title="Aucun resultat disponible",
            description=(
                "Lancez d'abord une simulation dans l'onglet 'Convergence' "
                "pour voir la comparaison entre BD-CeNN et les baselines."
            ),
        )
        return

    st.caption(
        "Evaluation comparative des quatre solveurs sur les deux modeles "
        "d'interference. Chaque metrique est rapportee separement pour CCI-only "
        "et CCI+ACI afin de mettre en evidence l'impact du modele physique "
        "adjacent."
    )

    # Metrics table
    section_title("Tableau recapitulatif")
    df = build_metrics_dataframe(record)
    st.dataframe(df, use_container_width=True, height=350)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Telecharger le CSV recapitulatif",
        data=csv_bytes,
        file_name=f"metrics_{record.experiment_id}.csv",
        mime="text/csv",
    )

    st.divider()

    # Bar charts
    section_title("Comparaison visuelle par metrique")

    metric = st.selectbox(
        "Metrique a comparer",
        options=["cost", "n_conflicts", "used_channels", "wall_time_seconds"],
        format_func=lambda m: {
            "cost": "Cout J(x)",
            "n_conflicts": "Nombre de conflits",
            "used_channels": "Canaux utilises",
            "wall_time_seconds": "Temps d'execution (s)",
        }[m],
        key="cmp_metric_select",
    )
    fig = create_comparison_bar_chart(record, metric=metric)
    fig.update_layout(transition_duration=500)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"cmp_{metric}_{record.experiment_id}",
    )

    st.divider()

    # Delta ACI
    section_title("Delta_ACI : impact du modele adjacent")
    fig_delta = create_delta_aci_chart(record)
    if fig_delta is not None:
        fig_delta.update_layout(transition_duration=500)
        st.plotly_chart(
            fig_delta,
            use_container_width=True,
            key=f"delta_{record.experiment_id}",
        )
        info_banner(
            "Delta_ACI mesure l'augmentation relative du cout lorsqu'on passe "
            "du modele CCI-only au modele CCI+ACI pour la meme allocation. "
            "Un Delta eleve = solution vulnerable aux fuites de canal adjacent.",
            kind="info",
        )
    else:
        info_banner(
            "Delta_ACI necessite que les deux modes (CCI-only ET CCI+ACI) "
            "soient actives dans la configuration.",
            kind="warning",
        )

    st.divider()

    # Radar
    section_title("Profils multi-metriques (radar normalise)")

    available_modes = list(record.metrics.get("BD-CeNN", {}).keys())
    if not available_modes:
        info_banner("Aucune donnee pour le radar.", kind="warning")
        return

    radar_tabs = st.tabs([MODE_LABELS[m] for m in available_modes])
    for tab, mode in zip(radar_tabs, available_modes):
        with tab:
            fig_radar = create_radar_chart(record, mode)
            if fig_radar is not None:
                fig_radar.update_layout(transition_duration=500)
                st.plotly_chart(
                    fig_radar,
                    use_container_width=True,
                    key=f"radar_{mode}_{record.experiment_id}",
                )
            else:
                info_banner(f"Pas de donnees pour {MODE_LABELS[mode]}.", kind="warning")