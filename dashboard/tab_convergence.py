"""
Tab 3: BD-CeNN Execution with Live Convergence Trajectory.

Enhanced version:
    - Launches all 4 solvers + both interference modes
    - Auto-saves results to outputs/interactive_runs/{cochannel|adjacent}/
    - Displays convergence curve with forward-fill
    - Shows CCI vs CCI+ACI overlay when both modes enabled
    - Full error handling
"""
import streamlit as st

import config

from dashboard import session_manager as sm
from dashboard.chart_factory import (
    create_convergence_plot,
    create_dual_convergence_plot,
    MODE_LABELS,
)
from dashboard.components import (
    section_title,
    kpi_row,
    empty_state,
    info_banner,
    render_scenario_summary,
)


def _render_run_traceability(topology):
    """Bandeau de reproductibilité du run courant (recommandation C de l'encadreur)."""
    record = sm.get("experiment_record")
    bd_modes = record.solver_results.get("BD-CeNN", {}) if record else {}
    bd_meta = (next(iter(bd_modes.values())).metadata if bd_modes else None) or {}
    modes = " + ".join(config.INTERFERENCE_MODE_LABELS[m] for m in bd_modes)
    st.caption(
        f"Traçabilité : topology_seed = {topology.seed} | "
        f"algorithm_seed = {config.algorithm_seed(topology.seed)} | "
        f"num_restarts = {bd_meta.get('num_restarts', sm.get('num_restarts'))} | "
        f"patience_sweeps = {config.PATIENCE_SWEEPS} | "
        f"max_sweeps = {bd_meta.get('max_sweeps', sm.get('max_iter'))} | "
        f"mode {modes or 'solveurs non exécutés'} | code v{config.CODE_VERSION}"
    )


def render():
    """Point d'entree de l'onglet."""
    st.header("Convergence energetique et sweeps")

    topology = sm.get("topology")
    if topology is None:
        empty_state(
            icon_text="[!]",
            title="Topologie absente",
            description="Generez d'abord une topologie dans le premier onglet.",
        )
        return

    render_scenario_summary(topology)
    _render_run_traceability(topology)
    st.divider()

    record = sm.get("experiment_record")
    if record is None:
        empty_state(
            icon_text="[>]",
            title="En attente de simulation",
            description="Lancez les solveurs depuis l'onglet 'Graphe reseau'.",
        )
        return

    bd_results = record.solver_results.get("BD-CeNN", {})
    if not bd_results:
        info_banner("Le solveur BD-CeNN n'a pas produit de resultats.", kind="warning")
        return

    # Présentation des KPIs détaillés (Chantier A & D)
    section_title("Indicateurs de stabilisation BD-CeNN")

    for mode, result in bd_results.items():
        st.markdown(f"**Regime d'evaluation : {MODE_LABELS[mode]}**")
        kpi_row([
            {"label": "Cout final J(x*)", "value": f"{result.cost:.4f}"},
            {"label": "Conflits Co-canal C_CCI", "value": str(result.n_conflicts_cci)},
            {"label": "Conflits Adjacents C_ACI", "value": str(result.n_conflicts_aci)},
            {"label": "Sweeps effectues (n_sweeps)", "value": str(result.n_sweeps)},
        ])

    st.divider()

    # Trajectoires en sweeps (Chantier D)
    section_title("Trajectoires temporelles en sweeps")

    if len(bd_results) == 2:
        st.markdown("**Superposition de convergence : CCI-only vs CCI+ACI**")
        fig_dual = create_dual_convergence_plot(
            bd_cci=bd_results.get("cci"),
            bd_cci_aci=bd_results.get("cci_aci"),
        )
        st.plotly_chart(fig_dual, use_container_width=True, key=f"conv_dual_{record.experiment_id}")

    st.markdown("**Courbes de convergence individuelles**")
    tabs_labels = [MODE_LABELS[m] for m in bd_results.keys()]
    conv_tabs = st.tabs(tabs_labels)
    for tab, (mode, result) in zip(conv_tabs, bd_results.items()):
        with tab:
            fig = create_convergence_plot(result, mode)
            st.plotly_chart(fig, use_container_width=True, key=f"conv_{mode}_{record.experiment_id}")

            with st.expander(f"Visualiser le detail de convergence ({MODE_LABELS[mode]})"):
                import pandas as pd
                curve = result.energy_curve_forward_filled()
                df = pd.DataFrame({
                    "Sweep Index": list(range(len(curve))),
                    "Cout best-so-far": curve,
                })
                st.dataframe(df, use_container_width=True, height=250)

    # Delta_ACI (Eq. delta-aci) : optimum CCI+ACI comparé à l'optimum CCI-only
    if "cci" in bd_results and "cci_aci" in bd_results:
        st.divider()
        section_title("Métrique Delta_ACI")
        delta = record.delta_aci_percent
        kpi_row([
            {"label": "J_CCI (optimum CCI-only)", "value": f"{bd_results['cci'].cost:.4f}"},
            {"label": "J_CCI+ACI (optimum CCI+ACI)", "value": f"{bd_results['cci_aci'].cost:.4f}"},
            {"label": "Delta_ACI (BD-CeNN)",
             "value": f"{delta:+.2f} %" if delta is not None else "Non défini",
             "help": "(J_CCI+ACI - J_CCI) / J_CCI x 100, non défini si J_CCI = 0."},
        ])
        info_banner(
            "Le Delta_ACI compare le coût de l'allocation optimisée en régime CCI+ACI au coût "
            "de l'allocation optimisée en régime CCI-only, chacun évalué dans son propre régime. "
            "Il mesure le surcoût d'interférence introduit par les canaux adjacents sur cette "
            "topologie, et non la robustesse d'une allocation. Une valeur élevée est attendue "
            "lorsque K est faible devant le cutoff ou que le graphe est dense.",
            kind="info",
        )