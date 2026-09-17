"""
Tab 3: BD-CeNN Execution with Live Convergence Trajectory.

Enhanced version:
    - Launches all 4 solvers + both interference modes
    - Auto-saves results to outputs/interactive_runs/{cochannel|adjacent}/
    - Displays convergence curve with forward-fill
    - Shows CCI vs CCI+ACI overlay when both modes enabled
    - Full error handling
"""

import time
import streamlit as st

from dashboard import session_manager as sm
from dashboard import toast
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
    divider_with_label,
    render_scenario_summary,
    render_error_box,
)
from dashboard.interactive_save import save_interactive_run


def _run_full_experiment():
    """Executes the full ExperimentRunner pipeline."""
    topology = sm.get("topology")
    if topology is None:
        toast.error("Aucune topologie disponible. Generez-en une dans l'onglet Configuration.")
        return

    runner = sm.get("runner")
    if runner is None:
        toast.error("Runner non initialise. Reinitialisez la session.")
        return

    modes = []
    if sm.get("run_mode_cci"):
        modes.append("cci")
    if sm.get("run_mode_cci_aci"):
        modes.append("cci_aci")

    if not modes:
        toast.warning("Activez au moins un mode d'interference dans l'onglet Configuration.")
        return

    sm.set_value("experiment_running", True)

    progress_bar = st.progress(0.0, text="Preparation...")
    status_placeholder = st.empty()

    def progress_callback(msg: str, ratio: float):
        try:
            status_placeholder.info(msg)
            progress_bar.progress(min(1.0, max(0.0, ratio)), text=msg)
        except Exception:
            pass

    try:
        record = runner.run_single(
            topology,
            modes=modes,
            progress_callback=progress_callback,
        )
        sm.set_value("experiment_record", record)
        progress_bar.progress(1.0, text="Termine")
        status_placeholder.success("Simulation terminee avec succes.")

        # Auto-save to disk
        try:
            saved = save_interactive_run(record, save_figures=True)
            n_files = len(saved.get("json", [])) + len(saved.get("csv", [])) + len(saved.get("figures", []))
            if n_files > 0:
                toast.success(f"Resultats sauvegardes : {n_files} fichiers dans outputs/interactive_runs/")
        except Exception as e:
            toast.warning(f"Sauvegarde partielle : {type(e).__name__}: {e}")

    except Exception as e:
        import traceback
        sm.set_value("last_error_message", str(e))
        status_placeholder.error(f"Simulation echouee : {e}")
        with st.expander("Traceback complet"):
            st.code(traceback.format_exc(), language="python")
    finally:
        sm.set_value("experiment_running", False)


def render():
    """Main tab entrypoint."""
    st.header("Execution BD-CeNN et trajectoire de convergence")

    topology = sm.get("topology")
    if topology is None:
        empty_state(
            icon_text="[!]",
            title="Aucune topologie generee",
            description=(
                "Vous devez d'abord generer une topologie dans l'onglet "
                "'Configuration' avant de lancer une simulation."
            ),
        )
        return

    render_scenario_summary(topology)

    st.divider()

    # -----------------------------------------------
    # Execution panel
    # -----------------------------------------------
    section_title("Lancement de la simulation")

    st.caption(
        "Execute le pipeline complet : BD-CeNN multistart + trois baselines "
        "(Random, Greedy, DSATUR) sous les modes d'interference selectionnes. "
        "Les resultats sont automatiquement sauvegardes dans "
        "`outputs/interactive_runs/`."
    )

    exec_cols = st.columns([1, 1, 2])
    with exec_cols[0]:
        launch = st.button(
            "Lancer la simulation",
            type="primary",
            use_container_width=True,
            disabled=sm.get("experiment_running", False),
            key="btn_launch_convergence",
        )
    with exec_cols[1]:
        st.caption(f"**Restarts** : {sm.get('num_restarts')}")
        st.caption(f"**Iter max** : {sm.get('max_iter')}")
    with exec_cols[2]:
        modes_active = []
        if sm.get("run_mode_cci"):
            modes_active.append("CCI-only")
        if sm.get("run_mode_cci_aci"):
            modes_active.append("CCI+ACI")
        st.caption(f"**Modes actifs** : {', '.join(modes_active) if modes_active else 'aucun'}")

    if launch:
        _run_full_experiment()
        st.rerun()

    st.divider()

    # -----------------------------------------------
    # Results display
    # -----------------------------------------------
    record = sm.get("experiment_record")
    if record is None:
        empty_state(
            icon_text="[>]",
            title="Aucune simulation executee",
            description=(
                "Cliquez sur 'Lancer la simulation' ci-dessus pour executer les "
                "solveurs et voir la trajectoire de convergence."
            ),
        )
        return

    bd_results = record.solver_results.get("BD-CeNN", {})
    if not bd_results:
        info_banner("BD-CeNN n'a produit aucun resultat.", kind="warning")
        return

    # -----------------------------------------------
    # Per-mode KPIs
    # -----------------------------------------------
    section_title("Metriques BD-CeNN par mode")

    for mode, result in bd_results.items():
        st.markdown(f"**Mode : {MODE_LABELS[mode]}**")
        kpi_row([
            {"label": "Cout final", "value": f"{result.cost:.4f}"},
            {"label": "Conflits", "value": str(result.n_conflicts)},
            {"label": "Temps (s)", "value": f"{result.wall_time_seconds:.3f}"},
            {"label": "Iterations", "value": str(result.n_iterations)},
        ])

    # -----------------------------------------------
    # Convergence trajectories
    # -----------------------------------------------
    st.divider()
    section_title("Trajectoires de convergence")

    if len(bd_results) == 2:
        st.markdown("**Superposition : CCI-only vs CCI+ACI**")
        fig_dual = create_dual_convergence_plot(
            bd_cci=bd_results.get("cci"),
            bd_cci_aci=bd_results.get("cci_aci"),
        )
        fig_dual.update_layout(transition_duration=400)
        st.plotly_chart(
            fig_dual,
            use_container_width=True,
            key=f"conv_dual_{record.experiment_id}",
        )

    st.markdown("**Vue par mode**")
    tabs_labels = [MODE_LABELS[m] for m in bd_results.keys()]
    conv_tabs = st.tabs(tabs_labels)
    for tab, (mode, result) in zip(conv_tabs, bd_results.items()):
        with tab:
            fig = create_convergence_plot(result, mode)
            fig.update_layout(transition_duration=400)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"conv_{mode}_{record.experiment_id}",
            )

            with st.expander(f"Historique brut ({MODE_LABELS[mode]})"):
                import pandas as pd
                curve = result.energy_curve_forward_filled()
                df = pd.DataFrame({
                    "iteration": list(range(len(curve))),
                    "cost_best_so_far": curve,
                })
                st.dataframe(df, use_container_width=True, height=250)

    # -----------------------------------------------
    # Delta ACI
    # -----------------------------------------------
    if record.delta_aci_percent is not None:
        st.divider()
        section_title("Metrique Delta_ACI")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(
                "Delta_ACI (BD-CeNN)",
                f"+{record.delta_aci_percent:.2f}%",
                help="Augmentation relative du cout en passant de CCI-only a CCI+ACI.",
            )
        with col2:
            info_banner(
                "Un Delta_ACI eleve indique que la solution BD-CeNN optimisee "
                "en mode CCI-only place beaucoup de cellules voisines sur des "
                "canaux adjacents, ce qui devient penalisant sous le modele "
                "physique CCI+ACI.",
                kind="info",
            )