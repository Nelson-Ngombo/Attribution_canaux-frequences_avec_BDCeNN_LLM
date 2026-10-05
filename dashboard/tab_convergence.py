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
    render_scenario_summary,
)
from dashboard.interactive_save import save_interactive_run


def _run_full_experiment():
    topology = sm.get("topology")
    if topology is None:
        toast.error("Aucune topologie disponible.")
        return

    runner = sm.get("runner")
    if runner is None:
        toast.error("Orchestrateur de run absent.")
        return

    modes = []
    if sm.get("run_mode_cci"):
        modes.append("cci")
    if sm.get("run_mode_cci_aci"):
        modes.append("cci_aci")

    if not modes:
        toast.warning("Veuillez selectionner au moins un mode spectral.")
        return

    sm.set_value("experiment_running", True)

    progress_bar = st.progress(0.0, text="Preparation du run...")
    status_placeholder = st.empty()

    def progress_callback(msg: str, ratio: float):
        try:
            status_placeholder.info(msg)
            progress_bar.progress(min(1.0, max(0.0, ratio)), text=msg)
        except Exception:
            pass

    try:
        # Exécution de l'orchestrateur (avec n_sweeps mis à jour)
        record = runner.run_single(
            topology,
            modes=modes,
            progress_callback=progress_callback,
        )
        sm.set_value("experiment_record", record)
        progress_bar.progress(1.0, text="Run accompli")
        status_placeholder.success("Simulation terminee avec succes.")

        # Sauvegarde automatique sur disque
        try:
            saved = save_interactive_run(record, save_figures=True)
            n_files = len(saved.get("json", [])) + len(saved.get("csv", [])) + len(saved.get("figures", []))
            if n_files > 0:
                toast.success(f"Sauvegarde reussie : {n_files} fichiers ecrits dans outputs/")
        except Exception as e:
            toast.warning(f"Sauvegarde partielle : {e}")

    except Exception as e:
        import traceback
        sm.set_value("last_error_message", str(e))
        status_placeholder.error(f"Echec critique : {e}")
        with st.expander("Details techniques (traceback)"):
            st.code(traceback.format_exc(), language="python")
    finally:
        sm.set_value("experiment_running", False)


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
    st.divider()

    section_title("Parametrage et exécution du solveur")

    exec_cols = st.columns([1.2, 1, 1.8])
    with exec_cols[0]:
        launch = st.button(
            "Lancer les solveurs",
            type="primary",
            use_container_width=True,
            disabled=sm.get("experiment_running", False),
            key="btn_launch_convergence",
        )
    with exec_cols[1]:
        st.caption(f"**Restarts multistart** : {sm.get('num_restarts')}")
        st.caption(f"**Sweeps max (T_max)** : {sm.get('max_iter')}")
    with exec_cols[2]:
        modes_active = []
        if sm.get("run_mode_cci"):
            modes_active.append("CCI-only")
        if sm.get("run_mode_cci_aci"):
            modes_active.append("CCI+ACI")
        st.caption(f"**Regimes evalues** : {', '.join(modes_active) if modes_active else 'aucun'}")

    if launch:
        _run_full_experiment()
        st.rerun()

    st.divider()

    record = sm.get("experiment_record")
    if record is None:
        empty_state(
            icon_text="[>]",
            title="En attente de simulation",
            description="Lancez les solveurs pour analyser les dynamiques de convergence.",
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

    # Métrique Delta_ACI
    if record.delta_aci_percent is not None:
        st.divider()
        section_title("Métrique Delta_ACI")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(
                "Delta_ACI (BD-CeNN)",
                f"+{record.delta_aci_percent:.2f}%",
                help="Surcoût relatif induit lors de l'évaluation sous modèle adjacent.",
            )
        with col2:
            info_banner(
                "Le Delta_ACI mesure le saut energetique lorsque la solution physique "
                "est soumise aux debordements spectraux du canal adjacent. Plus il est faible, "
                "plus l'allocation est robuste.",
                kind="info",
            )