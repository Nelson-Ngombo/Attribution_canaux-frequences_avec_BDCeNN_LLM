"""
BD-CeNN + LLM Dashboard - Main Entry Point.

Launch:
    streamlit run main.py

Automatically performs on startup:
    1. Generation of topologies if missing (fast, one-shot)
    2. Rendering of the tabbed interface

Tests are opt-in via sidebar to keep startup fast.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration (MUST be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BD-CeNN + LLM",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "BD-CeNN + LLM Framework\n\n"
            "Attribution de canaux neuro-symbolique sous contraintes "
            "d'interference.\n\n"
            "Auteur : Nelson Ngombo"
        ),
    },
)

# ---------------------------------------------------------------------------
# Global error boundary
# ---------------------------------------------------------------------------
def _run_dashboard():
    from dashboard.theme import apply_theme, render_app_header
    apply_theme()

    from dashboard import session_manager as sm
    sm.initialize_state()

    # Bootstrap (only topology generation is automatic)
    from dashboard.bootstrap import render_bootstrap_ui, render_manual_test_runner_in_sidebar
    render_bootstrap_ui()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("### BD-CeNN + LLM")
        st.caption(
            "Framework neuro-symbolique pour l'attribution de canaux radio "
            "sous contraintes d'interference."
        )

        st.markdown("### Etat du systeme")

        # Topologies
        if st.session_state.get("bootstrap_topo_ok", False):
            st.success("Topologies OK")
        else:
            st.error("Topologies indisponibles")
            if st.button("Regenerer maintenant", use_container_width=True):
                from dashboard.bootstrap import generate_topologies
                with st.spinner("Regeneration..."):
                    ok, msg = generate_topologies()
                if ok:
                    st.session_state["bootstrap_topo_ok"] = True
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        # LLM status (non-blocking, updates dynamically)
        try:
            from llm_assistant import (
                get_llm_status_label, is_llm_available,
                is_llm_initializing, get_initialization_error,
            )
            status_label = get_llm_status_label()

            if is_llm_available():
                st.success(f"LLM : {status_label}")
            elif is_llm_initializing():
                st.info(f"LLM : {status_label}")
                st.caption("Le test de connectivite tourne en arriere-plan. L'interface reste fluide.")
            else:
                st.warning(f"LLM : {status_label}")
                err = get_initialization_error()
                if err:
                    with st.expander("Detail de l'erreur"):
                        st.caption(err)
        except Exception as e:
            st.warning(f"LLM : erreur de chargement ({type(e).__name__})")

        # Optional test runner
        render_manual_test_runner_in_sidebar()

        st.markdown("### Configuration interactive")

        seed_global = st.number_input(
            "Seed global (usage interactif)",
            min_value=1, max_value=10000,
            value=sm.get("custom_seed", 1),
            key="sidebar_seed",
        )
        if seed_global != sm.get("custom_seed"):
            sm.set_value("custom_seed", int(seed_global))

        st.markdown("### Modeles d'interference")
        st.caption(
            "**CCI-only** : conflit si x_i = x_j.\n\n"
            "**CCI+ACI** : ajoute la penalite adjacente via matrice M "
            "(decroissance |x_i - x_j| <= cutoff)."
        )

        st.markdown("### Auteur")
        st.caption("Nelson Ngombo")

    # ---------- Header ----------
    render_app_header(
        title="BD-CeNN + LLM",
        subtitle=(
            "Attribution neuro-symbolique de canaux radio - "
            "Solveur BD-CeNN, baselines heuristiques, audit LLM avec garde-fou regex"
        ),
    )

    # ---------- Tabs ----------
    from dashboard import tab_config
    from dashboard import tab_graph
    from dashboard import tab_convergence
    from dashboard import tab_comparison
    from dashboard import tab_llm_report
    from dashboard import tab_experiments
    from dashboard import tab_export

    tab_labels = [
        "Configuration",
        "Graphe reseau",
        "Convergence",
        "Comparaison",
        "Rapport LLM",
        "Campagnes E1-E10",
        "Export",
    ]

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        tab_config.render()
    with tabs[1]:
        tab_graph.render()
    with tabs[2]:
        tab_convergence.render()
    with tabs[3]:
        tab_comparison.render()
    with tabs[4]:
        tab_llm_report.render()
    with tabs[5]:
        tab_experiments.render()
    with tabs[6]:
        tab_export.render()

    # Footer
    st.divider()
    import config
    st.caption(
        f"Scenarios : {', '.join(config.SCENARIOS.keys())} | "
        f"Runs par scenario : {config.NUM_RUNS} | "
        f"Restarts BD-CeNN par defaut : {config.NUM_RESTARTS}"
    )


# ---------------------------------------------------------------------------
# Entry point with catch-all exception handler
# ---------------------------------------------------------------------------
try:
    _run_dashboard()
except Exception as e:
    import traceback
    st.error(
        f"Une erreur fatale s'est produite dans l'application.\n\n"
        f"**Type** : {type(e).__name__}\n\n"
        f"**Message** : {e}"
    )
    with st.expander("Traceback complet"):
        st.code(traceback.format_exc(), language="python")
    st.info(
        "Actions possibles :\n"
        "1. Cliquez sur 'Reinitialiser la session' dans la sidebar\n"
        "2. Relancez l'application (Ctrl+C dans le terminal, puis streamlit run main.py)\n"
        "3. Regenerez les topologies si le probleme concerne les donnees"
    )