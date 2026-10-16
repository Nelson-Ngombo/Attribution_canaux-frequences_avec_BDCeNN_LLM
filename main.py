#main.py
"""
BD-CeNN + LLM Dashboard - Main Entry Point.

Launch command:
    streamlit run main.py

Automatically performs on startup:
    1. Generation of topologies if missing (one-shot)
    2. Rendering of the tabbed interface
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
    from dashboard.theme import (apply_theme, render_app_header, render_sidebar_status_card,
                                 render_sidebar_llm_params, render_sidebar_footer,
                                 install_plotly_resize_fix)
    apply_theme()
    install_plotly_resize_fix()

    from dashboard import session_manager as sm
    sm.initialize_state()

    # Bootstrap (only topology generation is automatic)
    from dashboard.bootstrap import render_bootstrap_ui
    render_bootstrap_ui()

    # ---------- Sidebar ----------
    with st.sidebar:
        # Title at the very top (no extra spacing above)
        # Titre hors style h3 : aucune marge héritée au-dessus
        st.markdown('<div class="sidebar-brand">BD-CeNN + LLM</div>', unsafe_allow_html=True)
        st.markdown(
            "<p style='margin-top:-0.3rem;'>Framework neuro-symbolique pour "
            "l'attribution de canaux radio sous contraintes d'interference.</p>",
            unsafe_allow_html=True
        )

        st.markdown("### Etat du systeme")

        # Topology status card
        if st.session_state.get("bootstrap_topo_ok", False):
            render_sidebar_status_card("Topologies", "Operationnelles", kind="ok")
        else:
            render_sidebar_status_card("Topologies", "Indisponibles", kind="err")
            if st.button("Regenerer maintenant", use_container_width=True, key="btn_regen_topo"):
                from dashboard.bootstrap import generate_topologies
                with st.spinner("Regeneration..."):
                    ok, msg = generate_topologies()
                if ok:
                    st.session_state["bootstrap_topo_ok"] = True
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        # Carte LLM et paramètres : sondage isolé dans un fragment, sans réexécution globale
        try:
            from llm_assistant import (
                is_llm_available, get_initialization_error,
                get_active_model, get_llm_inference_params,
            )

            llm_pending = not is_llm_available() and not get_initialization_error()

            @st.fragment(run_every=2.0 if llm_pending else None)
            def _render_llm_sidebar_block():
                if is_llm_available():
                    model_short = get_active_model().replace("models/", "")
                    render_sidebar_status_card("Assistant LLM", model_short, kind="ok")
                elif not get_initialization_error():
                    render_sidebar_status_card("Assistant LLM", "Initialisation...", kind="init")
                    st.caption(
                        "Connexion au modele en arriere-plan. "
                        "L'application reste utilisable pour toutes les fonctions non-LLM."
                    )
                else:
                    render_sidebar_status_card("Assistant LLM", "Non disponible", kind="err")
                    with st.expander("Detail de l'erreur"):
                        st.caption(get_initialization_error())

                st.markdown("### Parametres LLM")
                render_sidebar_llm_params(get_llm_inference_params())

                # Fin de l'initialisation : une seule réexécution globale arrête le sondage
                if llm_pending and (is_llm_available() or get_initialization_error()):
                    st.rerun()

            _render_llm_sidebar_block()
        except Exception as e:
            render_sidebar_status_card("Assistant LLM", f"Erreur ({type(e).__name__})", kind="err")

        st.markdown("### Modeles d'interference")
        st.caption(
            "**CCI-only** : conflit si x_i = x_j.\n\n"
            "**CCI+ACI** : ajoute la penalite adjacente via matrice M "
            "(decroissance |x_i - x_j| <= cutoff)."
        )

        st.markdown("### Auteur")
        import config
        render_sidebar_footer(author="Nelson Ngombo", code_version=config.CODE_VERSION)

    # ---------- Header ----------
    render_app_header(
        title="BD-CeNN + LLM",
        subtitle=(
            "Attribution neuro-symbolique de canaux radio - "
            "Solveur BD-CeNN, baselines heuristiques, audit LLM avec garde-fou regex"
        ),
    )

    # ---------- Tabs (reordered: Export before Campagnes) ----------
    from dashboard import tab_config
    from dashboard import tab_graph
    from dashboard import tab_convergence
    from dashboard import tab_comparison
    from dashboard import tab_llm_report
    from dashboard import tab_export
    from dashboard import tab_experiments

    tab_labels = [
        "Configuration",
        "Graphe reseau",
        "Convergence",
        "Comparaison",
        "Rapport LLM",
        "Export",
        "Campagnes E1-E10",
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
        tab_export.render()
    with tabs[6]:
        tab_experiments.render()

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
        "1. Relancez l'application (Ctrl+C dans le terminal, puis streamlit run main.py)\n"
        "2. Regenerez les topologies si le probleme concerne les donnees"
    )