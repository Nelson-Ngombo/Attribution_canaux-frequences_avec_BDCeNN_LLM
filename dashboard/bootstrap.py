"""
Application bootstrap manager.
Handles the initial setup sequence when the dashboard starts.

Only the topology generation is mandatory. Test execution is now
opt-in (via sidebar checkbox) since it slows down startup significantly.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

import config


def _topologies_exist() -> bool:
    """Checks whether the scenarios JSON file exists and is non-empty."""
    p = Path(config.SCENARIOS_FILE)
    return p.exists() and p.stat().st_size > 100


def generate_topologies(progress_callback: Optional[Callable[[str], None]] = None) -> tuple:
    """
    Runs data_generator.generate_all_scenarios() in-process.
    Returns (success: bool, message: str).
    """
    try:
        if progress_callback:
            progress_callback("Generation des topologies (30 seeds par scenario)...")

        from data_generator import generate_all_scenarios
        generate_all_scenarios()

        return True, f"Topologies generees dans {config.SCENARIOS_FILE.name}"
    except Exception as e:
        return False, f"Echec de generation : {type(e).__name__}: {e}"


def run_tests(progress_callback: Optional[Callable[[str], None]] = None) -> tuple:
    """
    Runs pytest as a subprocess.
    Returns (success: bool, summary: str, full_output: str).
    """
    try:
        if progress_callback:
            progress_callback("Execution des tests unitaires...")

        project_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=180,
        )

        output = result.stdout + "\n" + result.stderr
        success = result.returncode == 0

        lines = output.strip().split("\n")
        summary = "Tests termines"
        for line in reversed(lines):
            if "passed" in line or "failed" in line or "error" in line:
                summary = line.strip()
                break

        return success, summary, output
    except subprocess.TimeoutExpired:
        return False, "Timeout depasse (180s)", "Les tests ont pris trop de temps."
    except FileNotFoundError:
        return False, "pytest non disponible", "Installez pytest via : pip install pytest"
    except Exception as e:
        return False, f"Erreur : {type(e).__name__}", str(e)


def render_bootstrap_ui():
    """
    Renders the bootstrap panel in the UI.
    Only generates topologies if missing. Tests are opt-in via sidebar.
    """
    if st.session_state.get("bootstrap_done", False):
        return

    if _topologies_exist():
        st.session_state["bootstrap_done"] = True
        st.session_state["bootstrap_topo_ok"] = True
        st.session_state["bootstrap_test_ok"] = None
        return

    with st.status(
        "Initialisation de l'application BD-CeNN + LLM...",
        expanded=True,
        state="running",
    ) as status_box:
        st.write("Etape unique : generation des topologies (30 seeds par scenario S1 a S7)")
        st.caption(
            "Cette generation est faite une seule fois. Les topologies sont ensuite "
            "chargees depuis le fichier JSON pour toutes les experiences."
        )

        topo_ok, topo_msg = generate_topologies()

        if topo_ok:
            st.write(f"[OK] {topo_msg}")
            status_box.update(
                label="Application prete", state="complete", expanded=False,
            )
        else:
            st.write(f"[ECHEC] {topo_msg}")
            status_box.update(
                label="Initialisation echouee", state="error", expanded=True,
            )

    st.session_state["bootstrap_done"] = True
    st.session_state["bootstrap_topo_ok"] = topo_ok
    st.session_state["bootstrap_test_ok"] = None
    time.sleep(0.5)
    st.rerun()


def render_manual_test_runner_in_sidebar():
    """
    Sidebar-only optional test runner.
    Call this from main.py's sidebar block.
    """
    with st.expander("Executer les tests unitaires (optionnel)"):
        st.caption(
            "Verifie l'integrite des modules (baselines, solveur, metriques, "
            "verifieur). Prend environ 30-60 secondes."
        )
        if st.button("Lancer pytest", use_container_width=True, key="btn_run_tests_sidebar"):
            with st.spinner("Tests en cours..."):
                ok, summary, output = run_tests()
            if ok:
                st.success(f"Tests OK : {summary}")
                st.session_state["bootstrap_test_ok"] = True
            else:
                st.error(f"Tests echoues : {summary}")
                st.session_state["bootstrap_test_ok"] = False
            with st.expander("Voir la sortie complete"):
                st.code(output, language="text")