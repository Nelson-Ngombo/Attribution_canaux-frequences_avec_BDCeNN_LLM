"""
Centralized Streamlit session state manager.
Defines and initializes all keys used across tabs.

All state keys used anywhere in the dashboard MUST be declared here
to prevent KeyError at any interaction point.
"""

import streamlit as st
from typing import Any

import config
from experiment_runner import ExperimentRunner
from verifier import ResponseVerifier


# ============================================================================
# COMPLETE DEFAULT STATE MAP
# ============================================================================

_DEFAULT_STATE = {
    # ---------- Topology & experiment lifecycle ----------
    "topology": None,
    "experiment_record": None,
    "runner": None,
    "verifier": None,

    # ---------- Bootstrap state ----------
    "bootstrap_done": False,
    "bootstrap_topo_ok": False,
    "bootstrap_test_ok": None,
    "skip_tests_on_boot": False,

    # ---------- Scenario selection ----------
    "selected_scenario": "S3",
    "use_custom_params": False,
    "custom_N": 30,
    "custom_K": 4,
    "custom_area": 150,
    "custom_threshold": 35,
    "custom_seed": 1,

    # ---------- BD-CeNN hyperparameters ----------
    "num_restarts": config.NUM_RESTARTS,
    "max_iter": config.MAX_ITER_BD,

    # ---------- Interference mode toggles ----------
    "run_mode_cci": True,
    "run_mode_cci_aci": True,

    # ---------- Graph visualization ----------
    "graph_view_solver": "BD-CeNN",
    "graph_view_mode": "cci",
    "graph_show_only_conflicts": False,
    "graph_show_labels": True,
    "graph_renderer_backend": "Plotly",

    # ---------- Convergence view ----------
    "convergence_view_mode": "cci",

    # ---------- Comparison view ----------
    "comparison_metric": "cost",

    # ---------- LLM state ----------
    "llm_response_cci": None,
    "llm_response_cci_aci": None,
    "audit_report_cci": None,
    "audit_report_cci_aci": None,
    "llm_view_mode": "cci",
    "llm_case_data_cci": None,
    "llm_case_data_cci_aci": None,

    # ---------- Natural language translation ----------
    "nl_input_text": "",
    "nl_translated_params": None,

    # ---------- Global flags ----------
    "experiment_running": False,
    "last_error_message": None,

    # ---------- Experiment tab (campaigns E1-E10) ----------
    "exp_selected_E1": False,
    "exp_selected_E2": False,
    "exp_selected_E3": False,
    "exp_selected_E4": False,
    "exp_selected_E5": False,
    "exp_selected_E6": False,
    "exp_selected_E7": False,
    "exp_selected_E8": False,
    "exp_selected_E9": False,
    "exp_selected_E10": False,
    "show_outputs_browser": False,

    # ---------- Export tab ----------
    "export_include_llm": True,
    "export_include_figures": True,
}


# ============================================================================
# KEYS TO PRESERVE ACROSS RESETS
# ============================================================================

# Keys that must survive "Reset session" action
_PRESERVE_ON_RESET = {
    "bootstrap_done",
    "bootstrap_topo_ok",
    "bootstrap_test_ok",
    "skip_tests_on_boot",
}


# ============================================================================
# PUBLIC API
# ============================================================================

def initialize_state() -> None:
    """
    Populates st.session_state with all default keys if not yet present.
    Called at the top of main.py before any tab is rendered.
    Idempotent: safe to call multiple times.
    """
    for key, default_value in _DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Instantiate singletons (only once per session)
    if st.session_state.get("runner") is None:
        try:
            st.session_state["runner"] = ExperimentRunner(
                num_restarts=st.session_state.get("num_restarts", config.NUM_RESTARTS),
                max_iter=st.session_state.get("max_iter", config.MAX_ITER_BD),
            )
        except Exception as e:
            st.session_state["last_error_message"] = f"Runner init failed: {e}"

    if st.session_state.get("verifier") is None:
        try:
            st.session_state["verifier"] = ResponseVerifier(tolerance=0.01)
        except Exception as e:
            st.session_state["last_error_message"] = f"Verifier init failed: {e}"


def reset_experiment_state() -> None:
    """
    Clears experiment-related state without touching UI configuration
    or bootstrap flags. Called after generating a new topology.
    """
    keys_to_reset = [
        "experiment_record",
        "llm_response_cci",
        "llm_response_cci_aci",
        "audit_report_cci",
        "audit_report_cci_aci",
        "llm_case_data_cci",
        "llm_case_data_cci_aci",
        "last_error_message",
    ]
    for key in keys_to_reset:
        st.session_state[key] = _DEFAULT_STATE.get(key)


def reset_full_session() -> None:
    """
    Wipes the entire session state except bootstrap flags.
    Used by the sidebar 'Reinitialiser la session' button.
    """
    preserved = {k: st.session_state.get(k) for k in _PRESERVE_ON_RESET
                 if k in st.session_state}
    for key in list(st.session_state.keys()):
        if key not in _PRESERVE_ON_RESET:
            try:
                del st.session_state[key]
            except Exception:
                pass
    for k, v in preserved.items():
        st.session_state[k] = v
    initialize_state()


def get(key: str, default: Any = None) -> Any:
    """Safe accessor for session state values."""
    return st.session_state.get(key, default)


def set_value(key: str, value: Any) -> None:
    """Safe setter for session state values."""
    st.session_state[key] = value


def refresh_runner() -> None:
    """
    Recreates the ExperimentRunner instance when hyperparameters change.
    """
    try:
        st.session_state["runner"] = ExperimentRunner(
            num_restarts=st.session_state.get("num_restarts", config.NUM_RESTARTS),
            max_iter=st.session_state.get("max_iter", config.MAX_ITER_BD),
        )
    except Exception as e:
        st.session_state["last_error_message"] = f"Runner refresh failed: {e}"


def has_topology() -> bool:
    """Convenience check for topology availability."""
    return st.session_state.get("topology") is not None


def has_experiment_record() -> bool:
    """Convenience check for experiment results availability."""
    return st.session_state.get("experiment_record") is not None