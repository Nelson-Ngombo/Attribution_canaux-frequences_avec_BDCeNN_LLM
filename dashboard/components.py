"""
Reusable UI components for the BD-CeNN + LLM dashboard.
Ensures visual consistency across all tabs by unifying layout patterns.
Exposes theme-level styling blocks alongside structural components.
"""

from typing import Optional
import streamlit as st

# Import theme elements to expose them through the components interface
from dashboard.theme import COLORS, status_badge, mode_tag, section_title


def metric_card(label: str, value: str, delta: Optional[str] = None,
                delta_color: str = "normal", help_text: Optional[str] = None):
    """
    Wrapper around st.metric with consistent styling.

    Args:
        label (str): Title of the metric card.
        value (str): Main scalar value displayed.
        delta (str, optional): Change indicator.
        delta_color (str): Layout coloring strategy ('normal' | 'inverse' | 'off').
        help_text (str, optional): Tooltip text.
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def kpi_row(kpis: list):
    """
    Renders an horizontally aligned grid of styled KPI metric cards.

    Args:
        kpis (list): List of dictionaries containing keys:
                     {"label": str, "value": str, "delta": str (opt), "help": str (opt)}
    """
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            metric_card(
                label=kpi["label"],
                value=kpi["value"],
                delta=kpi.get("delta"),
                help_text=kpi.get("help"),
            )


def info_banner(message: str, kind: str = "info"):
    """
    Renders an alert box styled to match the central palette.

    Args:
        message (str): Descriptive text.
        kind (str): Semantic style category ('info' | 'success' | 'warning' | 'error').
    """
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)


def empty_state(icon_text: str, title: str, description: str,
                action_label: Optional[str] = None,
                action_callback=None):
    """
    Displays an elegant placeholder dashboard when data has not yet been computed.

    Args:
        icon_text (str): Unicode symbol or shorthand icon representation.
        title (str): Bold contextual header.
        description (str): Helper text directing the user on how to resolve the state.
        action_label (str, optional): Call to action button text.
        action_callback (callable, optional): Execution logic bound to the button.
    """
    st.markdown(
        f"""
        <div style="text-align:center; padding: 3rem 1rem;
                    background: #f8fafc; border-radius: 12px;
                    border: 2px dashed #cbd5e1; margin: 1rem 0;">
            <div style="font-size: 2rem; color: #64748b; margin-bottom: 0.5rem;">
                {icon_text}
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #1e3a5f;
                        margin-bottom: 0.5rem;">
                {title}
            </div>
            <div style="font-size: 0.9rem; color: #64748b;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if action_label and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label, use_container_width=True, key=f"empty_action_{title}"):
                action_callback()


def mode_selector(session_key: str = "current_view_mode",
                  available_modes: Optional[list] = None,
                  label: str = "Mode d'interference") -> str:
    """
    Renders a unified radio selector to toggle between CCI and CCI+ACI evaluations.

    Args:
        session_key (str): Target state key inside session_state.
        available_modes (list, optional): Options allowed ('cci' and/or 'cci_aci').
        label (str): Text label.

    Returns:
        str: Selected interference mode identifier.
    """
    if available_modes is None:
        available_modes = ["cci", "cci_aci"]

    current = st.session_state.get(session_key, available_modes[0])
    if current not in available_modes:
        current = available_modes[0]

    mode = st.radio(
        label,
        options=available_modes,
        format_func=lambda m: "CCI-only" if m == "cci" else "CCI+ACI",
        horizontal=True,
        index=available_modes.index(current),
        key=f"radio_{session_key}",
    )
    st.session_state[session_key] = mode
    return mode


def solver_selector(available_solvers: list,
                     session_key: str = "current_view_solver",
                     label: str = "Solveur",
                     default: str = "BD-CeNN") -> str:
    """
    Renders a standardized solver selection dropdown, prioritizing the BD-CeNN solver.

    Args:
        available_solvers (list): Solvers currently calculated and available.
        session_key (str): Target state key inside session_state.
        label (str): Text label.
        default (str): Initial preferred selection.

    Returns:
        str: Selected solver identifier.
    """
    if not available_solvers:
        return None
    if default not in available_solvers:
        default = available_solvers[0]

    current = st.session_state.get(session_key, default)
    if current not in available_solvers:
        current = default

    solver = st.selectbox(
        label,
        options=available_solvers,
        index=available_solvers.index(current),
        key=f"select_{session_key}",
    )
    st.session_state[session_key] = solver
    return solver


def divider_with_label(label: str):
    """
    Draws a styled horizontal line with a centered uppercase label.

    Args:
        label (str): Label content.
    """
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin: 1.5rem 0;">
            <div style="flex: 1; height: 1px; background: #e2e8f0;"></div>
            <div style="padding: 0 1rem; color: #64748b; font-size: 0.85rem;
                        text-transform: uppercase; letter-spacing: 0.05em;
                        font-weight: 600;">
                {label}
            </div>
            <div style="flex: 1; height: 1px; background: #e2e8f0;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_pill(label: str, value: str, color: str = COLORS["accent"]):
    """
    Returns an HTML string representing a compact, pill-shaped status badge.

    Args:
        label (str): KPI name.
        value (str): KPI value.
        color (str): Hex color value for text and border blending.
    """
    return (
        f'<span style="display:inline-block; padding: 0.3rem 0.8rem; '
        f'border-radius: 20px; background: {color}20; color: {color}; '
        f'font-size: 0.85rem; font-weight: 600; margin: 0.2rem;">'
        f'{label}: <strong>{value}</strong></span>'
    )


def render_error_box(error_type: str, message: str, traceback_text: Optional[str] = None):
    """
    Renders a styled error block with an optional traceback details drop-down.
    """
    st.error(f"**{error_type}** : {message}")
    if traceback_text:
        with st.expander("Traceback complet"):
            st.code(traceback_text, language="python")


def render_scenario_summary(topology, show_expand: bool = True):
    """
    Renders a horizontal row of compact status pills describing the active topology.
    """
    if topology is None:
        return

    cols = st.columns([1, 1, 1, 1, 1])
    cols[0].markdown(stat_pill("Scenario", str(topology.scenario_name or "CUSTOM")),
                     unsafe_allow_html=True)
    cols[1].markdown(stat_pill("N", str(topology.N), COLORS["primary"]),
                     unsafe_allow_html=True)
    cols[2].markdown(stat_pill("K", str(topology.K), COLORS["primary"]),
                     unsafe_allow_html=True)
    cols[3].markdown(stat_pill("Seed", str(topology.seed), COLORS["info"]),
                     unsafe_allow_html=True)
    cols[4].markdown(stat_pill("Aretes", str(topology.n_edges), COLORS["accent"]),
                     unsafe_allow_html=True)