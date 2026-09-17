"""
Tab 6: Interactive Campaign Runner for experiments E1 to E10.

Allows the user to:
    - Browse experiment cards with full metadata
    - Select one, multiple, or all experiments via checkboxes
    - Launch a serial campaign in the background
    - Monitor progress live with logs and status
    - Cancel the campaign
    - View outputs directory and completion summary

All experiments preserve their original 30-seed reproducibility.
Runs are serial (never parallel) to avoid np.random.seed contamination.
"""

import time
from pathlib import Path

import streamlit as st

import config
from dashboard.theme import section_title, status_badge, mode_tag
from dashboard import toast
from dashboard.experiment_wrappers import (
    list_experiments,
    get_experiment_metadata,
)
from dashboard import campaign_manager as cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = int(seconds % 60)
    return f"{minutes}m {remainder}s"


def _get_state_badge(state: str) -> str:
    mapping = {
        "idle": ("EN ATTENTE", "neutral"),
        "running": ("EN COURS", "info"),
        "done": ("TERMINEE", "success"),
        "aborted": ("ANNULEE", "warning"),
        "failed": ("ECHOUEE", "danger"),
    }
    label, kind = mapping.get(state, (state.upper(), "neutral"))
    return status_badge(label, kind=kind)


def _initialize_selection_state():
    """Initializes checkbox state for each experiment."""
    for code in list_experiments():
        key = f"exp_selected_{code}"
        if key not in st.session_state:
            st.session_state[key] = False


def _get_selected_codes() -> list:
    """Returns the list of currently selected experiment codes."""
    return [
        code for code in list_experiments()
        if st.session_state.get(f"exp_selected_{code}", False)
    ]


# ---------------------------------------------------------------------------
# UI sub-panels
# ---------------------------------------------------------------------------

def _render_experiment_card(code: str, meta: dict, disabled: bool):
    """Renders a single experiment card with a checkbox."""
    with st.container():
        st.markdown('<div class="experiment-card">', unsafe_allow_html=True)

        col_check, col_info = st.columns([0.08, 0.92])

        with col_check:
            key = f"exp_selected_{code}"
            checked = st.checkbox(
                label=f"select_{code}",
                value=st.session_state.get(key, False),
                key=key,
                disabled=disabled,
                label_visibility="collapsed",
            )

        with col_info:
            # Header line: code + title + mode tags
            tags_html = ""
            if meta.get("supports_cci"):
                tags_html += mode_tag("cci")
            if meta.get("supports_aci"):
                tags_html += mode_tag("cci_aci")

            st.markdown(
                f'<span class="experiment-card-code">{code}</span>'
                f'<strong>{meta["title"]}</strong>'
                f'&nbsp;&nbsp;{tags_html}',
                unsafe_allow_html=True,
            )

            st.caption(meta["description"])

            info_cols = st.columns(4)
            info_cols[0].caption(f"**Scenario** : {meta['scenario']}")
            info_cols[1].caption(f"**Seeds** : {meta['seeds_used']}")
            info_cols[2].caption(f"**Duree estimee** : {meta['estimated_time']}")
            info_cols[3].caption(f"**Sortie** : `{meta['outputs_dir']}`")

        st.markdown('</div>', unsafe_allow_html=True)


def _render_selection_panel(campaign_running: bool):
    """Renders the top control panel for selecting and launching."""
    st.markdown("### Selection des experiences a lancer")

    ctrl_cols = st.columns([1, 1, 1, 2])

    with ctrl_cols[0]:
        if st.button("Tout selectionner", use_container_width=True, disabled=campaign_running):
            for code in list_experiments():
                st.session_state[f"exp_selected_{code}"] = True
            st.rerun()

    with ctrl_cols[1]:
        if st.button("Tout deselectionner", use_container_width=True, disabled=campaign_running):
            for code in list_experiments():
                st.session_state[f"exp_selected_{code}"] = False
            st.rerun()

    with ctrl_cols[2]:
        light_selection = ["E1", "E3"]
        if st.button("Selection legere (E1+E3)", use_container_width=True, disabled=campaign_running):
            for code in list_experiments():
                st.session_state[f"exp_selected_{code}"] = code in light_selection
            st.rerun()

    with ctrl_cols[3]:
        selected = _get_selected_codes()
        st.markdown(
            f"**{len(selected)}** experience(s) selectionnee(s) : "
            f"`{', '.join(selected) if selected else 'aucune'}`"
        )


def _render_launch_bar(campaign_running: bool):
    """Renders the launch/cancel action bar."""
    selected = _get_selected_codes()

    st.divider()

    launch_cols = st.columns([1, 1, 3])

    with launch_cols[0]:
        launch_disabled = campaign_running or len(selected) == 0
        if st.button(
            "Lancer la campagne selectionnee",
            type="primary",
            use_container_width=True,
            disabled=launch_disabled,
        ):
            try:
                cm.start_new_campaign(selected)
                toast.success(f"Campagne lancee : {', '.join(selected)}")
                time.sleep(0.5)
                st.rerun()
            except RuntimeError as e:
                toast.error(str(e))
            except Exception as e:
                toast.error(f"Erreur au lancement : {type(e).__name__}: {e}")

    with launch_cols[1]:
        if campaign_running:
            if st.button(
                "Annuler la campagne",
                type="secondary",
                use_container_width=True,
            ):
                campaign = cm.get_current_campaign()
                if campaign:
                    campaign.request_stop()
                    toast.warning("Annulation demandee. L'experience en cours va se terminer.")

    with launch_cols[2]:
        if launch_disabled and not campaign_running:
            st.caption("Selectionnez au moins une experience pour activer le bouton.")


def _render_monitoring_panel():
    """Renders the live monitoring panel when a campaign is active."""
    campaign = cm.get_current_campaign()
    if campaign is None:
        return

    snapshot = campaign.get_snapshot()
    state = snapshot["state"]

    st.markdown("### Suivi de la campagne")

    # Header row: badge + info
    header_cols = st.columns([1, 2, 2])
    with header_cols[0]:
        st.markdown(_get_state_badge(state), unsafe_allow_html=True)
    with header_cols[1]:
        st.metric("Duree ecoulee", _format_duration(snapshot["elapsed_seconds"]))
    with header_cols[2]:
        codes_str = ", ".join(snapshot["codes"])
        st.caption(f"**Experiences** : {codes_str}")

    # Summary if finished
    summary = snapshot.get("summary")
    if summary:
        sum_cols = st.columns(4)
        sum_cols[0].metric("Total", summary["total"])
        sum_cols[1].metric("Reussies", summary["succeeded"])
        sum_cols[2].metric("Echouees", summary["failed"])
        sum_cols[3].metric("Duree totale", _format_duration(summary["total_duration_seconds"]))

        # Per-experiment breakdown
        if summary.get("per_experiment"):
            with st.expander("Detail par experience", expanded=(state != "done")):
                import pandas as pd
                rows = []
                for res in summary["per_experiment"]:
                    rows.append({
                        "Code": res["code"],
                        "Statut": "OK" if res["success"] else "ECHEC",
                        "Duree": _format_duration(res["duration_seconds"]),
                        "Erreur": res.get("error") or "-",
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

        if summary.get("fatal_error"):
            with st.expander("Erreur fatale (traceback)"):
                st.code(summary.get("traceback", summary["fatal_error"]))

    # Live logs
    st.markdown("#### Console de sortie (temps reel)")
    logs = campaign.get_logs(tail=200)
    log_text = "\n".join(logs) if logs else "(aucun log pour le moment)"
    st.markdown(
        f'<div class="log-console">{log_text}</div>',
        unsafe_allow_html=True,
    )

    # Actions after completion
    if state in ("done", "aborted", "failed"):
        st.divider()
        action_cols = st.columns([1, 1, 3])
        with action_cols[0]:
            if st.button("Fermer ce suivi", use_container_width=True):
                cm.clear_finished_campaign()
                st.rerun()
        with action_cols[1]:
            if st.button("Voir les fichiers generes", use_container_width=True):
                st.session_state["show_outputs_browser"] = True
                st.rerun()


def _render_outputs_browser():
    """Optional browser to inspect the outputs directory."""
    if not st.session_state.get("show_outputs_browser", False):
        return

    st.markdown("### Fichiers generes")

    results_dir = config.RESULTS_DIR
    if not results_dir.exists():
        st.info("Le dossier results/ est vide.")
        return

    # Group files by experiment folder
    grouped = {}
    for path in results_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(results_dir)
            parts = rel.parts
            if len(parts) >= 2:
                group_key = parts[1] if parts[0] in ("figures", "csv") else parts[0]
            else:
                group_key = parts[0]
            grouped.setdefault(str(group_key), []).append(str(rel))

    if not grouped:
        st.info("Aucun fichier trouve dans results/.")
    else:
        for group, files in sorted(grouped.items()):
            with st.expander(f"{group} ({len(files)} fichiers)"):
                for f in sorted(files)[:100]:
                    st.code(f)
                if len(files) > 100:
                    st.caption(f"... et {len(files) - 100} autres fichiers")

    if st.button("Fermer l'explorateur"):
        st.session_state["show_outputs_browser"] = False
        st.rerun()


# ---------------------------------------------------------------------------
# Auto-refresh mechanism during running campaigns
# ---------------------------------------------------------------------------

def _auto_refresh_if_running():
    """
    Automatically refreshes the page every 2 seconds while a campaign runs.
    Uses a native Streamlit rerun trick (avoids external dependencies).
    """
    campaign = cm.get_current_campaign()
    if campaign is not None and campaign.is_running():
        time.sleep(2.0)
        st.rerun()


# ---------------------------------------------------------------------------
# MAIN RENDERER
# ---------------------------------------------------------------------------

def render():
    """Main tab entrypoint."""
    _initialize_selection_state()

    st.header("Campagnes d'experiences E1 - E10")

    st.caption(
        "Lancez une ou plusieurs experiences en tache de fond. Les experiences "
        "sont executees en serie pour preserver la reproductibilite (30 seeds "
        "fixes par experience). Les resultats sont sauvegardes dans les dossiers "
        "`results/csv/` et `results/figures/` avec separation stricte "
        "co-canal / adjacent."
    )

    # Prerequisites check
    scenarios_ok = Path(config.SCENARIOS_FILE).exists()
    if not scenarios_ok:
        st.error(
            f"Le fichier {config.SCENARIOS_FILE.name} est manquant. "
            f"Regenerez les topologies via le bouton de la sidebar avant de "
            f"lancer une campagne."
        )
        return

    campaign = cm.get_current_campaign()
    campaign_running = campaign is not None and campaign.is_running()

    # Warning for E10 (needs LLM)
    from llm_assistant import is_llm_available
    if not is_llm_available():
        st.info(
            "L'experience E10 (audit LLM) necessite une cle GOOGLE_API_KEY valide. "
            "Les autres experiences (E1-E9) fonctionnent sans LLM."
        )

    # Section 1: Selection panel
    section_title("Selection")
    _render_selection_panel(campaign_running)

    # Section 2: Experiment cards
    for code in list_experiments():
        meta = get_experiment_metadata(code)
        _render_experiment_card(code, meta, disabled=campaign_running)

    # Section 3: Launch bar
    _render_launch_bar(campaign_running)

    # Section 4: Monitoring panel (only if campaign exists)
    if campaign is not None:
        st.divider()
        section_title("Suivi live")
        _render_monitoring_panel()

    # Section 5: Outputs browser
    _render_outputs_browser()

    # Auto-refresh if running
    _auto_refresh_if_running()