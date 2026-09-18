"""
Tab: Interactive interference graph with dynamic edge coloring.

Displays:
    - Nodes colored by channel assignment (per solver)
    - Edges colored by conflict status:
        * green = no conflict
        * red = co-channel conflict
        * orange dashed = adjacent-channel conflict (CCI+ACI mode only)
"""

import streamlit as st
import streamlit.components.v1 as components

from graph_model import InterferenceGraph
from dashboard import session_manager as sm
from dashboard.graph_renderer import (
    render_topology_plotly,
    render_topology_pyvis_html,
)
from dashboard.components import (
    section_title,
    kpi_row,
    empty_state,
    mode_selector,
    solver_selector,
    render_scenario_summary,
)


def _get_assignment_for_view(record, solver_name, mode):
    """Retrieves the assignment vector for a (solver, mode) tuple."""
    if record is None or solver_name not in record.solver_results:
        return None
    modes_dict = record.solver_results[solver_name]
    if mode not in modes_dict:
        return None
    return modes_dict[mode].assignment


def render():
    """Main tab entrypoint."""
    st.header("Graphe d'interference interactif")

    topology = sm.get("topology")
    if topology is None:
        empty_state(
            icon_text="[]",
            title="Aucune topologie generee",
            description=(
                "Configurez et generez d'abord une topologie dans l'onglet "
                "'Configuration'."
            ),
        )
        return

    render_scenario_summary(topology)

    st.divider()

    record = sm.get("experiment_record")

    st.caption(
        "Les noeuds sont colores selon le canal attribue par le solveur choisi. "
        "Les aretes sont colorees dynamiquement selon leur statut : "
        "**vert** = pas de conflit, **rouge** = conflit co-canal, "
        "**orange (pointille)** = conflit de canal adjacent (visible uniquement en mode CCI+ACI)."
    )

    # ------------------------------------------------------------------
    # Control panel
    # ------------------------------------------------------------------
    ctrl_cols = st.columns([1.2, 1, 1, 1])

    with ctrl_cols[0]:
        if record is not None:
            available_solvers = list(record.solver_results.keys())
            solver = solver_selector(
                available_solvers=available_solvers,
                session_key="graph_view_solver",
                label="Solveur a afficher",
                default="BD-CeNN",
            )
        else:
            solver = None
            st.selectbox(
                "Solveur a afficher",
                options=["(pas de resultat)"],
                disabled=True,
                key="graph_solver_disabled",
            )

    with ctrl_cols[1]:
        mode = mode_selector(
            session_key="graph_view_mode",
            available_modes=["cci", "cci_aci"],
            label="Mode",
        )

    with ctrl_cols[2]:
        show_only_conflicts = st.checkbox(
            "Uniquement les conflits",
            value=sm.get("graph_show_only_conflicts", False),
            key="graph_only_conflicts_chk",
        )
        sm.set_value("graph_show_only_conflicts", show_only_conflicts)

    with ctrl_cols[3]:
        show_labels = st.checkbox(
            "Etiquettes des cellules",
            value=(topology.N <= 60),
            key="graph_show_labels",
        )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    with st.expander("Legende des couleurs"):
        st.markdown("""
        **Noeuds** : chaque couleur correspond a un canal different attribue a la cellule.

        **Aretes** :
        - **Vert (fin)** : lien d'interference actif sans conflit (canaux differents et distants)
        - **Rouge (epais)** : conflit co-canal (les deux cellules utilisent le meme canal)
        - **Orange (pointille, mode CCI+ACI uniquement)** : fuite de canal adjacent
        """)

    st.divider()

    # ------------------------------------------------------------------
    # Graph rendering
    # ------------------------------------------------------------------
    graph = InterferenceGraph(topology)
    assignment = None
    solver_selected = solver if record is not None else None

    if solver_selected is not None:
        assignment = _get_assignment_for_view(record, solver_selected, mode)
        if assignment is None:
            st.warning(
                f"Aucun resultat pour le solveur **{solver_selected}** en mode "
                f"**{'CCI-only' if mode == 'cci' else 'CCI+ACI'}**. "
                f"Lancez une simulation dans l'onglet Convergence."
            )

    # Metrics for current view
    if assignment is not None:
        cost = graph.compute_cost(assignment, mode)
        n_conf = graph.count_conflicts(assignment, mode)
        n_used = len(set(assignment.tolist()))
        kpi_row([
            {"label": "Cout J(x)", "value": f"{cost:.4f}"},
            {"label": "Conflits", "value": str(n_conf)},
            {"label": "Canaux utilises", "value": str(n_used)},
            {"label": "Aretes actives", "value": str(topology.n_edges)},
        ])

    st.markdown("---")

    title = (
        f"{solver_selected or 'Topologie'} - "
        f"{'CCI-only' if mode == 'cci' else 'CCI+ACI'} - "
        f"Scenario {topology.scenario_name} (N={topology.N}, K={topology.K})"
    )

    # Backend selector
    render_cols = st.columns([4, 1])
    with render_cols[1]:
        backend = st.radio(
            "Moteur",
            options=["Plotly", "PyVis (experimental)"],
            key="graph_backend_radio",
            index=0,
        )

    # ------------------------------------------------------------------
    # BUG FIX: full-width rendering from first paint
    # Use a stable container and force autosize with explicit width handling
    # ------------------------------------------------------------------
    graph_container = st.container()

    with graph_container:
        if backend == "Plotly":
            fig = render_topology_plotly(
                graph,
                assignment=assignment,
                mode=mode,
                show_only_conflicts=show_only_conflicts,
                show_labels=show_labels,
                title=title,
            )
            # Force autosize and disable fixed dimensions in layout
            fig.update_layout(
                autosize=True,
                width=None,
                height=650,
                transition_duration=400,
            )
            # Use a stable key that doesn't change on filter toggles
            # to prevent Streamlit from re-mounting the component with wrong dims
            stable_key = f"graph_plot_{topology.scenario_name}_{topology.seed}"
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=stable_key,
                config={
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                },
            )
        else:
            try:
                html_str = render_topology_pyvis_html(
                    graph,
                    assignment=assignment,
                    mode=mode,
                    show_only_conflicts=show_only_conflicts,
                    height="650px",
                )
                components.html(html_str, height=680, scrolling=True)
            except Exception as e:
                st.error(f"Erreur PyVis : {type(e).__name__}: {e}")
                st.info("Basculez sur Plotly pour un rendu garanti.")

    # ------------------------------------------------------------------
    # Conflict details
    # ------------------------------------------------------------------
    if assignment is not None:
        st.divider()
        conflict_edges = graph.get_conflict_edges_cci(assignment)
        adj_edges = graph.get_conflict_edges_adjacent(assignment) if mode == "cci_aci" else []

        with st.expander(
            f"Detail des conflits ({len(conflict_edges)} co-canal"
            + (f", {len(adj_edges)} adjacent" if mode == "cci_aci" else "")
            + ")"
        ):
            if conflict_edges:
                st.markdown("**Aretes en conflit co-canal :**")
                for (i, j) in conflict_edges[:50]:
                    st.text(
                        f"  Cellule {i} <-> Cellule {j} | canal {int(assignment[i])} | "
                        f"poids {graph.W[i, j]:.1f}"
                    )
                if len(conflict_edges) > 50:
                    st.caption(f"... et {len(conflict_edges) - 50} autres")

            if adj_edges and mode == "cci_aci":
                st.markdown("**Aretes en conflit de canal adjacent :**")
                for (i, j) in adj_edges[:50]:
                    st.text(
                        f"  Cellule {i} <-> Cellule {j} | canaux "
                        f"{int(assignment[i])}/{int(assignment[j])} | "
                        f"poids {graph.W[i, j]:.1f}"
                    )
                if len(adj_edges) > 50:
                    st.caption(f"... et {len(adj_edges) - 50} autres")