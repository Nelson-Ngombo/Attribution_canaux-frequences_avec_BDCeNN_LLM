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
    if record is None or solver_name not in record.solver_results:
        return None
    modes_dict = record.solver_results[solver_name]
    if mode not in modes_dict:
        return None
    return modes_dict[mode].assignment


def render():
    """Point d'entree de l'onglet."""
    st.header("Graphe topologique interactif")

    topology = sm.get("topology")
    if topology is None:
        empty_state(
            icon_text="[]",
            title="Aucune topologie disponible",
            description="Generez d'abord une topologie dans l'onglet 'Configuration'.",
        )
        return

    render_scenario_summary(topology)
    st.divider()

    record = sm.get("experiment_record")

    st.caption(
        "Les noeuds du graphe representent les cellules du rèseau. "
        "Les couleurs des noeuds correspondent aux canaux affectes. Les aretes representent "
        "les couplages radio geographiques. Elles s'affichent en rouge si elles violent la contrainte "
        "co-canal, et en orange si elles provoquent un debordement adjacent."
    )

    # ------------------------------------------------------------------
    # Panneau de contrôle de la vue
    # ------------------------------------------------------------------
    ctrl_cols = st.columns([1.2, 1, 1, 1])

    with ctrl_cols[0]:
        if record is not None:
            available_solvers = list(record.solver_results.keys())
            solver = solver_selector(
                available_solvers=available_solvers,
                session_key="graph_view_solver",
                label="Choix du solveur",
                default="BD-CeNN",
            )
        else:
            solver = None
            st.selectbox(
                "Choix du solveur",
                options=["(Aucun run effectue)"],
                disabled=True,
                key="graph_solver_disabled",
            )

    with ctrl_cols[1]:
        mode = mode_selector(
            session_key="graph_view_mode",
            available_modes=["cci", "cci_aci"],
            label="Regime d'analyse",
        )

    with ctrl_cols[2]:
        show_only_conflicts = st.checkbox(
            "Masquer les liens sains",
            value=sm.get("graph_show_only_conflicts", False),
            key="graph_only_conflicts_chk",
        )
        sm.set_value("graph_show_only_conflicts", show_only_conflicts)

    with ctrl_cols[3]:
        show_labels = st.checkbox(
            "Numeros de cellules",
            value=sm.get("graph_show_labels", True),
            key="graph_show_labels",
        )

    # Légende explicite
    with st.expander("Voir la nomenclature des aretes d'interfèrence"):
        st.markdown("""
        **Noeuds** : Chaque couleur est un canal discret affecte.
        
        **Aretes actives** :
        - **Vert** : Liaison saine. Les deux cellules sont spectralement ecartees.
        - **Rouge** : Conflit co-canal strict ($C_{\mathrm{CCI}}$). Même canal sur deux voisines.
        - **Orange pointille** : Conflit adjacent strict ($C_{\mathrm{ACI}}$). Canaux contigus ($0 < |x_i - x_j| \\le 2$).
        """)

    st.divider()

    # ------------------------------------------------------------------
    # Rendu du Graphe
    # ------------------------------------------------------------------
    graph = InterferenceGraph(topology)
    assignment = None
    solver_selected = solver if record is not None else None

    if solver_selected is not None:
        assignment = _get_assignment_for_view(record, solver_selected, mode)
        if assignment is None:
            st.warning(
                f"Pas de resultats pour le solveur {solver_selected} sous le mode "
                f"{'CCI-only' if mode == 'cci' else 'CCI+ACI'}. Lancez d'abord une simulation."
            )

    # Affichage des KPIs d'affectation (Chantier A)
    if assignment is not None:
        cost = graph.compute_cost(assignment, mode)
        counts = graph.get_all_conflict_counts(assignment, mode)
        n_used = len(set(assignment.tolist()))
        
        kpi_row([
            {"label": "Cout final J(x*)", "value": f"{cost:.4f}"},
            {"label": "Conflits co-canal C_CCI", "value": str(counts["n_conflicts_cci"])},
            {"label": "Conflits adjacents C_ACI", "value": str(counts["n_conflicts_aci"])},
            {"label": "Total C_total", "value": str(counts["n_conflicts_total"])},
        ])

    st.markdown("---")

    title = (
        f"{solver_selected or 'Topologie nue'} - "
        f"{'CCI-only' if mode == 'cci' else 'CCI+ACI'} - "
        f"Scenario {topology.scenario_name} (N={topology.N}, seed={topology.seed})"
    )

    render_cols = st.columns([4, 1])
    with render_cols[1]:
        backend = st.radio(
            "Type de rendu",
            options=["Plotly", "PyVis (experimental)"],
            key="graph_backend_radio",
            index=0,
        )

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
            fig.update_layout(
                autosize=True,
                width=None,
                height=650,
                transition_duration=400,
            )
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
                st.error(f"Erreur PyVis : {e}")

    # Détails des liaisons incriminées
    if assignment is not None:
        st.divider()
        conflict_edges = graph.get_conflict_edges_cci(assignment)
        adj_edges = graph.get_conflict_edges_adjacent(assignment) if mode == "cci_aci" else []

        with st.expander(
            f"Consulter la liste detaillee des liaisons conflictuelles ({len(conflict_edges)} CCI"
            + (f", {len(adj_edges)} ACI" if mode == "cci_aci" else "")
            + ")"
        ):
            if conflict_edges:
                st.markdown("**Aretes en conflit co-canal strict (C_CCI) :**")
                for (i, j) in conflict_edges[:50]:
                    st.text(f"  Cellule {i} <-> Cellule {j} | canal commun {int(assignment[i])} | poids W_ij={graph.W[i, j]:.1d}")
                if len(conflict_edges) > 50:
                    st.caption(f"... et {len(conflict_edges) - 50} autres arêtes.")

            if adj_edges and mode == "cci_aci":
                st.markdown("**Aretes en conflit adjacent strict (C_ACI) :**")
                for (i, j) in adj_edges[:50]:
                    st.text(f"  Cellule {i} <-> Cellule {j} | canaux voisins {int(assignment[i])}/{int(assignment[j])} | poids W_ij={graph.W[i, j]:.1d}")
                if len(adj_edges) > 50:
                    st.caption(f"... et {len(adj_edges) - 50} autres arêtes.")