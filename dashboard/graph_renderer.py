"""
Graph rendering engine. Produces dynamic interactive network visualizations
with color-coded edges (red = co-channel, orange = adjacent, green = clear)
and colored nodes per assigned channel.

Uses Plotly as the primary backend. PyVis is available as an experimental
alternative but fails gracefully if unavailable or if rendering errors occur.
"""

from typing import Optional
import numpy as np
import plotly.graph_objects as go

from graph_model import InterferenceGraph


# Standard channel color palette (up to 20 distinct colors)
CHANNEL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

# Edge colors
EDGE_COLOR_CLEAR = "#2ecc71"
EDGE_COLOR_COCHANNEL = "#e74c3c"
EDGE_COLOR_ADJACENT = "#f39c12"


def _channel_color(channel_index: int) -> str:
    """Stable color for a given channel index."""
    return CHANNEL_PALETTE[channel_index % len(CHANNEL_PALETTE)]


def render_topology_plotly(
    graph: InterferenceGraph,
    assignment: Optional[np.ndarray] = None,
    mode: str = "cci",
    show_only_conflicts: bool = False,
    show_labels: bool = True,
    title: str = "",
) -> go.Figure:
    """
    Builds a fully interactive Plotly figure of the interference graph.

    Returns a Plotly Figure that never raises even if data is empty.
    """
    try:
        topology = graph.topology
        positions = topology.positions
        N = graph.N
    except Exception:
        fig = go.Figure()
        fig.add_annotation(
            text="Topologie invalide",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="#c0392b"),
        )
        return fig

    edges_by_status = {"clear": [], "cochannel": [], "adjacent": []}

    for i in range(N):
        for j in range(i + 1, N):
            if graph.W[i, j] == 0:
                continue
            if assignment is None:
                status = "clear"
            else:
                try:
                    status = graph.get_edge_status(i, j, assignment, mode)
                    if status == "no_edge":
                        continue
                except Exception:
                    status = "clear"
            edges_by_status[status].append((i, j, graph.W[i, j]))

    edge_traces = []

    def _make_edge_trace(edges_list, color, name, width_scale=1.0, dash=None):
        if not edges_list:
            return None
        xs, ys, widths_info = [], [], []
        for (i, j, w) in edges_list:
            xs.extend([positions[i, 0], positions[j, 0], None])
            ys.extend([positions[i, 1], positions[j, 1], None])
            widths_info.append(w)
        avg_width = float(np.mean(widths_info)) if widths_info else 1.0
        return go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(width=max(1.0, avg_width * width_scale),
                      color=color, dash=dash),
            hoverinfo="skip",
            name=name,
            showlegend=True,
        )

    if not show_only_conflicts:
        t = _make_edge_trace(edges_by_status["clear"], EDGE_COLOR_CLEAR,
                              "Pas de conflit", width_scale=0.5)
        if t: edge_traces.append(t)

    t = _make_edge_trace(edges_by_status["cochannel"], EDGE_COLOR_COCHANNEL,
                          "Conflit co-canal", width_scale=1.2)
    if t: edge_traces.append(t)

    if mode == "cci_aci":
        t = _make_edge_trace(edges_by_status["adjacent"], EDGE_COLOR_ADJACENT,
                              "Conflit adjacent", width_scale=1.0, dash="dash")
        if t: edge_traces.append(t)

    # Node traces
    node_traces = []
    if assignment is not None:
        unique_channels = sorted(set(int(c) for c in assignment))
        for ch in unique_channels:
            idx = [i for i in range(N) if int(assignment[i]) == ch]
            hover_texts = [
                f"Cellule {i}<br>Canal: {ch}<br>Degre: {int(np.sum(graph.W[i] > 0))}"
                for i in idx
            ]
            node_traces.append(go.Scatter(
                x=positions[idx, 0],
                y=positions[idx, 1],
                mode="markers+text" if show_labels else "markers",
                marker=dict(
                    size=18,
                    color=_channel_color(ch),
                    line=dict(width=1.5, color="black"),
                ),
                text=[str(i) for i in idx] if show_labels else None,
                textposition="top center",
                textfont=dict(size=10, color="black"),
                hovertext=hover_texts,
                hoverinfo="text",
                name=f"Canal {ch}",
                showlegend=True,
            ))
    else:
        hover_texts = [
            f"Cellule {i}<br>Degre: {int(np.sum(graph.W[i] > 0))}"
            for i in range(N)
        ]
        node_traces.append(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode="markers+text" if show_labels else "markers",
            marker=dict(size=18, color="lightblue",
                        line=dict(width=1.5, color="black")),
            text=[str(i) for i in range(N)] if show_labels else None,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            hovertext=hover_texts,
            hoverinfo="text",
            name="Cellules (pas d'affectation)",
            showlegend=True,
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        showlegend=True,
        hovermode="closest",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=650,
        legend=dict(
            orientation="v", yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        transition_duration=400,
    )
    return fig


def render_topology_pyvis_html(
    graph: InterferenceGraph,
    assignment: Optional[np.ndarray] = None,
    mode: str = "cci",
    show_only_conflicts: bool = False,
    height: str = "600px",
) -> str:
    """
    Alternative renderer via PyVis. Returns HTML string.
    Fails gracefully with an HTML error message if PyVis unavailable.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return (
            '<div style="padding: 2rem; text-align: center; color: #c0392b;">'
            'PyVis non installe. Executez : pip install pyvis'
            '</div>'
        )

    try:
        net = Network(height=height, width="100%",
                      bgcolor="#ffffff", font_color="#000000", notebook=False)
        net.barnes_hut()

        N = graph.N
        for i in range(N):
            if assignment is not None:
                color = _channel_color(int(assignment[i]))
                label = f"{i} (ch{int(assignment[i])})"
            else:
                color = "#87ceeb"
                label = str(i)
            net.add_node(i, label=label, color=color, size=20)

        for i in range(N):
            for j in range(i + 1, N):
                if graph.W[i, j] == 0:
                    continue
                if assignment is not None:
                    status = graph.get_edge_status(i, j, assignment, mode)
                    if status == "no_edge":
                        continue
                    if status == "cochannel":
                        color = EDGE_COLOR_COCHANNEL
                    elif status == "adjacent" and mode == "cci_aci":
                        color = EDGE_COLOR_ADJACENT
                    else:
                        if show_only_conflicts:
                            continue
                        color = EDGE_COLOR_CLEAR
                else:
                    color = EDGE_COLOR_CLEAR
                net.add_edge(i, j, color=color, width=float(graph.W[i, j]))

        try:
            return net.generate_html(notebook=False)
        except TypeError:
            return net.generate_html()
    except Exception as e:
        return (
            f'<div style="padding: 2rem; text-align: center; color: #c0392b;">'
            f'Erreur PyVis: {type(e).__name__}: {e}<br>'
            f'Utilisez le moteur Plotly a la place.'
            f'</div>'
        )