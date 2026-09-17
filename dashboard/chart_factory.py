"""
Chart factory. Centralizes creation of all Plotly figures used in the dashboard
(convergence curves, comparison bar charts, delta ACI plots, radar charts).

All figures include transition_duration for smooth animations on data updates.
"""

from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go

from data_structures import ExperimentRecord, SolverResult


# ============================================================================
# COLOR MAPS (aligned with dashboard/theme.py)
# ============================================================================

SOLVER_COLORS = {
    "Random": "#3498db",
    "Greedy": "#27ae60",
    "DSATUR": "#e67e22",
    "BD-CeNN": "#c0392b",
}

MODE_COLORS = {
    "cci": "#3b82f6",
    "cci_aci": "#ef4444",
}

MODE_LABELS = {
    "cci": "CCI-only",
    "cci_aci": "CCI+ACI",
}


# ============================================================================
# CONVERGENCE CURVES (forward-filled)
# ============================================================================

def create_convergence_plot(bd_result: SolverResult, mode: str) -> go.Figure:
    """
    Renders the BD-CeNN convergence trajectory with forward fill.
    Displays a monotone non-increasing best-so-far curve.
    """
    curve = bd_result.energy_curve_forward_filled() if bd_result else []
    iterations = list(range(len(curve)))
    color = MODE_COLORS.get(mode, "#666666")

    fig = go.Figure()

    if curve:
        fig.add_trace(go.Scatter(
            x=iterations,
            y=curve,
            mode="lines+markers",
            line=dict(color=color, width=2.5, shape="hv"),
            marker=dict(size=5),
            name="J(x) best-so-far",
            hovertemplate="Iteration %{x}<br>Cout: %{y:.4f}<extra></extra>",
        ))
        fig.add_annotation(
            x=len(curve) - 1,
            y=curve[-1],
            text=f"Final: {curve[-1]:.4f}",
            showarrow=True,
            arrowhead=2,
            ax=-40, ay=-40,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=color,
            borderwidth=1,
        )

    fig.update_layout(
        title=dict(
            text=f"Convergence BD-CeNN ({MODE_LABELS.get(mode, mode)})",
            x=0.5, font=dict(size=13),
        ),
        xaxis_title="Iteration globale (tous redemarrages)",
        yaxis_title="Cout J(x)",
        hovermode="x unified",
        plot_bgcolor="white",
        height=450,
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        transition_duration=400,
    )
    return fig


def create_dual_convergence_plot(
    bd_cci: Optional[SolverResult],
    bd_cci_aci: Optional[SolverResult],
) -> go.Figure:
    """
    Overlays both CCI-only and CCI+ACI convergence curves.
    """
    fig = go.Figure()

    for result, mode in [(bd_cci, "cci"), (bd_cci_aci, "cci_aci")]:
        if result is None:
            continue
        curve = result.energy_curve_forward_filled()
        if not curve:
            continue
        fig.add_trace(go.Scatter(
            x=list(range(len(curve))),
            y=curve,
            mode="lines+markers",
            line=dict(color=MODE_COLORS[mode], width=2.5, shape="hv"),
            marker=dict(size=4),
            name=MODE_LABELS[mode],
            hovertemplate=f"{MODE_LABELS[mode]}<br>Iter %{{x}}<br>Cout %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Convergence BD-CeNN : CCI-only vs CCI+ACI",
            x=0.5, font=dict(size=13),
        ),
        xaxis_title="Iteration globale",
        yaxis_title="Cout J(x) (best-so-far)",
        hovermode="x unified",
        plot_bgcolor="white",
        height=450,
        legend=dict(x=0.75, y=0.95, bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        transition_duration=400,
    )
    return fig


# ============================================================================
# COMPARISON BAR CHARTS
# ============================================================================

def create_comparison_bar_chart(
    record: ExperimentRecord,
    metric: str = "cost",
) -> go.Figure:
    """
    Grouped bar chart comparing solvers across CCI-only and CCI+ACI.
    """
    metric_labels = {
        "cost": "Cout J(x)",
        "n_conflicts": "Nombre de conflits",
        "used_channels": "Canaux utilises",
        "wall_time_seconds": "Temps d'execution (s)",
    }
    ylabel = metric_labels.get(metric, metric)

    solvers = ["Random", "Greedy", "DSATUR", "BD-CeNN"]
    modes = ["cci", "cci_aci"]

    fig = go.Figure()

    any_data = False
    for mode in modes:
        values = []
        for solver in solvers:
            if (solver in record.metrics
                    and mode in record.metrics[solver]):
                mr = record.metrics[solver][mode]
                val = getattr(mr, metric, None)
                values.append(val if val is not None else 0)
                any_data = True
            else:
                values.append(0)

        # Skip empty trace if all zero and mode never appeared
        mode_present_at_all = any(
            mode in record.metrics.get(s, {}) for s in solvers
        )
        if not mode_present_at_all:
            continue

        text_labels = [
            f"{v:.2f}" if isinstance(v, float) else str(v)
            for v in values
        ]

        fig.add_trace(go.Bar(
            x=solvers,
            y=values,
            name=MODE_LABELS[mode],
            marker_color=MODE_COLORS[mode],
            text=text_labels,
            textposition="outside",
            hovertemplate=f"{MODE_LABELS[mode]}<br>Solveur: %{{x}}<br>{ylabel}: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"{ylabel} par solveur et mode d'interference",
            x=0.5, font=dict(size=13),
        ),
        xaxis_title="Solveur",
        yaxis_title=ylabel,
        barmode="group",
        plot_bgcolor="white",
        height=500,
        legend=dict(x=0.75, y=1.0, bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        transition_duration=500,
    )

    if not any_data:
        fig.add_annotation(
            text="Aucune donnee disponible",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#888888"),
        )

    return fig


# ============================================================================
# DELTA_ACI VISUALIZATION
# ============================================================================

def create_delta_aci_chart(record: ExperimentRecord) -> Optional[go.Figure]:
    """
    Bar chart of Delta_ACI (%) per solver.
    Returns None if neither mode is available for any solver.
    """
    solvers = ["Random", "Greedy", "DSATUR", "BD-CeNN"]
    deltas = []
    valid_solvers = []
    for solver in solvers:
        modes = record.metrics.get(solver, {})
        if "cci" in modes and "cci_aci" in modes:
            cci = modes["cci"].cost
            aci = modes["cci_aci"].cost
            if cci > 0:
                delta = ((aci - cci) / cci) * 100.0
            else:
                delta = 0.0
            deltas.append(delta)
            valid_solvers.append(solver)

    if not valid_solvers:
        return None

    colors = [SOLVER_COLORS.get(s, "#7f7f7f") for s in valid_solvers]

    fig = go.Figure(data=[
        go.Bar(
            x=valid_solvers,
            y=deltas,
            marker_color=colors,
            text=[f"+{d:.1f}%" for d in deltas],
            textposition="outside",
            hovertemplate="%{x}<br>Delta_ACI: %{y:.2f}%<extra></extra>",
        )
    ])

    fig.update_layout(
        title=dict(
            text="Delta_ACI : augmentation relative du cout de CCI-only vers CCI+ACI",
            x=0.5, font=dict(size=13),
        ),
        xaxis_title="Solveur",
        yaxis_title="Delta_ACI (%)",
        plot_bgcolor="white",
        height=450,
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        transition_duration=500,
    )
    return fig


# ============================================================================
# RADAR CHART
# ============================================================================

def create_radar_chart(record: ExperimentRecord, mode: str) -> Optional[go.Figure]:
    """
    Normalized radar chart for a given mode.
    Higher radar value = better performance.
    """
    solvers = ["Random", "Greedy", "DSATUR", "BD-CeNN"]
    metrics_keys = ["cost", "n_conflicts", "wall_time_seconds", "used_channels"]
    metrics_labels = ["Cout", "Conflits", "Temps", "Canaux"]

    matrix = []
    valid_solvers = []
    for solver in solvers:
        modes_dict = record.metrics.get(solver, {})
        if mode not in modes_dict:
            continue
        mr = modes_dict[mode]
        try:
            row = [float(getattr(mr, k, 0) or 0) for k in metrics_keys]
        except Exception:
            continue
        matrix.append(row)
        valid_solvers.append(solver)

    if not matrix:
        return None

    matrix = np.array(matrix, dtype=float)

    normalized = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-9:
            normalized[:, j] = 1.0
        else:
            normalized[:, j] = 1.0 - (col - col_min) / (col_max - col_min)

    fig = go.Figure()
    for i, solver in enumerate(valid_solvers):
        values = normalized[i].tolist()
        values.append(values[0])
        labels_closed = metrics_labels + [metrics_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels_closed,
            fill="toself",
            name=solver,
            line=dict(color=SOLVER_COLORS.get(solver, "#7f7f7f")),
            opacity=0.6,
        ))

    fig.update_layout(
        title=dict(
            text=f"Profil multi-metriques ({MODE_LABELS.get(mode, mode)}) - plus grand = meilleur",
            x=0.5, font=dict(size=13),
        ),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor="white",
        ),
        showlegend=True,
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        transition_duration=500,
    )
    return fig


# ============================================================================
# HEATMAP W
# ============================================================================

def create_weight_matrix_heatmap(W: np.ndarray) -> go.Figure:
    """Interactive heatmap of the interference weight matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=W,
        colorscale="Reds",
        zmin=0,
        zmax=max(4, float(W.max()) if W.size > 0 else 4),
        hovertemplate="Cellule i: %{y}<br>Cellule j: %{x}<br>W[i,j] = %{z}<extra></extra>",
        colorbar=dict(title="Poids"),
    ))
    fig.update_layout(
        title=dict(text="Matrice d'interference W", x=0.5, font=dict(size=13)),
        xaxis_title="Cellule j",
        yaxis_title="Cellule i",
        height=500,
        margin=dict(l=60, r=30, t=60, b=50),
        yaxis=dict(autorange="reversed"),
        transition_duration=300,
    )
    return fig


# ============================================================================
# METRICS DATAFRAME
# ============================================================================

def build_metrics_dataframe(record: ExperimentRecord):
    """
    Flattens metrics into a pandas DataFrame ready for display.
    """
    import pandas as pd
    rows = []
    for solver_name, modes in record.metrics.items():
        for mode, mr in modes.items():
            rows.append({
                "Solveur": solver_name,
                "Mode": MODE_LABELS.get(mode, mode),
                "Cout J(x)": round(mr.cost, 4),
                "Conflits": mr.n_conflicts,
                "Canaux utilises": mr.used_channels,
                "Temps (s)": round(mr.wall_time_seconds, 4),
                "Iterations": mr.n_iterations,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "Solveur", "Mode", "Cout J(x)", "Conflits",
            "Canaux utilises", "Temps (s)", "Iterations"
        ])
    return pd.DataFrame(rows)