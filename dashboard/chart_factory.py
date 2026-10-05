"""
Chart factory. Centralizes creation of all Plotly figures used in the dashboard
(convergence curves, comparison bar charts, delta ACI plots, radar charts).

All figures include transition_duration for smooth animations on data updates.
"""

from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from data_structures.solver_result import SolverResult
from data_structures.experiment_record import ExperimentRecord


# ============================================================================
# CHART COLOR PALETTES
# ============================================================================

SOLVER_COLORS = {
    "Random": "#3498db",
    "Greedy": "#27ae60",
    "DSATUR": "#e67e22",
    "BD-CeNN": "#c0392b",
}

# Couleurs sémantiques par régime (Chantier A)
MODE_COLORS = {
    "cci": "#3b82f6",      # Bleu pour CCI-only
    "cci_aci": "#ef4444",  # Rouge pour CCI+ACI
}

MODE_LABELS = {
    "cci": "CCI-only",
    "cci_aci": "CCI+ACI",
}


# ============================================================================
# 1. COURBES DE CONVERGENCE (BD-CeNN)
# ============================================================================

def create_convergence_plot(bd_result: SolverResult, mode: str) -> go.Figure:
    """
    Génère la trajectoire d'optimisation lissée (best-so-far) en sweeps (Chantier D).

    Parameters
    ----------
    bd_result : SolverResult
        Résultat de l'exécution du solveur BD-CeNN.
    mode : str
        Régime d'interférence ("cci" ou "cci_aci").

    Returns
    -------
    fig : go.Figure
    """
    curve = bd_result.energy_curve_forward_filled() if bd_result else []
    sweeps_axis = list(range(len(curve)))
    color = MODE_COLORS.get(mode, "#666666")

    fig = go.Figure()

    if curve:
        fig.add_trace(go.Scatter(
            x=sweeps_axis,
            y=curve,
            mode="lines+markers",
            line=dict(color=color, width=2.5, shape="hv"),
            marker=dict(size=5),
            name="J(x) best-so-far",
            hovertemplate="Sweep %{x}<br>Cout: %{y:.4f}<extra></extra>",
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
            text=f"Courbe de convergence de l'energie ({MODE_LABELS.get(mode, mode)})",
            x=0.5, font=dict(size=13),
        ),
        xaxis_title="Indice de balayage global (sweeps cumulatifs)",
        yaxis_title="Energie globale J(x)",
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
    Superpose les trajectoires de convergence des deux régimes (CCI vs CCI+ACI)
    sur le même graphique (Chantier D).
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
            hovertemplate=f"{MODE_LABELS[mode]}<br>Sweep %{{x}}<br>Cout %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Convergence BD-CeNN : CCI-only vs CCI+ACI",
            x=0.5, font=dict(size=13),
        ),
        xaxis_title="Indice de balayage global (sweeps)",
        yaxis_title="Energie globale J(x)",
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
# 2. COMPARATEURS HISTOGRAMMES (BAR CHARTS GROUPÉS)
# ============================================================================

def create_comparison_bar_chart(
    record: ExperimentRecord,
    metric: str = "cost",
) -> go.Figure:
    """
    Génère un histogramme groupé comparant les solveurs sur une mètric donnée.

    Chantier A : Prise en compte de conflicts_cci, conflicts_aci et conflicts_total.
    Chantier D : Remplacement de wall_time par n_sweeps.
    """
    metric_labels = {
        "cost": "Cout J(x)",
        "conflicts_cci": "Conflits co-canal C_CCI",
        "conflicts_aci": "Conflits adjacents C_ACI",
        "conflicts_total": "Conflits totaux C_total",
        "used_channels": "Canaux utilises",
        "wall_time_seconds": "Temps de calcul t_exec (s)",
        "n_sweeps": "Nombre de balayages (sweeps)",
    }
    ylabel = metric_labels.get(metric, metric)

    solvers = ["Random", "Greedy", "DSATUR", "BD-CeNN"]
    modes = ["cci", "cci_aci"]

    fig = go.Figure()
    any_data = False

    for mode in modes:
        values = []
        for solver in solvers:
            if solver in record.metrics and mode in record.metrics[solver]:
                mr = record.metrics[solver][mode]
                # Mapping dynamique des clés internes vers les variables SolverResult (Chantier A & D)
                if metric == "conflicts_cci":
                    val = getattr(mr, "n_conflicts_cci", 0)
                elif metric == "conflicts_aci":
                    val = getattr(mr, "n_conflicts_aci", 0)
                elif metric == "conflicts_total":
                    val = getattr(mr, "n_conflicts_total", 0)
                elif metric == "n_sweeps":
                    val = getattr(mr, "n_sweeps", 0)
                else:
                    val = getattr(mr, metric, 0)
                values.append(val if val is not None else 0)
                any_data = True
            else:
                values.append(0)

        # On n'affiche la barre que si le mode d'interférence a produit des métriques
        mode_present = any(mode in record.metrics.get(s, {}) for s in solvers)
        if not mode_present:
            continue

        text_labels = [
            f"{v:.2f}" if isinstance(v, float) and metric in ["cost", "wall_time_seconds"] else str(int(v)) if isinstance(v, (int, float)) else str(v)
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
            text="Aucun resultat disponible pour generer l'analyse comparative.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#888888"),
        )

    return fig


# ============================================================================
# 3. COMPARAISON DE SENSIBILITÉ SPECTRE (DELTA_ACI)
# ============================================================================

def create_delta_aci_chart(record: ExperimentRecord) -> Optional[go.Figure]:
    """
    Génère la métrique Delta_ACI (%) de surcoût entre les deux régimes.
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
            text="Delta_ACI : degradation du cout en regime adjacent",
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
# 4. DIAGRAMME RADAR (MULTI-METRIC PROFILES)
# ============================================================================

def create_radar_chart(record: ExperimentRecord, mode: str) -> Optional[go.Figure]:
    """
    Rend le radar multi-métriques normalisé pour un régime donné.

    Chantiers rattachés :
        - Chantier A : Inclusion des compteurs séparés (C_CCI, C_ACI).
        - Chantier D : Intégration de n_sweeps à la place des itérations.
    """
    solvers = ["Random", "Greedy", "DSATUR", "BD-CeNN"]
    
    # Axes du radar (Chantiers A & D)
    metrics_keys = ["cost", "n_conflicts_cci", "n_conflicts_aci", "wall_time_seconds", "n_sweeps"]
    metrics_labels = ["Cout", "Conflits CCI", "Conflits ACI", "Temps calcul", "Sweeps"]

    matrix = []
    valid_solvers = []
    for solver in solvers:
        modes_dict = record.metrics.get(solver, {})
        if mode not in modes_dict:
            continue
        mr = modes_dict[mode]
        try:
            row = [
                float(getattr(mr, "cost", 0) or 0),
                float(getattr(mr, "n_conflicts_cci", 0) or 0),
                float(getattr(mr, "n_conflicts_aci", 0) or 0),
                float(getattr(mr, "wall_time_seconds", 0) or 0),
                float(getattr(mr, "n_sweeps", 0) or 0)
            ]
        except Exception:
            continue
        matrix.append(row)
        valid_solvers.append(solver)

    if not matrix:
        return None

    matrix = np.array(matrix, dtype=float)

    # Normalisation : convertit en échelle d'efficacité [0, 1] où 1 est le meilleur
    normalized = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-9:
            normalized[:, j] = 1.0
        else:
            # Toutes ces métriques sont à minimiser -> on inverse
            normalized[:, j] = 1.0 - (col - col_min) / (col_max - col_min)

    fig = go.Figure()
    for i, solver in enumerate(valid_solvers):
        values = normalized[i].tolist()
        values.append(values[0])  # Fermeture du polygone
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
            text=f"Empreinte comportementale ({MODE_LABELS.get(mode, mode)})",
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
# 5. HEATMAP DE COUPLAGE GÉOGRAPHIQUE W
# ============================================================================

def create_weight_matrix_heatmap(W: np.ndarray) -> go.Figure:
    """Rend la carte thermique d'intensité W."""
    fig = go.Figure(data=go.Heatmap(
        z=W,
        colorscale="Reds",
        zmin=0,
        zmax=max(4, float(W.max()) if W.size > 0 else 4),
        hovertemplate="Cellule i: %{y}<br>Cellule j: %{x}<br>W_ij = %{z}<extra></extra>",
        colorbar=dict(title="Poids"),
    ))
    fig.update_layout(
        title=dict(text="Intensite des couplages geographiques W_ij", x=0.5, font=dict(size=13)),
        xaxis_title="Cellule j",
        yaxis_title="Cellule i",
        height=500,
        margin=dict(l=60, r=30, t=60, b=50),
        yaxis=dict(autorange="reversed"),
        transition_duration=300,
    )
    return fig


def create_channel_matrix_heatmap(M: np.ndarray) -> go.Figure:
    """
    Rend la carte thermique de la matrice spectrale M (interference entre canaux).

    Parameters
    ----------
    M : ndarray, shape (K, K)
        Matrice d'interference inter-canaux construite par
        create_channel_interference_matrix.

    Returns
    -------
    fig : go.Figure
    """
    K = M.shape[0]
    fig = go.Figure(data=go.Heatmap(
        z=M,
        x=[f"Canal {j}" for j in range(K)],
        y=[f"Canal {i}" for i in range(K)],
        colorscale="Blues",
        zmin=0.0,
        zmax=1.0,
        text=[[f"{M[i, j]:.2f}" for j in range(K)] for i in range(K)],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Canal i : %{y}<br>Canal j : %{x}<br>M_ij = %{z:.2f}<extra></extra>",
        colorbar=dict(title="Attenuation"),
    ))
    fig.update_layout(
        title=dict(
            text="Matrice d'interference inter-canaux M<br>"
                 "<sub>alpha = 0,5, cutoff = 2</sub>",
            x=0.5, font=dict(size=13),
        ),
        xaxis_title="Canal j",
        yaxis_title="Canal i",
        height=500,
        margin=dict(l=70, r=30, t=90, b=60),
        yaxis=dict(autorange="reversed"),
        transition_duration=300,
    )
    return fig
# ============================================================================
# 6. EXPORT DE DATAFRAME FLATTENED (Chantiers A + D)
# ============================================================================

def build_metrics_dataframe(record: ExperimentRecord):
    """
    Rend la table condensée pour l'onglet de comparaison.
    """
    rows = []
    for solver_name, modes in record.metrics.items():
        for mode, mr in modes.items():
            rows.append({
                "Solveur": solver_name,
                "Mode": MODE_LABELS.get(mode, mode),
                "Cout J(x)": round(mr.cost, 4),
                "Conflits CCI (C_CCI)": int(mr.n_conflicts_cci),
                "Conflits ACI (C_ACI)": int(mr.n_conflicts_aci),
                "Conflits Totaux (C_total)": int(mr.n_conflicts_total),
                "Canaux utilises": int(mr.used_channels),
                "Temps t_exec (s)": round(mr.wall_time_seconds, 4),
                "Sweeps de convergence": int(mr.n_sweeps) if mr.n_sweeps > 0 else "N/A",
            })
    if not rows:
        return pd.DataFrame(columns=[
            "Solveur", "Mode", "Cout J(x)", "Conflits CCI (C_CCI)",
            "Conflits ACI (C_ACI)", "Conflits Totaux (C_total)",
            "Canaux utilises", "Temps t_exec (s)", "Sweeps de convergence"
        ])
    return pd.DataFrame(rows)