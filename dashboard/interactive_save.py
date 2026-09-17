"""
Automatic disk persistence for interactive dashboard sessions.
Distinguishes from batch campaign outputs by using a dedicated folder tree:
    outputs/interactive_runs/{cochannel|adjacent}/
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from data_structures import ExperimentRecord


def _base_dir() -> Path:
    """Returns the root output directory for interactive runs."""
    base = Path(config.BASE_DIR) / "outputs" / "interactive_runs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _mode_dir(mode: str) -> Path:
    """Returns the mode-specific subdirectory."""
    sub = "cochannel" if mode == "cci" else "adjacent"
    path = _base_dir() / sub
    for subfolder in ("csv", "json", "figures", "reports"):
        (path / subfolder).mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_interactive_run(record: ExperimentRecord,
                          save_figures: bool = True,
                          extra_figures: Optional[list] = None) -> dict:
    """
    Persists a complete interactive experiment run to disk.

    Args:
        record: ExperimentRecord returned by ExperimentRunner
        save_figures: If True, generates and saves comparison figures
        extra_figures: List of (name, plotly_figure) tuples to save as HTML

    Returns:
        Dict mapping "type" -> list of saved paths
    """
    ts = _timestamp()
    experiment_id = record.experiment_id
    saved = {"json": [], "csv": [], "figures": []}

    try:
        # --- Full JSON in both mode dirs (contains both modes info) ---
        for mode in ["cci", "cci_aci"]:
            if mode in record.solver_results.get("BD-CeNN", {}):
                mdir = _mode_dir(mode)
                json_path = mdir / "json" / f"experiment_{ts}_{experiment_id}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
                saved["json"].append(str(json_path))

        # --- Per-mode CSV of metrics ---
        for mode in ["cci", "cci_aci"]:
            if mode not in record.metrics.get("BD-CeNN", {}):
                continue
            rows = []
            for solver_name, modes in record.metrics.items():
                if mode in modes:
                    mr = modes[mode]
                    rows.append({
                        "solver": solver_name,
                        "interference_mode": "CCI-only" if mode == "cci" else "CCI+ACI",
                        "cost": mr.cost,
                        "n_conflicts": mr.n_conflicts,
                        "used_channels": mr.used_channels,
                        "wall_time_seconds": mr.wall_time_seconds,
                        "n_iterations": mr.n_iterations,
                    })
            if rows:
                df = pd.DataFrame(rows)
                mdir = _mode_dir(mode)
                csv_path = mdir / "csv" / f"metrics_{ts}_{experiment_id}.csv"
                df.to_csv(csv_path, index=False, float_format="%.6f")
                saved["csv"].append(str(csv_path))

        # --- Convergence CSV (BD-CeNN) per mode ---
        for mode, result in record.solver_results.get("BD-CeNN", {}).items():
            curve = result.energy_curve_forward_filled()
            if not curve:
                continue
            conv_df = pd.DataFrame({
                "iteration": list(range(len(curve))),
                "cost_best_so_far": curve,
            })
            mdir = _mode_dir(mode)
            conv_path = mdir / "csv" / f"convergence_{ts}_{experiment_id}.csv"
            conv_df.to_csv(conv_path, index=False, float_format="%.6f")
            saved["csv"].append(str(conv_path))

        # --- Assignment CSV per (solver, mode) ---
        for solver_name, modes in record.solver_results.items():
            for mode, res in modes.items():
                assign_df = pd.DataFrame({
                    "cell": list(range(len(res.assignment))),
                    "channel": [int(c) for c in res.assignment],
                })
                mdir = _mode_dir(mode)
                fname = f"assignment_{ts}_{solver_name}_{experiment_id}.csv"
                assign_path = mdir / "csv" / fname
                assign_df.to_csv(assign_path, index=False)
                saved["csv"].append(str(assign_path))

        # --- Figures (from chart_factory) ---
        if save_figures:
            try:
                from dashboard.chart_factory import (
                    create_convergence_plot,
                    create_comparison_bar_chart,
                    create_delta_aci_chart,
                )

                # Convergence per mode
                for mode, result in record.solver_results.get("BD-CeNN", {}).items():
                    fig = create_convergence_plot(result, mode)
                    mdir = _mode_dir(mode)
                    fpath = mdir / "figures" / f"convergence_{ts}_{experiment_id}.html"
                    fig.write_html(str(fpath), include_plotlyjs="cdn")
                    saved["figures"].append(str(fpath))

                # Comparisons (saved once in each mode folder for completeness)
                for metric in ["cost", "n_conflicts", "wall_time_seconds"]:
                    fig = create_comparison_bar_chart(record, metric=metric)
                    for mode in record.solver_results.get("BD-CeNN", {}).keys():
                        mdir = _mode_dir(mode)
                        fpath = mdir / "figures" / f"comparison_{metric}_{ts}_{experiment_id}.html"
                        fig.write_html(str(fpath), include_plotlyjs="cdn")
                        saved["figures"].append(str(fpath))

                # Delta ACI
                delta_fig = create_delta_aci_chart(record)
                if delta_fig is not None:
                    for mode in record.solver_results.get("BD-CeNN", {}).keys():
                        mdir = _mode_dir(mode)
                        fpath = mdir / "figures" / f"delta_aci_{ts}_{experiment_id}.html"
                        delta_fig.write_html(str(fpath), include_plotlyjs="cdn")
                        saved["figures"].append(str(fpath))
            except Exception:
                pass

        # --- Extra figures (e.g., graph renders) ---
        if extra_figures:
            for name, fig in extra_figures:
                try:
                    mode = "cci"
                    if "aci" in name.lower():
                        mode = "cci_aci"
                    mdir = _mode_dir(mode)
                    fpath = mdir / "figures" / f"{name}_{ts}_{experiment_id}.html"
                    fig.write_html(str(fpath), include_plotlyjs="cdn")
                    saved["figures"].append(str(fpath))
                except Exception:
                    continue

    except Exception as e:
        saved["error"] = f"{type(e).__name__}: {e}"

    return saved


def get_interactive_outputs_summary() -> dict:
    """Returns a summary of all files in outputs/interactive_runs/."""
    base = _base_dir()
    summary = {"cochannel": {}, "adjacent": {}, "total_files": 0}

    for mode in ["cochannel", "adjacent"]:
        mode_dir = base / mode
        if not mode_dir.exists():
            continue
        for sub in ["csv", "json", "figures", "reports"]:
            sub_dir = mode_dir / sub
            if sub_dir.exists():
                files = list(sub_dir.glob("*"))
                summary[mode][sub] = len(files)
                summary["total_files"] += len(files)

    return summary