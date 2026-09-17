"""
Export manager. Centralizes serialization of experiment results to disk
in multiple formats (CSV, JSON, PNG) with strict CCI vs CCI+ACI separation.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import io
import zipfile

import numpy as np
import pandas as pd

import config
from data_structures import ExperimentRecord


# ---------------------------------------------------------------------------
# Directory helpers with strict CCI / CCI+ACI separation
# ---------------------------------------------------------------------------

def get_output_dirs(interference_mode: str) -> dict:
    """
    Returns the standard output directory tree for a given interference mode.
    Ensures physical separation of co-channel and adjacent results on disk.
    """
    sub = "cochannel" if interference_mode == "cci" else "adjacent"
    base = config.RESULTS_DIR / "dashboard_runs" / sub
    dirs = {
        "base": base,
        "csv": base / "csv",
        "json": base / "json",
        "figures": base / "figures",
        "reports": base / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_record_json(record: ExperimentRecord,
                        output_path: Optional[Path] = None) -> Path:
    """
    Serializes the full ExperimentRecord to a JSON file.
    If output_path is None, a default location is derived from the record ID.
    """
    if output_path is None:
        primary_mode = "cci" if "cci" in record.metrics.get("BD-CeNN", {}) else "cci_aci"
        dirs = get_output_dirs(primary_mode)
        output_path = dirs["json"] / f"experiment_{record.experiment_id}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

    return output_path


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _build_metrics_dataframe(record: ExperimentRecord) -> pd.DataFrame:
    """
    Flattens the nested metrics dictionary into a tidy DataFrame with one
    row per (solver, interference_mode) pair.
    """
    rows = []
    for solver_name, modes in record.metrics.items():
        for mode, mr in modes.items():
            rows.append({
                "experiment_id": record.experiment_id,
                "timestamp": record.timestamp,
                "solver": solver_name,
                "interference_mode": "CCI-only" if mode == "cci" else "CCI+ACI",
                "cost": mr.cost,
                "n_conflicts": mr.n_conflicts,
                "used_channels": mr.used_channels,
                "wall_time_seconds": mr.wall_time_seconds,
                "n_iterations": mr.n_iterations,
            })
    return pd.DataFrame(rows)


def export_metrics_csv(record: ExperimentRecord,
                        output_path: Optional[Path] = None) -> Path:
    """
    Exports the flattened metrics table to CSV.
    """
    df = _build_metrics_dataframe(record)

    if output_path is None:
        primary_mode = "cci" if "cci" in record.metrics.get("BD-CeNN", {}) else "cci_aci"
        dirs = get_output_dirs(primary_mode)
        output_path = dirs["csv"] / f"metrics_{record.experiment_id}.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    return output_path


def export_metrics_split_csv(record: ExperimentRecord) -> dict:
    """
    Exports metrics into TWO separate CSV files:
    one for CCI-only and one for CCI+ACI. Returns dict {mode: path}.
    """
    df = _build_metrics_dataframe(record)
    paths = {}

    for mode_label, mode_key in [("CCI-only", "cci"), ("CCI+ACI", "cci_aci")]:
        sub_df = df[df["interference_mode"] == mode_label]
        if sub_df.empty:
            continue
        dirs = get_output_dirs(mode_key)
        path = dirs["csv"] / f"metrics_{record.experiment_id}_{mode_key}.csv"
        sub_df.to_csv(path, index=False, float_format="%.6f")
        paths[mode_key] = path

    return paths


# ---------------------------------------------------------------------------
# Convergence history export
# ---------------------------------------------------------------------------

def export_convergence_csv(record: ExperimentRecord,
                            output_path: Optional[Path] = None) -> Optional[Path]:
    """
    Exports the BD-CeNN convergence trajectories (both modes if available)
    into a single CSV suitable for plotting.
    """
    rows = []
    bd_results = record.solver_results.get("BD-CeNN", {})
    for mode, result in bd_results.items():
        curve = result.energy_curve_forward_filled()
        for i, cost in enumerate(curve):
            rows.append({
                "experiment_id": record.experiment_id,
                "interference_mode": "CCI-only" if mode == "cci" else "CCI+ACI",
                "iteration": i,
                "cost_best_so_far": cost,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if output_path is None:
        dirs = get_output_dirs("cci")
        output_path = dirs["csv"] / f"convergence_{record.experiment_id}.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    return output_path


# ---------------------------------------------------------------------------
# Figure export (Plotly)
# ---------------------------------------------------------------------------

def export_figure_png(fig, output_path: Path,
                      width: int = 1200, height: int = 800,
                      scale: int = 2) -> Optional[Path]:
    """
    Exports a Plotly figure to PNG.
    Requires 'kaleido' to be installed; silently returns None if unavailable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(output_path), width=width, height=height, scale=scale)
        return output_path
    except Exception as e:
        print(f"[EXPORT_WARNING] PNG export failed ({e}). Install kaleido: pip install kaleido")
        return None


def export_figure_html(fig, output_path: Path) -> Path:
    """
    Exports a Plotly figure to interactive HTML (always available).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


# ---------------------------------------------------------------------------
# Bundled ZIP export
# ---------------------------------------------------------------------------

def build_zip_bundle(record: ExperimentRecord,
                     figures: Optional[List] = None,
                     include_llm_report: bool = True) -> bytes:
    """
    Packages all exportable artifacts into an in-memory ZIP archive.
    Returns raw bytes for Streamlit download button.

    Args:
        record: ExperimentRecord to export
        figures: List of (filename, plotly_figure) tuples to include as HTML
        include_llm_report: Whether to embed the LLM explanation as .txt
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Full JSON record
        record_json = json.dumps(record.to_dict(), indent=2, ensure_ascii=False)
        zf.writestr(f"experiment_{record.experiment_id}.json", record_json)

        # 2. Metrics CSV
        df = _build_metrics_dataframe(record)
        zf.writestr(f"metrics_{record.experiment_id}.csv", df.to_csv(index=False))

        # 3. Split metrics CSV per mode
        for mode_label, mode_key in [("CCI-only", "cci"), ("CCI+ACI", "cci_aci")]:
            sub_df = df[df["interference_mode"] == mode_label]
            if not sub_df.empty:
                zf.writestr(f"metrics_{mode_key}.csv", sub_df.to_csv(index=False))

        # 4. Convergence CSV
        conv_rows = []
        for mode, result in record.solver_results.get("BD-CeNN", {}).items():
            curve = result.energy_curve_forward_filled()
            for i, cost in enumerate(curve):
                conv_rows.append({
                    "interference_mode": "CCI-only" if mode == "cci" else "CCI+ACI",
                    "iteration": i,
                    "cost_best_so_far": cost,
                })
        if conv_rows:
            conv_df = pd.DataFrame(conv_rows)
            zf.writestr(f"convergence_{record.experiment_id}.csv", conv_df.to_csv(index=False))

        # 5. LLM report
        if include_llm_report and record.llm_explanation:
            zf.writestr(f"llm_report_{record.experiment_id}.txt", record.llm_explanation)

        # 6. Audit report
        if record.audit_report is not None:
            audit_json = json.dumps(record.audit_report.to_dict(), indent=2, ensure_ascii=False)
            zf.writestr(f"audit_{record.experiment_id}.json", audit_json)

        # 7. Figures as HTML
        if figures:
            for fname, fig in figures:
                try:
                    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
                    zf.writestr(f"figures/{fname}.html", html_str)
                except Exception:
                    continue

    buffer.seek(0)
    return buffer.getvalue()