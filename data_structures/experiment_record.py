"""
ExperimentRecord and MetricsRecord dataclasses. Represent the complete
outcome of a single experiment run across all solvers and interference modes.
Uses absolute imports to survive Streamlit's module reloading cycles.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
import uuid

# Use absolute imports to prevent Streamlit hot-reloading context issues
from data_structures.topology import NetworkTopology
from data_structures.solver_result import SolverResult
from data_structures.audit_report import AuditReport


@dataclass
class MetricsRecord:
    """
    Compact scalar metrics for a single (solver, interference_mode) pair.
    """
    solver_name: str
    interference_mode: str
    cost: float
    n_conflicts: int
    used_channels: int
    wall_time_seconds: float
    n_iterations: int = 0

    def to_dict(self) -> dict:
        return {
            "solver_name": self.solver_name,
            "interference_mode": self.interference_mode,
            "cost": float(self.cost),
            "n_conflicts": int(self.n_conflicts),
            "used_channels": int(self.used_channels),
            "wall_time_seconds": float(self.wall_time_seconds),
            "n_iterations": int(self.n_iterations),
        }


@dataclass
class ExperimentRecord:
    """
    Full experiment result bundle: topology, all solver outcomes,
    aggregated metrics, and optional LLM audit trail.
    """
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    topology: Optional[NetworkTopology] = None
    solver_results: Dict[str, Dict[str, SolverResult]] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, MetricsRecord]] = field(default_factory=dict)
    delta_aci_percent: Optional[float] = None
    llm_explanation: Optional[str] = None
    audit_report: Optional[AuditReport] = None

    def add_solver_result(self, result: SolverResult) -> None:
        """
        Registers a SolverResult under the correct (solver_name, mode) slot
        and derives its MetricsRecord.
        """
        solver_name = result.solver_name
        mode = result.interference_mode

        if solver_name not in self.solver_results:
            self.solver_results[solver_name] = {}
        self.solver_results[solver_name][mode] = result

        if solver_name not in self.metrics:
            self.metrics[solver_name] = {}
        self.metrics[solver_name][mode] = MetricsRecord(
            solver_name=solver_name,
            interference_mode=mode,
            cost=result.cost,
            n_conflicts=result.n_conflicts,
            used_channels=result.used_channels,
            wall_time_seconds=result.wall_time_seconds,
            n_iterations=result.n_iterations,
        )

    def compute_delta_aci(self, solver_name: str = "BD-CeNN") -> Optional[float]:
        """
        Computes relative cost increase from CCI-only to CCI+ACI models
        for a given solver.
        """
        if solver_name not in self.metrics:
            return None
        modes = self.metrics[solver_name]
        if "cci" not in modes or "cci_aci" not in modes:
            return None
        cost_cci = modes["cci"].cost
        cost_aci = modes["cci_aci"].cost
        if cost_cci <= 0:
            return None
        delta = ((cost_aci - cost_cci) / cost_cci) * 100.0
        self.delta_aci_percent = delta
        return delta

    def to_dict(self) -> dict:
        """Serializes the full record to a JSON-compatible dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "topology": self.topology.to_dict() if self.topology else None,
            "solver_results": {
                solver: {mode: res.to_dict() for mode, res in modes.items()}
                for solver, modes in self.solver_results.items()
            },
            "metrics": {
                solver: {mode: mr.to_dict() for mode, mr in modes.items()}
                for solver, modes in self.metrics.items()
            },
            "delta_aci_percent": self.delta_aci_percent,
            "llm_explanation": self.llm_explanation,
            "audit_report": self.audit_report.to_dict() if self.audit_report else None,
        }