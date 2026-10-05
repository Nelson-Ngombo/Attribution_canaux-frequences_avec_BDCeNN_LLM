"""
ExperimentRecord and MetricsRecord dataclasses. Represent the complete
outcome of a single experiment run across all solvers and interference modes.
Uses absolute imports to survive Streamlit's module reloading cycles.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
import uuid

from data_structures.topology import NetworkTopology
from data_structures.solver_result import SolverResult
from data_structures.audit_report import AuditReport


@dataclass
class MetricsRecord:
    """
    Indicateurs condensés d'exécution pour un couple (solveur, régime).
    """
    solver_name: str
    interference_mode: str
    cost: float
    n_conflicts_cci: int
    n_conflicts_aci: int
    n_conflicts_total: int
    used_channels: int
    wall_time_seconds: float
    n_sweeps: int = 0

    def to_dict(self) -> dict:
        return {
            "solver_name": self.solver_name,
            "interference_mode": self.interference_mode,
            "cost": float(self.cost),
            "n_conflicts_cci": int(self.n_conflicts_cci),
            "n_conflicts_aci": int(self.n_conflicts_aci),
            "n_conflicts_total": int(self.n_conflicts_total),
            "used_channels": int(self.used_channels),
            "wall_time_seconds": float(self.wall_time_seconds),
            "n_sweeps": int(self.n_sweeps),
        }


@dataclass
class ExperimentRecord:
    """
    Enregistrement maître d'une expérience interactive ou batch.
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
        Inscrit un SolverResult dans le dictionnaire et dérive son MetricsRecord.
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
            n_conflicts_cci=result.n_conflicts_cci,
            n_conflicts_aci=result.n_conflicts_aci,
            n_conflicts_total=result.n_conflicts_total,
            used_channels=result.used_channels,
            wall_time_seconds=result.wall_time_seconds,
            n_sweeps=result.n_sweeps,
        )

    def compute_delta_aci(self, solver_name: str = "BD-CeNN") -> Optional[float]:
        """
        Calcule le surcoût relatif Delta_ACI entre les deux régimes d'interférence.
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
        """Sérialise en dictionnaire JSON."""
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