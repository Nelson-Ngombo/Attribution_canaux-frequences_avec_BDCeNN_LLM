"""
SolverResult dataclass. Uniform contract returned by any solver
(BD-CeNN, Random, Greedy, DSATUR) after execution.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class SolverResult:
    """
    Structured outcome of a single solver invocation.

    Attributes:
        solver_name: Human-readable identifier ("BD-CeNN", "Random", ...)
        assignment: Channel assignment vector, shape (N,), integer dtype
        cost: Final scalar objective value (interpretation depends on mode)
        interference_mode: "cci" for co-channel only, "cci_aci" for combined
        n_conflicts: Number of active conflicting edges
        used_channels: Count of distinct channels effectively used
        wall_time_seconds: Total wall-clock runtime
        n_iterations: Total iterations performed (0 for non-iterative solvers)
        best_iteration: Iteration index at which the best cost was reached
        energy_history: Convergence trajectory as list of (iteration, cost)
        initial_assignment: Optional initial channel vector (for BD-CeNN)
        metadata: Arbitrary solver-specific auxiliary information
    """
    solver_name: str
    assignment: np.ndarray
    cost: float
    interference_mode: str
    n_conflicts: int
    used_channels: int
    wall_time_seconds: float
    n_iterations: int = 0
    best_iteration: int = 0
    energy_history: List = field(default_factory=list)
    initial_assignment: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def energy_curve_forward_filled(self) -> List[float]:
        """
        Returns the monotone non-increasing best-so-far cost curve
        for smooth visualization in the dashboard.
        """
        if not self.energy_history:
            return []
        costs = [entry[1] if isinstance(entry, (list, tuple)) else entry
                 for entry in self.energy_history]
        best_so_far = costs[0]
        filled = [best_so_far]
        for c in costs[1:]:
            best_so_far = min(best_so_far, c)
            filled.append(best_so_far)
        return filled

    def to_dict(self) -> dict:
        """Serializes result to a JSON-compatible dictionary."""
        history_serialized = []
        for entry in self.energy_history:
            if isinstance(entry, (list, tuple)):
                # Only keep (iteration, cost), drop any array
                history_serialized.append([int(entry[0]), float(entry[1])])
            else:
                history_serialized.append(float(entry))

        return {
            "solver_name": self.solver_name,
            "assignment": [int(c) for c in self.assignment],
            "cost": float(self.cost),
            "interference_mode": self.interference_mode,
            "n_conflicts": int(self.n_conflicts),
            "used_channels": int(self.used_channels),
            "wall_time_seconds": float(self.wall_time_seconds),
            "n_iterations": int(self.n_iterations),
            "best_iteration": int(self.best_iteration),
            "energy_history": history_serialized,
            "initial_assignment": (
                [int(c) for c in self.initial_assignment]
                if self.initial_assignment is not None else None
            ),
            "metadata": self.metadata,
        }