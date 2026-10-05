"""
SolverResult dataclass. Uniform contract returned by any solver
(BD-CeNN, Random, Greedy, DSATUR) after execution.
"""


from dataclasses import dataclass, field
from typing import List, Optional
import warnings
import numpy as np


@dataclass
class SolverResult:
    solver_name: str
    assignment: np.ndarray
    cost: float
    interference_mode: str
    n_conflicts_cci: int
    n_conflicts_aci: int
    n_conflicts_total: int
    used_channels: int
    wall_time_seconds: float
    n_sweeps: int = 0
    best_sweep_index: int = 0
    energy_history: List = field(default_factory=list)
    initial_assignment: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Alias de rétrocompatibilité (dépréciés, Chantiers A + D)
    # ------------------------------------------------------------------

    @property
    def n_conflicts(self) -> int:
        """
        Déprécié. Utiliser n_conflicts_total, n_conflicts_cci
        ou n_conflicts_aci selon le besoin.
        """
        warnings.warn(
            "SolverResult.n_conflicts est déprécié. "
            "Utiliser n_conflicts_total, n_conflicts_cci ou n_conflicts_aci.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_conflicts_total

    @property
    def n_iterations(self) -> int:
        """
        Déprécié. Utiliser n_sweeps.
        """
        warnings.warn(
            "SolverResult.n_iterations est déprécié. Utiliser n_sweeps.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_sweeps

    @property
    def best_iteration(self) -> int:
        """
        Déprécié. Utiliser best_sweep_index.
        """
        warnings.warn(
            "SolverResult.best_iteration est déprécié. Utiliser best_sweep_index.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.best_sweep_index

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------

    def energy_curve_forward_filled(self) -> List[float]:
        """
        Retourne la courbe de coût monotone non-croissante (best-so-far)
        pour visualisation lissée.

        Returns
        -------
        filled : list of float
            Courbe forward-filled de longueur égale à energy_history.
        """
        if not self.energy_history:
            return []
        costs = [
            entry[1] if isinstance(entry, (list, tuple)) else entry
            for entry in self.energy_history
        ]
        best_so_far = costs[0]
        filled = [best_so_far]
        for c in costs[1:]:
            best_so_far = min(best_so_far, c)
            filled.append(best_so_far)
        return filled

    def to_dict(self) -> dict:
        """
        Sérialise le résultat en dictionnaire compatible JSON.

        Returns
        -------
        d : dict
            Dictionnaire contenant tous les champs sérialisables.
        """
        history_serialized = []
        for entry in self.energy_history:
            if isinstance(entry, (list, tuple)):
                history_serialized.append([int(entry[0]), float(entry[1])])
            else:
                history_serialized.append(float(entry))

        return {
            "solver_name": self.solver_name,
            "assignment": [int(c) for c in self.assignment],
            "cost": float(self.cost),
            "interference_mode": self.interference_mode,
            "n_conflicts_cci": int(self.n_conflicts_cci),
            "n_conflicts_aci": int(self.n_conflicts_aci),
            "n_conflicts_total": int(self.n_conflicts_total),
            "used_channels": int(self.used_channels),
            "wall_time_seconds": float(self.wall_time_seconds),
            "n_sweeps": int(self.n_sweeps),
            "best_sweep_index": int(self.best_sweep_index),
            "energy_history": history_serialized,
            "initial_assignment": (
                [int(c) for c in self.initial_assignment]
                if self.initial_assignment is not None else None
            ),
            "metadata": self.metadata,
        }