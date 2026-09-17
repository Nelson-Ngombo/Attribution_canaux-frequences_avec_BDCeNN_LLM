"""
NetworkTopology dataclass. Encapsulates a full cellular network instance
with positions, weight matrix, and generation metadata.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class NetworkTopology:
    """
    Immutable representation of a synthetic cellular network topology.

    Attributes:
        n_cells: Number of cellular antennas (N)
        n_channels: Number of available frequency channels (K)
        positions: Cell 2D coordinates, shape (N, 2)
        weight_matrix: Spatial interference weights W, shape (N, N)
        threshold: Distance threshold used during generation
        area: Bounding square side length for cell placement
        seed: Random seed used for reproducibility
        scenario_name: Optional scenario identifier (e.g., "S3")
    """
    n_cells: int
    n_channels: int
    positions: np.ndarray
    weight_matrix: np.ndarray
    threshold: float
    area: float
    seed: int
    scenario_name: Optional[str] = None

    @property
    def N(self) -> int:
        """Alias for compatibility with legacy code."""
        return self.n_cells

    @property
    def K(self) -> int:
        """Alias for compatibility with legacy code."""
        return self.n_channels

    @property
    def W(self) -> np.ndarray:
        """Alias for compatibility with legacy code."""
        return self.weight_matrix

    @property
    def n_edges(self) -> int:
        """Total number of active interference edges."""
        return int(np.sum(self.weight_matrix > 0) // 2)

    @property
    def density(self) -> float:
        """Edge density ratio."""
        max_edges = self.n_cells * (self.n_cells - 1) / 2
        return self.n_edges / max_edges if max_edges > 0 else 0.0

    def to_dict(self) -> dict:
        """Serializes topology to a JSON-compatible dictionary."""
        return {
            "n_cells": int(self.n_cells),
            "n_channels": int(self.n_channels),
            "threshold": float(self.threshold),
            "area": float(self.area),
            "seed": int(self.seed),
            "scenario_name": self.scenario_name,
            "positions": self.positions.tolist(),
            "weight_matrix": self.weight_matrix.tolist(),
            "n_edges": self.n_edges,
            "density": self.density,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NetworkTopology":
        """Reconstructs topology from a serialized dictionary."""
        return cls(
            n_cells=int(data["n_cells"]),
            n_channels=int(data["n_channels"]),
            positions=np.asarray(data["positions"], dtype=float),
            weight_matrix=np.asarray(data["weight_matrix"], dtype=float),
            threshold=float(data["threshold"]),
            area=float(data["area"]),
            seed=int(data["seed"]),
            scenario_name=data.get("scenario_name"),
        )