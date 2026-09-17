"""
InterferenceGraph abstraction. Object-oriented wrapper around a NetworkTopology
that exposes energy computations and conflict detection for both CCI-only
and CCI+ACI models.

This module is a thin facade over the certified functions in metrics.py.
It centralizes graph-related operations for the dashboard and solvers.
"""

from typing import List, Tuple, Optional
import numpy as np

from data_structures import NetworkTopology
from metrics import (
    compute_cochannel_cost,
    count_cochannel_conflicts,
    compute_adjacent_cost,
    count_adjacent_conflicts,
    create_channel_interference_matrix,
)


class InterferenceGraph:
    """
    Object-oriented facade around a NetworkTopology.
    Provides cost functions, conflict listings, and local energy operators
    for both the CCI-only model and the CCI+ACI (adjacent channel) model.
    """

    def __init__(self, topology: NetworkTopology,
                 aci_decay: float = 0.5, aci_cutoff: int = 2):
        """
        Args:
            topology: NetworkTopology instance
            aci_decay: Adjacent-channel decay factor (used when M is built)
            aci_cutoff: Maximum channel distance affected by ACI leakage
        """
        self.topology = topology
        self.N = topology.n_cells
        self.K = topology.n_channels
        self.W = topology.weight_matrix
        self.aci_decay = aci_decay
        self.aci_cutoff = aci_cutoff
        self._M_cache: Optional[np.ndarray] = None

    @property
    def M(self) -> np.ndarray:
        """
        Lazily builds and caches the channel adjacency penalty matrix M.
        """
        if self._M_cache is None:
            self._M_cache = create_channel_interference_matrix(
                self.K, decay=self.aci_decay, cutoff=self.aci_cutoff
            )
        return self._M_cache

    # ------------------------------------------------------------------
    # Cost functions (CCI-only vs CCI+ACI)
    # ------------------------------------------------------------------

    def compute_cost_cci(self, assignment: np.ndarray) -> float:
        """Total cost under strict co-channel interference model."""
        return compute_cochannel_cost(assignment, self.W)

    def compute_cost_cci_aci(self, assignment: np.ndarray) -> float:
        """Total cost under combined co-channel and adjacent-channel model."""
        return compute_adjacent_cost(assignment, self.W, self.M)

    def compute_cost(self, assignment: np.ndarray, mode: str) -> float:
        """
        Dispatches to the appropriate cost function based on interference mode.

        Args:
            assignment: Channel assignment vector
            mode: Either "cci" or "cci_aci"
        """
        if mode == "cci":
            return self.compute_cost_cci(assignment)
        if mode == "cci_aci":
            return self.compute_cost_cci_aci(assignment)
        raise ValueError(f"Unknown interference mode: {mode!r}")

    def compute_delta_aci(self, assignment: np.ndarray) -> float:
        """
        Returns the relative cost increase (%) when switching from CCI-only
        to CCI+ACI evaluation for the same assignment.
        """
        cost_cci = self.compute_cost_cci(assignment)
        cost_aci = self.compute_cost_cci_aci(assignment)
        if cost_cci <= 0:
            return 0.0
        return ((cost_aci - cost_cci) / cost_cci) * 100.0

    # ------------------------------------------------------------------
    # Conflict counting
    # ------------------------------------------------------------------

    def count_conflicts_cci(self, assignment: np.ndarray) -> int:
        """Number of co-channel conflicting edges."""
        return count_cochannel_conflicts(assignment, self.W)

    def count_conflicts_cci_aci(self, assignment: np.ndarray) -> int:
        """Number of edges with any active interference (co or adjacent)."""
        return count_adjacent_conflicts(assignment, self.W, self.M)

    def count_conflicts(self, assignment: np.ndarray, mode: str) -> int:
        if mode == "cci":
            return self.count_conflicts_cci(assignment)
        if mode == "cci_aci":
            return self.count_conflicts_cci_aci(assignment)
        raise ValueError(f"Unknown interference mode: {mode!r}")

    # ------------------------------------------------------------------
    # Conflict edge listings (for dashboard visualization)
    # ------------------------------------------------------------------

    def get_conflict_edges_cci(self, assignment: np.ndarray) -> List[Tuple[int, int]]:
        """Returns list of (i, j) edges with co-channel conflict."""
        edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.W[i, j] > 0 and assignment[i] == assignment[j]:
                    edges.append((i, j))
        return edges

    def get_conflict_edges_adjacent(self, assignment: np.ndarray) -> List[Tuple[int, int]]:
        """
        Returns list of (i, j) edges with adjacent-channel conflict
        (excluding pure co-channel matches).
        """
        edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if (self.W[i, j] > 0
                        and assignment[i] != assignment[j]
                        and abs(int(assignment[i]) - int(assignment[j])) <= self.aci_cutoff):
                    edges.append((i, j))
        return edges

    def get_edge_status(self, i: int, j: int, assignment: np.ndarray,
                        mode: str) -> str:
        """
        Classifies a single edge (i, j) as one of:
            "no_edge"  : W[i, j] == 0
            "cochannel": same channel used
            "adjacent" : contiguous channels (only relevant in cci_aci mode)
            "clear"    : no active interference
        """
        if self.W[i, j] == 0:
            return "no_edge"
        ci, cj = int(assignment[i]), int(assignment[j])
        if ci == cj:
            return "cochannel"
        if mode == "cci_aci" and abs(ci - cj) <= self.aci_cutoff:
            return "adjacent"
        return "clear"

    def all_active_edges(self) -> List[Tuple[int, int, float]]:
        """Returns list of (i, j, weight) for all edges with W > 0."""
        edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.W[i, j] > 0:
                    edges.append((i, j, float(self.W[i, j])))
        return edges