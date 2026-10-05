"""
InterferenceGraph abstraction. Object-oriented wrapper around a NetworkTopology
that exposes energy computations and conflict detection for both CCI-only
and CCI+ACI models.

This module is a thin facade over the certified functions in metrics.py.
It centralizes graph-related operations for the dashboard and solvers.
"""


from typing import List, Tuple, Optional
import numpy as np

from data_structures.topology import NetworkTopology
from metrics import (
    compute_cost_cci,
    compute_cost_cci_aci,
    count_conflicts_cci,
    count_conflicts_aci,
    count_conflicts_total,
    create_channel_interference_matrix,
)


class InterferenceGraph:
   
    def __init__(self, topology: NetworkTopology,
                 aci_decay: float = 0.5, aci_cutoff: int = 2):
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
        Matrice spectrale M (construite et mise en cache à la première demande).

        Returns
        -------
        M : ndarray, shape (K, K)
        """
        if self._M_cache is None:
            self._M_cache = create_channel_interference_matrix(
                self.K, decay=self.aci_decay, cutoff=self.aci_cutoff
            )
        return self._M_cache

    # ------------------------------------------------------------------
    # Fonctions de coût (CCI-only vs CCI+ACI)
    # ------------------------------------------------------------------

    def compute_cost_cci(self, assignment: np.ndarray) -> float:
        """
        Coût global sous le régime CCI-only (Eq. 2).

        Parameters
        ----------
        assignment : ndarray, shape (N,)

        Returns
        -------
        cost : float
        """
        return compute_cost_cci(assignment, self.W)

    def compute_cost_cci_aci(self, assignment: np.ndarray) -> float:
        """
        Coût global sous le régime CCI+ACI (Eq. 3).

        Parameters
        ----------
        assignment : ndarray, shape (N,)

        Returns
        -------
        cost : float
        """
        return compute_cost_cci_aci(assignment, self.W, self.M)

    def compute_cost(self, assignment: np.ndarray, mode: str) -> float:
       
        if mode == "cci":
            return self.compute_cost_cci(assignment)
        if mode == "cci_aci":
            return self.compute_cost_cci_aci(assignment)
        raise ValueError(f"Mode d'interférence inconnu : {mode!r}")

    def compute_delta_aci(self, assignment: np.ndarray) -> float:
        """
        Augmentation relative du coût (%) en passant de CCI-only à CCI+ACI.

        Parameters
        ----------
        assignment : ndarray, shape (N,)

        Returns
        -------
        delta : float
            Pourcentage d'augmentation. 0.0 si le coût CCI est nul.
        """
        cost_cci = self.compute_cost_cci(assignment)
        cost_aci = self.compute_cost_cci_aci(assignment)
        if cost_cci <= 0:
            return 0.0
        return ((cost_aci - cost_cci) / cost_cci) * 100.0

    # ------------------------------------------------------------------
    # Compteurs de conflits détaillés (Eq. 4, 5, 6)
    # ------------------------------------------------------------------

    def count_conflicts_cci(self, assignment: np.ndarray) -> int:
        """
        Nombre de conflits co-canal (Eq. 4).

        Parameters
        ----------
        assignment : ndarray, shape (N,)

        Returns
        -------
        n_cci : int
        """
        return count_conflicts_cci(assignment, self.W)

    def count_conflicts_aci(self, assignment: np.ndarray) -> int:
        """
        Nombre de conflits de canal adjacent (Eq. 5).

        Parameters
        ----------
        assignment : ndarray, shape (N,)

        Returns
        -------
        n_aci : int
        """
        return count_conflicts_aci(assignment, self.W, cutoff=self.aci_cutoff)

    def count_conflicts_total(self, assignment: np.ndarray) -> int:
        
        return count_conflicts_total(assignment, self.W, cutoff=self.aci_cutoff)

    def count_conflicts(self, assignment: np.ndarray, mode: str) -> int:
       
        if mode == "cci":
            return self.count_conflicts_cci(assignment)
        if mode == "cci_aci":
            return self.count_conflicts_total(assignment)
        raise ValueError(f"Mode d'interférence inconnu : {mode!r}")

    def get_all_conflict_counts(self, assignment: np.ndarray, mode: str) -> dict:
        
        n_cci = self.count_conflicts_cci(assignment)
        if mode == "cci":
            return {
                "n_conflicts_cci": n_cci,
                "n_conflicts_aci": 0,
                "n_conflicts_total": n_cci,
            }
        n_aci = self.count_conflicts_aci(assignment)
        return {
            "n_conflicts_cci": n_cci,
            "n_conflicts_aci": n_aci,
            "n_conflicts_total": n_cci + n_aci,
        }

    # ------------------------------------------------------------------
    # Listage des arêtes en conflit (pour visualisation)
    # ------------------------------------------------------------------

    def get_conflict_edges_cci(self, assignment: np.ndarray) -> List[Tuple[int, int]]:
        """
        Liste des arêtes (i, j) en conflit co-canal.

        Parameters
        ----------
        assignment : ndarray, shape (N,)

        Returns
        -------
        edges : list of (int, int)
        """
        edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.W[i, j] > 0 and assignment[i] == assignment[j]:
                    edges.append((i, j))
        return edges

    def get_conflict_edges_aci(self, assignment: np.ndarray) -> List[Tuple[int, int]]:
        
        edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.W[i, j] > 0:
                    diff = abs(int(assignment[i]) - int(assignment[j]))
                    if 0 < diff <= self.aci_cutoff:
                        edges.append((i, j))
        return edges

    def get_edge_status(self, i: int, j: int, assignment: np.ndarray,
                        mode: str) -> str:
       
        if self.W[i, j] == 0:
            return "no_edge"
        ci, cj = int(assignment[i]), int(assignment[j])
        if ci == cj:
            return "cochannel"
        if mode == "cci_aci" and 0 < abs(ci - cj) <= self.aci_cutoff:
            return "adjacent"
        return "clear"

    def all_active_edges(self) -> List[Tuple[int, int, float]]:
       
        edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.W[i, j] > 0:
                    edges.append((i, j, float(self.W[i, j])))
        return edges