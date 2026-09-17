"""
Unit tests for the graph_model module.
Validates the InterferenceGraph facade over metrics.py.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_structures import NetworkTopology
from graph_model import InterferenceGraph


def _make_simple_topology(N=6, K=3, seed=42) -> NetworkTopology:
    """Helper to create a minimal test topology."""
    np.random.seed(seed)
    positions = np.random.rand(N, 2) * 100
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 50:
                W[i, j] = 2.0
                W[j, i] = 2.0
    return NetworkTopology(
        n_cells=N,
        n_channels=K,
        positions=positions,
        weight_matrix=W,
        threshold=50,
        area=100,
        seed=seed,
        scenario_name="TEST",
    )


class TestInterferenceGraph:
    """Tests for the InterferenceGraph class."""

    def test_initialization(self):
        topo = _make_simple_topology()
        graph = InterferenceGraph(topo)
        assert graph.N == 6
        assert graph.K == 3

    def test_cost_cci_zero_for_distinct_channels(self):
        topo = _make_simple_topology(N=4, K=4)
        graph = InterferenceGraph(topo)
        x = np.array([0, 1, 2, 3])
        assert graph.compute_cost_cci(x) == 0.0

    def test_cost_dispatch(self):
        topo = _make_simple_topology()
        graph = InterferenceGraph(topo)
        x = np.array([0, 1, 0, 1, 0, 1])
        cost_cci = graph.compute_cost(x, "cci")
        cost_aci = graph.compute_cost(x, "cci_aci")
        assert cost_aci >= cost_cci

    def test_invalid_mode_raises(self):
        topo = _make_simple_topology()
        graph = InterferenceGraph(topo)
        x = np.zeros(6, dtype=int)
        with pytest.raises(ValueError):
            graph.compute_cost(x, "invalid_mode")

    def test_delta_aci_non_negative(self):
        topo = _make_simple_topology()
        graph = InterferenceGraph(topo)
        x = np.array([0, 1, 2, 0, 1, 2])
        delta = graph.compute_delta_aci(x)
        assert delta >= 0.0

    def test_conflict_edges_cci(self):
        topo = _make_simple_topology(N=3, K=2)
        # Force a fully connected topology
        topo.weight_matrix = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        graph = InterferenceGraph(topo)
        x = np.array([0, 0, 1])
        edges = graph.get_conflict_edges_cci(x)
        assert (0, 1) in edges
        assert len(edges) == 1

    def test_edge_status_classification(self):
        topo = _make_simple_topology(N=3, K=4)
        topo.weight_matrix = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        graph = InterferenceGraph(topo)
        x = np.array([0, 0, 2])
        assert graph.get_edge_status(0, 1, x, "cci") == "cochannel"
        assert graph.get_edge_status(0, 2, x, "cci") == "clear"
        assert graph.get_edge_status(1, 2, x, "cci_aci") == "adjacent"

    def test_all_active_edges(self):
        topo = _make_simple_topology(N=3, K=2)
        topo.weight_matrix = np.array([
            [0, 2, 0],
            [2, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        graph = InterferenceGraph(topo)
        edges = graph.all_active_edges()
        assert len(edges) == 2
        weights = {e[2] for e in edges}
        assert weights == {1.0, 2.0}

    def test_m_matrix_cached(self):
        topo = _make_simple_topology()
        graph = InterferenceGraph(topo)
        M1 = graph.M
        M2 = graph.M
        assert M1 is M2  # same object (cached)