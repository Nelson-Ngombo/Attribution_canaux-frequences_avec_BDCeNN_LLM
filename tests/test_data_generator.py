"""
Unit tests for the data_generator module.
Validates topology generation determinism, weight matrix properties,
and scenario reproducibility.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_generator import generate_network


class TestGenerateNetwork:
    """Tests for the generate_network function."""

    def test_output_shapes(self):
        """W must be N x N, positions must be N x 2."""
        N, K, area, threshold, seed = 10, 4, 100, 30, 42
        W, positions, G = generate_network(N, K, area, threshold, seed)
        assert W.shape == (N, N)
        assert positions.shape == (N, 2)

    def test_weight_matrix_symmetry(self):
        """W must be symmetric: W[i, j] == W[j, i]."""
        W, _, _ = generate_network(15, 4, 100, 30, seed=7)
        np.testing.assert_array_equal(W, W.T)

    def test_weight_matrix_zero_diagonal(self):
        """Diagonal of W must be zero (no self-interference)."""
        W, _, _ = generate_network(20, 6, 150, 40, seed=3)
        np.testing.assert_array_equal(np.diag(W), 0)

    def test_weight_values_in_valid_range(self):
        """All weights must be in {0, 1, 2, 4}."""
        W, _, _ = generate_network(25, 4, 100, 35, seed=11)
        unique_values = set(np.unique(W))
        assert unique_values.issubset({0.0, 1.0, 2.0, 4.0})

    def test_positions_within_area(self):
        """All coordinates must be in [0, area)."""
        area = 200
        _, positions, _ = generate_network(30, 4, area, 50, seed=5)
        assert np.all(positions >= 0)
        assert np.all(positions < area)

    def test_deterministic_reproducibility(self):
        """Same seed must produce identical W and positions."""
        args = (20, 6, 150, 40, 99)
        W1, pos1, _ = generate_network(*args)
        W2, pos2, _ = generate_network(*args)
        np.testing.assert_array_equal(W1, W2)
        np.testing.assert_array_equal(pos1, pos2)

    def test_different_seeds_produce_different_topologies(self):
        """Different seeds should (almost certainly) produce different W."""
        W1, _, _ = generate_network(20, 4, 100, 30, seed=1)
        W2, _, _ = generate_network(20, 4, 100, 30, seed=2)
        assert not np.array_equal(W1, W2)

    def test_graph_node_count(self):
        """NetworkX graph must have exactly N nodes."""
        N = 12
        _, _, G = generate_network(N, 3, 100, 30, seed=8)
        assert G.number_of_nodes() == N

    def test_graph_edge_consistency_with_W(self):
        """Graph edges must match non-zero entries in W."""
        W, _, G = generate_network(10, 3, 100, 30, seed=15)
        for i in range(10):
            for j in range(i + 1, 10):
                if W[i, j] > 0:
                    assert G.has_edge(i, j)
                else:
                    assert not G.has_edge(i, j)