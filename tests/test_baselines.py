"""
Unit tests for the baselines module.
Validates Random, Greedy, and DSATUR allocation correctness.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import (
    random_allocation,
    greedy_allocation,
    dsatur_allocation,
    create_channel_interference_matrix,
)


class TestRandomAllocation:
    """Tests for the random channel assignment."""

    def test_output_shape(self):
        x = random_allocation(20, 5)
        assert x.shape == (20,)

    def test_values_in_range(self):
        x = random_allocation(50, 8)
        assert np.all(x >= 0)
        assert np.all(x < 8)

    def test_single_channel(self):
        """With K=1, all cells must be assigned channel 0."""
        x = random_allocation(10, 1)
        np.testing.assert_array_equal(x, 0)


class TestGreedyAllocation:
    """Tests for the greedy sequential assignment."""

    def test_output_shape_and_range(self):
        N, K = 15, 4
        W = np.ones((N, N)) - np.eye(N)
        x = greedy_allocation(N, K, W)
        assert x.shape == (N,)
        assert np.all(x >= 0)
        assert np.all(x < K)

    def test_no_conflict_when_enough_channels(self):
        """With K >= N and a star topology, greedy should find zero cost."""
        N, K = 5, 5
        W = np.zeros((N, N))
        W[0, 1:] = 1
        W[1:, 0] = 1
        x = greedy_allocation(N, K, W)
        cost = 0
        for i in range(N):
            for j in range(i + 1, N):
                if W[i, j] > 0 and x[i] == x[j]:
                    cost += W[i, j]
        assert cost == 0

    def test_custom_order(self):
        """Custom order should be respected without errors."""
        N, K = 6, 3
        W = np.ones((N, N)) - np.eye(N)
        order = [5, 3, 1, 0, 2, 4]
        x = greedy_allocation(N, K, W, order=order)
        assert x.shape == (N,)
        assert np.all(x >= 0)

    def test_adjacent_aware_mode(self):
        """Greedy with M should not crash and should produce valid output."""
        N, K = 10, 4
        W = np.ones((N, N)) - np.eye(N)
        M = create_channel_interference_matrix(K)
        x = greedy_allocation(N, K, W, M=M)
        assert x.shape == (N,)
        assert np.all(x >= 0)
        assert np.all(x < K)


class TestDSATURAllocation:
    """Tests for the DSATUR assignment."""

    def test_output_shape_and_range(self):
        N, K = 12, 5
        W = np.ones((N, N)) - np.eye(N)
        x = dsatur_allocation(N, K, W)
        assert x.shape == (N,)
        assert np.all(x >= 0)
        assert np.all(x < K)

    def test_deterministic(self):
        """DSATUR should be deterministic for the same input."""
        N, K = 8, 3
        W = np.array([
            [0, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0],
        ], dtype=float)
        x1 = dsatur_allocation(N, K, W)
        x2 = dsatur_allocation(N, K, W)
        np.testing.assert_array_equal(x1, x2)

    def test_adjacent_aware_mode(self):
        """DSATUR with M should produce valid output."""
        N, K = 8, 4
        W = np.ones((N, N)) - np.eye(N)
        M = create_channel_interference_matrix(K)
        x = dsatur_allocation(N, K, W, M=M)
        assert x.shape == (N,)
        assert np.all(x >= 0)
        assert np.all(x < K)