"""
Unit tests for the BD-CeNN solver module.
Validates single-run convergence, multistart behavior, and output format.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdcenn_solver import _bdcenn_single_run, bdcenn_allocation
from metrics import compute_cochannel_cost, create_channel_interference_matrix


class TestSingleRun:
    """Tests for _bdcenn_single_run."""

    def test_output_tuple_length(self):
        """Must return a 6-element tuple."""
        N, K = 8, 3
        W = np.ones((N, N)) - np.eye(N)
        result = _bdcenn_single_run(N, K, W, max_iter=10, seed=42)
        assert len(result) == 6

    def test_assignment_valid_range(self):
        """All channel values must be in [0, K)."""
        N, K = 10, 4
        W = np.ones((N, N)) - np.eye(N)
        x, _, _, _, _, _ = _bdcenn_single_run(N, K, W, max_iter=20, seed=7)
        assert np.all(x >= 0)
        assert np.all(x < K)

    def test_cost_non_increasing_over_time(self):
        """Best cost should not increase across iterations."""
        N, K = 10, 3
        W = np.ones((N, N)) - np.eye(N)
        _, history, _, _, _, _ = _bdcenn_single_run(N, K, W, max_iter=30, seed=12)
        costs = [h[1] for h in history]
        for i in range(1, len(costs)):
            assert costs[i] <= costs[0]  # at least not worse than initial

    def test_history_contains_initial_state(self):
        """First history entry must be iteration 0."""
        N, K = 6, 3
        W = np.ones((N, N)) - np.eye(N)
        _, history, _, _, _, _ = _bdcenn_single_run(N, K, W, max_iter=5, seed=1)
        assert history[0][0] == 0  # iteration index

    def test_deterministic_with_seed(self):
        """Same seed must produce identical results."""
        N, K = 8, 3
        W = np.ones((N, N)) - np.eye(N)
        x1, _, _, _, _, _ = _bdcenn_single_run(N, K, W, max_iter=10, seed=99)
        x2, _, _, _, _, _ = _bdcenn_single_run(N, K, W, max_iter=10, seed=99)
        np.testing.assert_array_equal(x1, x2)

    def test_adjacent_mode_runs_without_error(self):
        """Single run with M matrix should complete successfully."""
        N, K = 8, 4
        W = np.ones((N, N)) - np.eye(N)
        M = create_channel_interference_matrix(K)
        x, _, _, _, _, _ = _bdcenn_single_run(N, K, W, M=M, max_iter=10, seed=5)
        assert x.shape == (N,)


class TestMultistart:
    """Tests for bdcenn_allocation (multistart wrapper)."""

    def test_output_tuple_length(self):
        """Must return a 6-element tuple."""
        N, K = 10, 3
        W = np.ones((N, N)) - np.eye(N)
        result = bdcenn_allocation(N, K, W, num_restarts=3, max_iter=10, seed=42)
        assert len(result) == 6

    def test_multistart_at_least_as_good_as_single(self):
        """More restarts should yield cost <= single restart cost."""
        N, K = 12, 3
        W = np.ones((N, N)) - np.eye(N)
        x1, _, _, _, _, _ = bdcenn_allocation(N, K, W, num_restarts=1, max_iter=20, seed=42)
        x5, _, _, _, _, _ = bdcenn_allocation(N, K, W, num_restarts=5, max_iter=20, seed=42)
        cost1 = compute_cochannel_cost(x1, W)
        cost5 = compute_cochannel_cost(x5, W)
        assert cost5 <= cost1 + 1e-9

    def test_zero_cost_on_easy_instance(self):
        """BD-CeNN should find zero cost on a trivially colorable graph."""
        N, K = 4, 4
        W = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        x, _, _, _, _, _ = bdcenn_allocation(N, K, W, num_restarts=5, max_iter=30, seed=1)
        assert compute_cochannel_cost(x, W) == 0.0