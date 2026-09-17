"""
Unit tests for the metrics module.
Validates co-channel and adjacent-channel cost computations,
conflict counting, and statistical functions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import (
    compute_cochannel_cost,
    count_cochannel_conflicts,
    compute_adjacent_cost,
    count_adjacent_conflicts,
    create_channel_interference_matrix,
    compute_confidence_interval,
)


class TestCochannelMetrics:
    """Tests for CCI-only cost and conflict functions."""

    def test_zero_cost_no_conflicts(self):
        """Distinct channels on all edges should yield zero cost."""
        x = np.array([0, 1, 2, 3])
        W = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ], dtype=float)
        assert compute_cochannel_cost(x, W) == 0.0
        assert count_cochannel_conflicts(x, W) == 0

    def test_full_conflict(self):
        """All cells on the same channel should maximize cost."""
        x = np.array([0, 0, 0])
        W = np.array([
            [0, 2, 1],
            [2, 0, 4],
            [1, 4, 0],
        ], dtype=float)
        expected_cost = 2 + 1 + 4  # all pairs conflict
        assert compute_cochannel_cost(x, W) == expected_cost
        assert count_cochannel_conflicts(x, W) == 3

    def test_partial_conflict(self):
        """Only conflicting pairs should contribute to cost."""
        x = np.array([0, 0, 1])
        W = np.array([
            [0, 3, 2],
            [3, 0, 1],
            [2, 1, 0],
        ], dtype=float)
        # Only (0,1) conflicts (both channel 0)
        assert compute_cochannel_cost(x, W) == 3.0
        assert count_cochannel_conflicts(x, W) == 1


class TestAdjacentMetrics:
    """Tests for CCI+ACI cost and conflict functions."""

    def test_channel_matrix_properties(self):
        """M must be symmetric with 1.0 on diagonal."""
        M = create_channel_interference_matrix(6, decay=0.5, cutoff=2)
        assert M.shape == (6, 6)
        np.testing.assert_array_equal(np.diag(M), 1.0)
        np.testing.assert_array_equal(M, M.T)

    def test_channel_matrix_decay(self):
        """Off-diagonal values must follow decay^distance."""
        M = create_channel_interference_matrix(5, decay=0.5, cutoff=2)
        assert M[0, 1] == pytest.approx(0.5)
        assert M[0, 2] == pytest.approx(0.25)
        assert M[0, 3] == pytest.approx(0.0)  # beyond cutoff

    def test_adjacent_cost_higher_than_cochannel(self):
        """CCI+ACI cost must be >= CCI cost for the same assignment."""
        x = np.array([0, 1, 2, 0])
        W = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ], dtype=float)
        M = create_channel_interference_matrix(3, decay=0.5, cutoff=2)
        cost_cci = compute_cochannel_cost(x, W)
        cost_aci = compute_adjacent_cost(x, W, M)
        assert cost_aci >= cost_cci

    def test_adjacent_conflict_count(self):
        """Adjacent conflicts should include near-channel pairs."""
        x = np.array([0, 1, 3])
        W = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        M = create_channel_interference_matrix(4, decay=0.5, cutoff=2)
        # (0,1): channels 0,1 -> M[0,1]=0.5 > 0 -> conflict
        # (0,2): channels 0,3 -> M[0,3]=0.0 -> no conflict
        # (1,2): channels 1,3 -> M[1,3]=0.25 > 0 -> conflict
        assert count_adjacent_conflicts(x, W, M) == 2


class TestStatisticalFunctions:
    """Tests for confidence interval computation."""

    def test_confidence_interval_symmetry(self):
        """CI should be symmetric around the mean."""
        data = [10, 12, 14, 11, 13]
        lower, upper = compute_confidence_interval(data)
        mean_val = np.mean(data)
        assert lower < mean_val < upper
        assert pytest.approx(mean_val - lower, abs=1e-6) == pytest.approx(upper - mean_val, abs=1e-6)

    def test_single_element(self):
        """Single element should return (val, val)."""
        lower, upper = compute_confidence_interval([5.0])
        assert lower == 5.0
        assert upper == 5.0