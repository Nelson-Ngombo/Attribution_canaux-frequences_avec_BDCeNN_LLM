"""
Unit tests for the verifier module.
Validates number extraction, whitelist construction, and audit verdicts.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifier import ResponseVerifier


@pytest.fixture
def verifier():
    return ResponseVerifier(tolerance=0.01)


@pytest.fixture
def sample_case_data():
    """Minimal case_data for testing the audit pipeline."""
    return {
        "case_id": "#3",
        "scenario": "S2",
        "N": 30,
        "K": 4,
        "seed": 1,
        "metrics": {
            "cost_initial": 120.5,
            "cost_final": 15.75,
            "conflicts": 8,
            "time_seconds": 0.342,
            "iterations": 12,
            "used_channels": 4,
        },
        "baselines": {
            "Random": {"cost": 85.0, "conflicts": 22, "time": 0.001},
            "Greedy": {"cost": 25.0, "conflicts": 10, "time": 0.005},
            "DSATUR": {"cost": 20.0, "conflicts": 9, "time": 0.008},
        },
        "conflicting_cells": [3, 7, 15],
        "cell_positions": {0: [10.5, 20.3], 1: [30.1, 40.2]},
        "topology_edges": [(0, 1, 4), (1, 2, 2)],
        "total_edges": 45,
        "topology_meta": {"seed": 1, "threshold": 35, "N": 30, "K": 4},
        "allocations": {"BD-CeNN": [0, 1, 2, 3, 0, 1]},
        "cell_summary": [{"cell": 0, "degree": 5, "strength": 12}],
    }


class TestNumberExtraction:
    """Tests for regex-based number parsing."""

    def test_integer_extraction(self, verifier):
        result = verifier.extract_numbers("The cost is 42 and conflicts 8.")
        values = [v for v, _ in result]
        assert 42.0 in values
        assert 8.0 in values

    def test_float_extraction(self, verifier):
        result = verifier.extract_numbers("Runtime was 0.342 seconds.")
        values = [v for v, _ in result]
        assert 0.342 in values

    def test_negative_number(self, verifier):
        result = verifier.extract_numbers("Delta is -5.2 percent.")
        values = [v for v, _ in result]
        assert -5.2 in values

    def test_comma_decimal_separator(self, verifier):
        result = verifier.extract_numbers("Cost is 15,75 units.")
        values = [v for v, _ in result]
        assert 15.75 in values

    def test_no_numbers(self, verifier):
        result = verifier.extract_numbers("No numbers here at all.")
        assert len(result) == 0


class TestAllowedNumbers:
    """Tests for the certified number whitelist."""

    def test_structural_params_included(self, verifier, sample_case_data):
        allowed = verifier.collect_allowed_numbers(sample_case_data)
        assert 30.0 in allowed   # N
        assert 4.0 in allowed    # K
        assert 1.0 in allowed    # seed

    def test_metrics_included(self, verifier, sample_case_data):
        allowed = verifier.collect_allowed_numbers(sample_case_data)
        assert 15.75 in allowed  # cost_final
        assert 8.0 in allowed    # conflicts

    def test_baseline_values_included(self, verifier, sample_case_data):
        allowed = verifier.collect_allowed_numbers(sample_case_data)
        assert 85.0 in allowed   # Random cost
        assert 25.0 in allowed   # Greedy cost

    def test_case_id_included(self, verifier, sample_case_data):
        allowed = verifier.collect_allowed_numbers(sample_case_data)
        assert 3.0 in allowed    # from "#3"

    def test_scenario_digit_included(self, verifier, sample_case_data):
        allowed = verifier.collect_allowed_numbers(sample_case_data)
        assert 2.0 in allowed    # from "S2"


class TestAudit:
    """Tests for the full audit pipeline."""

    def test_certified_response(self, verifier, sample_case_data):
        """A response citing only valid numbers should be CERTIFIED."""
        text = "The final cost is 15.75 with 8 conflicts across 30 cells."
        report = verifier.audit(text, sample_case_data)
        assert report.status == "CERTIFIED"
        assert report.invented_count == 0

    def test_failed_response(self, verifier, sample_case_data):
        """A response with fabricated numbers should be FAILED."""
        text = "The cost is 999.99 and there are 500 conflicts."
        report = verifier.audit(text, sample_case_data)
        assert report.status == "FAILED"
        assert report.invented_count >= 1

    def test_warning_on_empty(self, verifier, sample_case_data):
        """A response with no numbers should trigger WARNING."""
        text = "The results look good overall with no specific figures."
        report = verifier.audit(text, sample_case_data)
        assert report.status == "WARNING"

    def test_accuracy_rate(self, verifier, sample_case_data):
        """Accuracy should reflect the ratio of valid to total numbers."""
        text = "Cost 15.75 and fabricated 999."
        report = verifier.audit(text, sample_case_data)
        # 15.75 is valid, 999 is not
        assert report.valid_count >= 1
        assert report.invented_count >= 1
        assert 0 < report.accuracy_rate < 100