"""
Scenario registry. Loads predefined scenarios from presets.json
and exposes lookup helpers used by the dashboard and experiment runner.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

_REGISTRY_PATH = Path(__file__).resolve().parent / "presets.json"


def _load_registry() -> Dict[str, dict]:
    """
    Loads the scenario registry from disk. Falls back to hardcoded values
    if the JSON file is missing (safety net for deployment).
    """
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback hardcoded registry
    return {
        "S1": {"N": 8, "K": 3, "area": 100, "threshold": 30, "description": "Small visual graph"},
        "S2": {"N": 30, "K": 4, "area": 150, "threshold": 35, "description": "Medium standard network"},
        "S3": {"N": 50, "K": 6, "area": 200, "threshold": 60, "description": "Dense congested network"},
        "S4": {"N": 50, "K": 2, "area": 200, "threshold": 40, "description": "Severely constrained spectrum"},
        "S5": {"N": 100, "K": 8, "area": 300, "threshold": 50, "description": "Large scalability testbed"},
        "S6": {"N": 50, "K": 4, "area": 200, "threshold": 35, "description": "Noise-robustness evaluation"},
        "S7": {"N": 45, "K": 5, "area": 200, "threshold": 30, "description": "Dynamic topology testbed"},
    }


_SCENARIOS = _load_registry()

SCENARIO_DESCRIPTIONS = {
    name: params.get("description", "")
    for name, params in _SCENARIOS.items()
}


def get_scenario(name: str) -> Optional[dict]:
    """
    Returns parameter dictionary for a scenario name, or None if not found.
    """
    return _SCENARIOS.get(name)


def list_scenarios() -> List[str]:
    """
    Returns the ordered list of registered scenario identifiers.
    """
    return sorted(_SCENARIOS.keys())


def get_scenario_description(name: str) -> str:
    """
    Returns the human-readable description for a scenario name.
    """
    return SCENARIO_DESCRIPTIONS.get(name, "")