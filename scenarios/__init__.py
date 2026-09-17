"""
Scenarios package. Provides centralized access to predefined scenario families.
"""

from .scenario_registry import (
    get_scenario,
    list_scenarios,
    get_scenario_description,
    SCENARIO_DESCRIPTIONS,
)

__all__ = [
    "get_scenario",
    "list_scenarios",
    "get_scenario_description",
    "SCENARIO_DESCRIPTIONS",
]