"""
Data structures package. Exposes all dataclasses used across the framework
for type-safe inter-module communication using absolute imports.
"""

from data_structures.topology import NetworkTopology
from data_structures.solver_result import SolverResult
from data_structures.experiment_record import ExperimentRecord, MetricsRecord
from data_structures.llm_exchange import LLMQuery, LLMResponse
from data_structures.audit_report import AuditReport, AuditVerification

__all__ = [
    "NetworkTopology",
    "SolverResult",
    "ExperimentRecord",
    "MetricsRecord",
    "LLMQuery",
    "LLMResponse",
    "AuditReport",
    "AuditVerification",
]