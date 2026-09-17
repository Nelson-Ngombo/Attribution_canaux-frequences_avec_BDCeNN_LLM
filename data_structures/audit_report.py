"""
AuditReport and AuditVerification dataclasses. Formalize the outcome
of the regex-based numerical audit of LLM responses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class AuditVerification:
    """
    Single number verification record.
    """
    extracted_raw: str
    extracted_value: float
    matched: bool
    matched_reference: float = 0.0
    tolerance: float = 0.01

    def to_dict(self) -> dict:
        return {
            "extracted_raw": self.extracted_raw,
            "extracted_value": float(self.extracted_value),
            "matched": bool(self.matched),
            "matched_reference": float(self.matched_reference),
            "tolerance": float(self.tolerance),
        }


@dataclass
class AuditReport:
    """
    Full audit trail for an LLM response.

    Status semantics:
        CERTIFIED: all extracted numbers match certified data
        WARNING: no numbers extracted (no basis for verification)
        FAILED: at least one number contradicts certified data
    """
    status: str
    accuracy_rate: float
    total_numbers: int
    valid_count: int
    invented_count: int
    verifications: List[AuditVerification] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    tolerance: float = 0.01

    @property
    def divergences(self) -> List[AuditVerification]:
        return [v for v in self.verifications if not v.matched]

    @property
    def is_certified(self) -> bool:
        return self.status == "CERTIFIED"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "accuracy_rate": float(self.accuracy_rate),
            "total_numbers": int(self.total_numbers),
            "valid_count": int(self.valid_count),
            "invented_count": int(self.invented_count),
            "verifications": [v.to_dict() for v in self.verifications],
            "divergences": [v.to_dict() for v in self.divergences],
            "timestamp": self.timestamp,
            "tolerance": float(self.tolerance),
        }