"""
LLMQuery and LLMResponse dataclasses. Model the request/response cycle
with the language model backend.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class LLMQuery:
    """
    Structured representation of a prompt sent to the language model.
    """
    function: str
    prompt: str
    system_prompt: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "function": self.function,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "timestamp": self.timestamp,
        }


@dataclass
class LLMResponse:
    """
    Structured representation of a raw response returned by the LLM.
    """
    raw_text: str
    model_name: str
    latency_seconds: float = 0.0
    parsed_data: Optional[dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    error: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.error is not None or self.raw_text.startswith("[LLM_ERROR]") or self.raw_text.startswith("[ERREUR LLM]")

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "model_name": self.model_name,
            "latency_seconds": float(self.latency_seconds),
            "parsed_data": self.parsed_data,
            "timestamp": self.timestamp,
            "error": self.error,
        }