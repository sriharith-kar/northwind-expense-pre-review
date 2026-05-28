from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal


TraceStatus = Literal["ok", "retried", "fallback", "degraded", "error"]


# Pricing is centralized so deployments can swap in provider-specific rates.
# Gemini entries default to zero unless explicit rates are added, which avoids
# silently presenting stale pricing as an audited cost.
MODEL_COST_PER_1K_TOKENS_USD: dict[str, dict[str, float]] = {
    "n/a": {"input": 0.0, "output": 0.0},
    "deterministic": {"input": 0.0, "output": 0.0},
    "pytesseract": {"input": 0.0, "output": 0.0},
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},
    "gemini-embedding-001": {"input": 0.0, "output": 0.0},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
}


def calculate_model_cost_usd(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Computes token cost for traced model calls so per-step economics are auditable."""
    prices = MODEL_COST_PER_1K_TOKENS_USD.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1000 * prices["input"]) + (output_tokens / 1000 * prices["output"])


@dataclass
class TraceStep:
    step_name: str
    model_used: str = "n/a"
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    status: TraceStatus = "ok"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "model_used": self.model_used,
            "latency_ms": round(self.latency_ms, 3),
            "cost_usd": round(self.cost_usd, 8),
            "status": self.status,
            "notes": self.notes,
        }


class _StepContext(AbstractContextManager[TraceStep]):
    def __init__(self, tracer: "StepTracer", step_name: str, model_used: str = "n/a"):
        self.tracer = tracer
        self.step = TraceStep(step_name=step_name, model_used=model_used)
        self._started = 0.0

    def __enter__(self) -> TraceStep:
        self._started = time.perf_counter()
        return self.step

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.step.latency_ms = (time.perf_counter() - self._started) * 1000
        if exc is not None and self.step.status == "ok":
            self.step.status = "error"
            self.step.notes = self.step.notes or str(exc)
        self.tracer.steps.append(self.step)
        return False


class StepTracer:
    """Collects per-receipt pipeline telemetry without affecting verdict logic."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.steps: list[TraceStep] = []

    def step(self, step_name: str, model_used: str = "n/a") -> AbstractContextManager[TraceStep]:
        if not self.enabled:
            return _DisabledStep()
        return _StepContext(self, step_name, model_used)

    def to_list(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]


class _DisabledStep(AbstractContextManager[TraceStep]):
    def __enter__(self) -> TraceStep:
        return TraceStep(step_name="disabled")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False
