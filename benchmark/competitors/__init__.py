"""Competitor Domain Adapters (PFD)."""

from typing import Dict, Type
from inference import IDecisionEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine

AVAILABLE_COMPETITORS: Dict[str, Type[IDecisionEngine]] = {
    "laya": LayaEngine,
    "verdict": VerdictEngine,
}

__all__ = ["LayaEngine", "VerdictEngine", "AVAILABLE_COMPETITORS"]
