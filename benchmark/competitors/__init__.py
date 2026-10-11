"""Competitor Domain Adapters (PFD)."""

from typing import Dict, Type
from inference import IDecisionEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine
from benchmark.competitors.kev import KevEngine
from benchmark.competitors.gliner_decide import GlinerDecideEngine

AVAILABLE_COMPETITORS: Dict[str, Type[IDecisionEngine]] = {
    "laya": LayaEngine,
    "verdict": VerdictEngine,
    "kev": KevEngine,
    "gliner_decide": GlinerDecideEngine,
}

__all__ = ["LayaEngine", "VerdictEngine", "KevEngine", "GlinerDecideEngine", "AVAILABLE_COMPETITORS"]
