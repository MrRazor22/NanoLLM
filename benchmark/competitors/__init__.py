from benchmark.competitors.gliner_decide import GlinerDecideEngine
from benchmark.competitors.kev import KevEngine
from benchmark.competitors.laya import LayaEngine
from benchmark.competitors.verdict import VerdictEngine

AVAILABLE_COMPETITORS = {
    "laya": LayaEngine,
    "verdict": VerdictEngine,
    "kev": KevEngine,
    "gliner": GlinerDecideEngine,
}

__all__ = [
    "AVAILABLE_COMPETITORS",
    "GlinerDecideEngine",
    "KevEngine",
    "LayaEngine",
    "VerdictEngine",
]
