from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable
from inference import IDecisionEngine, DecisionCase, Question

DEFAULT_BASELINES = Path(__file__).resolve().parent / "baselines" / "competitor_cache.json"

BenchmarkItem = DecisionCase

@dataclass(frozen=True)
class BenchmarkReport:
    track: str
    overall: float
    count: int
    p50_ms: float
    slices: Mapping[str, float]

@runtime_checkable
class IEvaluator(Protocol):
    """Sovereign primitive contract for benchmark evaluation."""
    def evaluate(
        self,
        engine: IDecisionEngine,
        items: Sequence[BenchmarkItem],
        log_interval: int = 100,
        track_name: str = "general",
    ) -> BenchmarkReport: ...

class Scorecard(IEvaluator):
    """Sovereign primitive: evaluates engines against benchmark tracks and renders scorecards."""

    def __init__(self, baselines_path: Optional[Path] = None):
        p = baselines_path or DEFAULT_BASELINES
        raw = json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}
        self.baselines = {k: v for k, v in raw.items() if k != "p50_latencies_ms"}
        self.p50s = raw.get("p50_latencies_ms", {})

    def evaluate(
        self,
        engine: IDecisionEngine,
        items: Sequence[BenchmarkItem],
        log_interval: int = 100,
        track_name: str = "general",
    ) -> BenchmarkReport:
        stats: Dict[str, Dict[str, int]] = {}
        latencies, hits = [], []

        for item in items:
            cat = item.category
            if cat not in stats:
                stats[cat] = {"correct": 0, "total": 0}
            res = engine.decide(item.state, item.questions)
            if hasattr(res, "latency_ms") and res.latency_ms:
                latencies.append(res.latency_ms)

            match = bool(item.gold) and all(
                str(res.answers[qid].value) == str(target)
                for qid, target in item.gold.items()
                if qid in res.answers
            )
            hits.append(1.0 if match else 0.0)
            stats[cat]["total"] += 1
            if match:
                stats[cat]["correct"] += 1

        overall = sum(hits) / max(1, len(hits))
        p50 = float(sorted(latencies)[len(latencies) // 2]) if latencies else 0.0
        slices = {cat: d["correct"] / max(1, d["total"]) for cat, d in stats.items()}
        return BenchmarkReport(
            track=track_name,
            overall=overall,
            count=len(items),
            p50_ms=p50,
            slices=slices,
        )

    def render(self, reports: Mapping[str, BenchmarkReport], model_name: str = "Piqo") -> None:
        hdr = f"{'Benchmark Track':<32} | {'Count':>6} | {model_name:>12} | {'Verdict 2.0':>12} | {'Laya (421M)':>12}"
        print("\n" + "=" * len(hdr) + "\n" + hdr + "\n" + "-" * len(hdr))
        for track, rep in reports.items():
            base = self.baselines.get(track, {})
            v_val = base.get("Verdict 2.0 (151M)", {}).get("overall")
            l_val = base.get("Laya (421M)", {}).get("overall")
            v_str = f"{v_val*100:11.1f}%" if v_val is not None else "-"
            l_str = f"{l_val*100:11.1f}%" if l_val is not None else "-"
            print(f"{track:<32} | {rep.count:>6} | {rep.overall*100:11.1f}% | {v_str:>12} | {l_str:>12}")
        print("=" * len(hdr) + "\n")

__all__ = [
    "BenchmarkItem",
    "BenchmarkReport",
    "IEvaluator",
    "Scorecard",
]
