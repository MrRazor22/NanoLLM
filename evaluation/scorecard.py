from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable
from inference import DecisionCase, IDecisionEngine

DEFAULT_BASELINES = Path(__file__).resolve().parent / "baselines" / "competitor_cache.json"

BenchmarkItem = DecisionCase

@dataclass
class CategoryHits:
    """Explicit domain DTO tracking category accuracy hits and totals."""
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

@dataclass(frozen=True)
class BaselineTrack:
    """Explicit domain DTO representing competitor benchmark accuracy."""
    verdict_overall: Optional[float] = None
    laya_overall: Optional[float] = None

@dataclass(frozen=True)
class BenchmarkReport:
    """Immutable domain DTO summarizing benchmark evaluation results."""
    track: str
    overall: float
    count: int
    p50_ms: float
    slices: Mapping[str, float]

@runtime_checkable
class IScorecard(Protocol):
    """Core Primitive: Operational boundary for evaluating decision engines against benchmarks."""
    def evaluate(
        self,
        engine: IDecisionEngine,
        items: Sequence[BenchmarkItem],
        track_name: str = "general",
    ) -> BenchmarkReport: ...

class Scorecard(IScorecard):
    """Benchmark runner: evaluates decision engines against benchmark tracks and renders scorecards."""

    def __init__(self, baselines_path: Optional[Path] = None):
        p = baselines_path or DEFAULT_BASELINES
        raw = json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}
        baselines: Dict[str, BaselineTrack] = {}
        for track, data in raw.items():
            if isinstance(data, dict):
                v = data.get("Verdict 2.0 (151M)", {}).get("overall") if isinstance(data.get("Verdict 2.0 (151M)"), dict) else None
                l = data.get("Laya (421M)", {}).get("overall") if isinstance(data.get("Laya (421M)"), dict) else None
                baselines[track] = BaselineTrack(verdict_overall=v, laya_overall=l)
        self.baselines = baselines

    def evaluate(
        self,
        engine: IDecisionEngine,
        items: Sequence[BenchmarkItem],
        track_name: str = "general",
    ) -> BenchmarkReport:
        stats: Dict[str, CategoryHits] = {}
        latencies, hits = [], []

        for item in items:
            cat = item.category
            if cat not in stats:
                stats[cat] = CategoryHits()
            res = engine.decide(item.state, item.questions)
            if res.latency_ms:
                latencies.append(res.latency_ms)

            q_by_name = {q.name: q for q in item.questions}
            def _matches(qid: str, target) -> bool:
                if qid not in res.answers: return False
                ans_val = str(res.answers[qid].value)
                if ans_val == str(target): return True
                q = q_by_name.get(qid)
                if isinstance(q, Choice):
                    keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                    if isinstance(target, int) and 0 <= target < len(keys): return ans_val == str(keys[target])
                    if str(target).isdigit() and 0 <= int(target) < len(keys): return ans_val == str(keys[int(target)])
                return False

            match = bool(item.gold) and all(_matches(qid, tgt) for qid, tgt in item.gold.items())
            hits.append(1.0 if match else 0.0)
            stats[cat].total += 1
            if match:
                stats[cat].correct += 1

        overall = sum(hits) / max(1, len(hits))
        p50 = float(sorted(latencies)[len(latencies) // 2]) if latencies else 0.0
        slices = {cat: ch.accuracy for cat, ch in stats.items()}
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
            base = self.baselines.get(track, BaselineTrack())
            v_str = f"{base.verdict_overall * 100:11.1f}%" if base.verdict_overall is not None else "-"
            l_str = f"{base.laya_overall * 100:11.1f}%" if base.laya_overall is not None else "-"
            print(f"{track:<32} | {rep.count:>6} | {rep.overall*100:11.1f}% | {v_str:>12} | {l_str:>12}")
        print("=" * len(hdr) + "\n")

__all__ = ["CategoryHits", "BaselineTrack", "BenchmarkItem", "BenchmarkReport", "IScorecard", "Scorecard"]
