from dataclasses import dataclass, field
import hashlib
import statistics
import time
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase
from inference.decision_engine import IDecisionEngine

@dataclass(slots=True)
class CategoryHits:
    correct: int = 0
    total: int = 0
    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    model: str
    overall: float
    count: int
    p50_ms: float = 0.0
    slices: Mapping[str, float] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    suite: str = "default"

def signature_for(items: Sequence[DecisionCase]) -> str:
    """Computes a lightweight deterministic hash signature over benchmark cases."""
    raw = f"{len(items)}|{items[0].state[:30] if items else ''}|{items[-1].state[:30] if items else ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

def _matches(answers: Mapping[str, Any], qid: str, target: Any, q_map: Mapping[str, Any]) -> bool:
    if qid not in answers:
        return False
    val = str(answers[qid].value)
    if val == str(target):
        return True
    opts = list(getattr(q_map.get(qid), "options", []))
    return val == str(opts[int(target)]) if str(target).isdigit() and 0 <= int(target) < len(opts) else False

@runtime_checkable
class IScorecard(Protocol):
    """Core Primitive: Evaluates decision engines on benchmark items to produce verified score reports."""
    def evaluate(
        self,
        engine: IDecisionEngine,
        items: Sequence[DecisionCase],
        model_name: str = "Piqo",
        suite: str = "default",
    ) -> BenchmarkReport: ...

class Scorecard(IScorecard):
    """Core Primitive Implementation: Evaluates decision engines on benchmark items."""

    def __init__(self, show_progress: bool = True, log_interval_sec: float = 3.0):
        self.show_progress = show_progress
        self.log_interval_sec = log_interval_sec

    def evaluate(
        self,
        engine: IDecisionEngine,
        items: Sequence[DecisionCase],
        model_name: str = "Piqo",
        suite: str = "default",
    ) -> BenchmarkReport:
        stats: Dict[str, CategoryHits] = {}
        latencies, hits = [], []
        total_items = len(items)
        start_time = time.perf_counter()
        last_log = start_time

        for idx, item in enumerate(items, 1):
            cat = item.category or "general"
            if cat not in stats:
                stats[cat] = CategoryHits()
            t_call = time.perf_counter()
            res = engine.decide(item.state, item.questions)
            dt_ms = res.latency_ms if res.latency_ms > 0 else (time.perf_counter() - t_call) * 1000.0
            latencies.append(dt_ms)

            if item.gold:
                q_map = {q.name: q for q in item.questions}
                for qid, tgt in item.gold.items():
                    m = _matches(res.answers, qid, tgt, q_map)
                    hits.append(1.0 if m else 0.0)
                    stats[cat].total += 1
                    if m:
                        stats[cat].correct += 1

            now = time.perf_counter()
            if self.show_progress and (now - last_log >= self.log_interval_sec or idx == total_items):
                p50 = float(statistics.median(latencies)) if latencies else 0.0
                acc = (sum(hits) / max(1, len(hits))) * 100.0
                speed = idx / max(0.001, now - start_time)
                print(f"[{model_name}] {idx}/{total_items} ({idx*100.0/total_items:.1f}%) | Acc: {acc:.1f}% | P50: {p50:.1f}ms | {speed:.1f} it/s", flush=True)
                last_log = now

        overall = sum(hits) / max(1, len(hits))
        p50 = float(statistics.median(latencies)) if latencies else 0.0
        return BenchmarkReport(
            model=model_name,
            overall=overall,
            count=len(hits),
            p50_ms=p50,
            slices={c: h.accuracy for c, h in stats.items()},
            counts={c: h.total for c, h in stats.items()},
            suite=suite,
        )
