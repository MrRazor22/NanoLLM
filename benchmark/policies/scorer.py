import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase
from inference.decision_engine import IDecisionEngine

@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Pure domain DTO: verified evaluation report for a decision engine on a benchmark suite."""
    model: str
    overall: float
    count: int
    p50_ms: float = 0.0
    brier_score: float = 0.0
    slices: Mapping[str, float] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    suite: str = "default"

@runtime_checkable
class IScorerPolicy(Protocol):
    """Opinion sub-primitive: scores decisions against ground truth to produce a report."""
    def score(
        self,
        engine: IDecisionEngine,
        cases: Sequence[DecisionCase],
        model_name: str = "Piqo",
        suite: str = "default",
    ) -> BenchmarkReport: ...

class ScorerPolicy(IScorerPolicy):
    """Concrete evaluation policy: runs live engine inference, tracks metrics, and computes scores."""

    def __init__(self, show_progress: bool = True, log_interval_sec: float = 3.0):
        self.show_progress = show_progress
        self.log_interval_sec = log_interval_sec

    def score(
        self,
        engine: IDecisionEngine,
        cases: Sequence[DecisionCase],
        model_name: str = "Piqo",
        suite: str = "default",
    ) -> BenchmarkReport:
        cat_correct: Dict[str, int] = {}
        cat_total: Dict[str, int] = {}
        latencies: list[float] = []
        hits: list[float] = []
        brier_diffs: list[float] = []
        total_items = len(cases)
        start_time = time.perf_counter()
        last_log = start_time

        for idx, item in enumerate(cases, 1):
            cat = item.category or "general"
            t_call = time.perf_counter()
            res = engine.decide(item.state, item.questions)
            dt_ms = res.latency_ms if res.latency_ms > 0 else (time.perf_counter() - t_call) * 1000.0
            latencies.append(dt_ms)

            for q in item.questions:
                if q.target is not None:
                    cat_total[cat] = cat_total.get(cat, 0) + 1
                    if q.name in res.answers:
                        ans = res.answers[q.name]
                        is_hit = q.matches(ans)
                        hits.append(1.0 if is_hit else 0.0)
                        if is_hit:
                            cat_correct[cat] = cat_correct.get(cat, 0) + 1

                        probs = getattr(ans, "probabilities", None)
                        if probs and isinstance(probs, dict):
                            target_str = str(q.target)
                            b_sum = sum((float(p_val) - (1.0 if str(opt_k) == target_str else 0.0)) ** 2 for opt_k, p_val in probs.items())
                            brier_diffs.append(b_sum)
                        elif hasattr(ans, "confidence") and ans.confidence is not None:
                            c = float(ans.confidence)
                            brier_diffs.append((c - (1.0 if is_hit else 0.0)) ** 2)
                    else:
                        hits.append(0.0)
                        brier_diffs.append(1.0)

            now = time.perf_counter()
            if self.show_progress and (now - last_log >= self.log_interval_sec or idx == total_items):
                p50 = float(statistics.median(latencies)) if latencies else 0.0
                acc = (sum(hits) / max(1, len(hits))) * 100.0
                speed = idx / max(0.001, now - start_time)
                print(f"[{model_name}] {idx}/{total_items} ({idx*100.0/total_items:.1f}%) | Acc: {acc:.1f}% | P50: {p50:.1f}ms | {speed:.1f} it/s", flush=True)
                last_log = now

        overall = sum(hits) / max(1, len(hits))
        p50 = float(statistics.median(latencies)) if latencies else 0.0
        brier = float(sum(brier_diffs) / max(1, len(brier_diffs))) if brier_diffs else 0.0
        return BenchmarkReport(
            model=model_name,
            overall=overall,
            count=len(hits),
            p50_ms=p50,
            brier_score=brier,
            slices={c: cat_correct.get(c, 0) / cat_total[c] for c in cat_total},
            counts=cat_total,
            suite=suite,
        )
