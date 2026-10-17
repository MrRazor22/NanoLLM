from dataclasses import dataclass, field
import statistics
import time
from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable
from data.case import DecisionCase
from inference.decision_engine import IDecisionEngine
from inference.questions import Choice, Noul, Score, choice_keys, matches_answer, score_to_norm

@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Pure domain DTO: verified evaluation report for a decision engine on a benchmark suite."""
    model: str
    overall: float
    count: int
    p50_ms: float = 0.0
    brier_score: float = 0.0
    regression_mse: float = 0.0
    slices: Mapping[str, float] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    suite: str = "default"
    omissions: int = 0

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
        regression_diffs: list[float] = []
        omissions = 0
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
                        is_hit = matches_answer(q, ans)
                        hits.append(1.0 if is_hit else 0.0)
                        if is_hit:
                            cat_correct[cat] = cat_correct.get(cat, 0) + 1

                        if isinstance(q, Choice):
                            probs = getattr(ans, "probabilities", None)
                            keys = choice_keys(q)
                            b_val = sum((float(probs.get(k, 0.0)) - (1.0 if str(k) == str(q.target) else 0.0)) ** 2 for k in keys) if (probs and len(probs) >= len(keys)) else 1.0
                            brier_diffs.append(b_val)
                        elif isinstance(q, Noul):
                            prob = getattr(ans, "probability", None)
                            tgt_b = 1.0 if (q.target is True or str(q.target).lower() in ("true", "1", "yes")) else 0.0
                            brier_diffs.append((float(prob) - tgt_b) ** 2 if prob is not None else 1.0)
                        elif isinstance(q, Score):
                            norm = getattr(ans, "normalized", None)
                            regression_diffs.append((float(norm) - score_to_norm(q, float(q.target))) ** 2 if norm is not None else 1.0)
                    else:
                        hits.append(0.0)
                        omissions += 1
                        (regression_diffs if isinstance(q, Score) else brier_diffs).append(1.0)

            now = time.perf_counter()
            if self.show_progress and (now - last_log >= self.log_interval_sec or idx == total_items):
                p50 = float(statistics.median(latencies)) if latencies else 0.0
                acc = (sum(hits) / max(1, len(hits))) * 100.0
                speed = idx / max(0.001, now - start_time)
                print(f"[{model_name}] {idx}/{total_items} ({idx*100.0/total_items:.1f}%) | Acc: {acc:.1f}% | P50: {p50:.1f}ms | {speed:.1f} it/s", flush=True)
                last_log = now

        return BenchmarkReport(
            model=model_name,
            overall=sum(hits) / max(1, len(hits)),
            count=len(hits),
            p50_ms=float(statistics.median(latencies)) if latencies else 0.0,
            brier_score=float(sum(brier_diffs) / max(1, len(brier_diffs))) if brier_diffs else 0.0,
            regression_mse=float(sum(regression_diffs) / max(1, len(regression_diffs))) if regression_diffs else 0.0,
            slices={c: cat_correct.get(c, 0) / cat_total[c] for c in cat_total},
            counts=cat_total,
            suite=suite,
            omissions=omissions,
        )
