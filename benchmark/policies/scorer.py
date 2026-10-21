from dataclasses import dataclass, field
import statistics
import time
from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from inference.decision_engine import IDecisionEngine
from inference.options import ChoiceAnswer, ChoiceOption, IOption

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

            if hasattr(item, "candidates") and item.candidates:
                resolved = engine.decide(item.state, ChoiceOption(options=item.candidates))
                dt_ms = getattr(engine, "last_latency_ms", 0.0) or (time.perf_counter() - t_call) * 1000.0
                latencies.append(dt_ms)
                cat_total[cat] = cat_total.get(cat, 0) + 1
                ans = resolved.answer
                pred_choice = ans.choice if isinstance(ans, ChoiceAnswer) else ""
                gold_choice = item.choice()
                is_hit = (pred_choice == gold_choice)
                hits.append(1.0 if is_hit else 0.0)
                if is_hit:
                    cat_correct[cat] = cat_correct.get(cat, 0) + 1
                probs = ans.probabilities if isinstance(ans, ChoiceAnswer) else {}
                b_val = sum((float(probs.get(c, 0.0)) - float(item.probabilities.get(c, 0.0))) ** 2 for c in item.candidates)
                brier_diffs.append(b_val)

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
