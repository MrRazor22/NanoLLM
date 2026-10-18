from dataclasses import dataclass, field
import statistics
import time
from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable
from data.case import (
    ChoiceOption, DecisionCase, IOption, NoulOption, ScoreOption,
    choice_keys, matches_answer, score_to_norm,
)

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
            res_opt = engine.decide(item.state, item.option)
            dt_ms = getattr(engine, "last_latency_ms", 0.0) or (time.perf_counter() - t_call) * 1000.0
            latencies.append(dt_ms)

            gold_opt = item.option
            if gold_opt.answer is not None:
                cat_total[cat] = cat_total.get(cat, 0) + 1
                if res_opt.answer is not None:
                    is_hit = matches_answer(gold_opt, res_opt)
                    hits.append(1.0 if is_hit else 0.0)
                    if is_hit:
                        cat_correct[cat] = cat_correct.get(cat, 0) + 1

                    if isinstance(gold_opt, ChoiceOption):
                        probs = getattr(res_opt.answer, "probabilities", None)
                        keys = choice_keys(gold_opt)
                        tgt_c = str(gold_opt.answer.choice)
                        b_val = sum((float(probs.get(k, 0.0)) - (1.0 if str(k) == tgt_c else 0.0)) ** 2 for k in keys) if (probs and len(probs) >= len(keys)) else (0.0 if is_hit else 1.0)
                        brier_diffs.append(b_val)
                    elif isinstance(gold_opt, NoulOption):
                        conf = float(res_opt.answer.confidence)
                        tgt_b = 1.0 if gold_opt.answer.decision else 0.0
                        brier_diffs.append((conf - tgt_b) ** 2)
                    elif isinstance(gold_opt, ScoreOption):
                        regression_diffs.append((score_to_norm(gold_opt, res_opt.answer.score) - score_to_norm(gold_opt, gold_opt.answer.score)) ** 2)
                else:
                    hits.append(0.0)
                    omissions += 1
                    (regression_diffs if isinstance(gold_opt, ScoreOption) else brier_diffs).append(1.0)

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
