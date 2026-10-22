import logging
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable

from benchmark.suites.case import BenchmarkCase
from inference.decision_engine import IDecisionEngine
from inference.options import ChoiceAnswer, NoulAnswer, ScoreAnswer

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class DecisionMetrics:
    accuracy: float
    latency_p50_ms: float
    brier_score: float
    category_accuracy: Mapping[str, float]
    category_counts: Mapping[str, int]
    count: int

@runtime_checkable
class IScorerPolicy(Protocol):
    def score(
        self,
        engine: IDecisionEngine,
        cases: Sequence[BenchmarkCase],
    ) -> DecisionMetrics: ...

class ScorerPolicy(IScorerPolicy):
    def score(
        self,
        engine: IDecisionEngine,
        cases: Sequence[BenchmarkCase],
    ) -> DecisionMetrics:
        cat_correct: Dict[str, int] = {}
        cat_total: Dict[str, int] = {}
        latencies: list[float] = []
        hits: list[float] = []
        brier_diffs: list[float] = []
        total_items = len(cases)
        engine_name = engine.__class__.__name__

        for idx, item in enumerate(cases, 1):
            t0 = time.perf_counter()
            ans = engine.decide(item.state, item.option)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            cat_total[item.category] = cat_total.get(item.category, 0) + 1

            match (ans, item.expected):
                case (ChoiceAnswer() as a, ChoiceAnswer() as e):
                    is_hit = (a.choice == e.choice)
                    hits.append(1.0 if is_hit else 0.0)
                    if is_hit:
                        cat_correct[item.category] = cat_correct.get(item.category, 0) + 1
                    if e.probabilities:
                        brier_diffs.append(sum((float(a.probabilities.get(c, 0.0)) - float(e.probabilities[c])) ** 2 for c in e.probabilities))
                case (NoulAnswer() as a, NoulAnswer() as e):
                    is_hit = (a.decision == e.decision)
                    hits.append(1.0 if is_hit else 0.0)
                    if is_hit:
                        cat_correct[item.category] = cat_correct.get(item.category, 0) + 1
                    brier_diffs.append((a.noul - e.noul) ** 2)
                case (ScoreAnswer() as a, ScoreAnswer() as e):
                    is_hit = (abs(a.score - e.score) < 0.5)
                    hits.append(1.0 if is_hit else 0.0)
                    if is_hit:
                        cat_correct[item.category] = cat_correct.get(item.category, 0) + 1
                case _:
                    hits.append(0.0)

            if total_items >= 250 and (idx % 250 == 0 or idx == total_items):
                p50 = float(statistics.median(latencies))
                running_acc = (sum(hits) / idx) * 100.0
                logger.info("[%s] %d/%d (%.0f%%) | Running Acc: %.1f%% | P50: %.1fms", engine_name, idx, total_items, idx * 100 / total_items, running_acc, p50)

        acc = sum(hits) / max(1, len(hits))
        p50 = float(statistics.median(latencies)) if latencies else 0.0
        brier = float(sum(brier_diffs) / max(1, len(brier_diffs))) if brier_diffs else 0.0
        cat_acc = {c: cat_correct.get(c, 0) / cat_total[c] for c in cat_total}

        logger.info("Completed %s: accuracy=%.2f%%, P50=%.2fms, brier=%.4f across %d cases", engine_name, acc * 100, p50, brier, len(hits))

        return DecisionMetrics(
            accuracy=acc,
            latency_p50_ms=p50,
            brier_score=brier,
            category_accuracy=cat_acc,
            category_counts=cat_total,
            count=len(hits),
        )
