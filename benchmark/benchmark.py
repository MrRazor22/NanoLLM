import logging
from typing import Mapping, Optional, Protocol, Sequence
from benchmark.policies.scorecard import IScorecardPolicy, ScorecardPolicy
from benchmark.policies.scorer import DecisionMetrics, IScorerPolicy, ScorerPolicy
from benchmark.result import BenchmarkResult
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import IDecisionEngine

logger = logging.getLogger(__name__)

class IBenchmark(Protocol):
    def run(
        self,
        suites: Mapping[str, IBenchmarkSuite],
        engines: Mapping[str, IDecisionEngine],
    ) -> Sequence[BenchmarkResult]: ...

class Benchmark(IBenchmark):
    def __init__(
        self,
        scorer: Optional[IScorerPolicy] = None,
        scorecard: Optional[IScorecardPolicy] = None,
        limit: Optional[int] = None,
        force: bool = False,
    ):
        self.scorecard = scorecard or ScorecardPolicy()
        self.scorer = scorer or ScorerPolicy()
        self.limit = limit
        self.force = force

    def run(
        self,
        suites: Mapping[str, IBenchmarkSuite],
        engines: Mapping[str, IDecisionEngine],
    ) -> Sequence[BenchmarkResult]:
        results: list[BenchmarkResult] = []

        for suite_name, suite in suites.items():
            items = list(suite.test_cases())
            if self.limit:
                items = items[:self.limit]

            cached_map = {r.engine: r.metrics for r in self.scorecard.load(suites=[suite_name])}

            for label, engine in engines.items():
                if not self.force and label in cached_map and cached_map[label].count == len(items):
                    logger.info("Using cached scorecard entry for '%s' [%s]", suite_name, label)
                    metrics = cached_map[label]
                else:
                    metrics = self.scorer.score(engine, items)
                    self.scorecard.record(BenchmarkResult(suite=suite_name, engine=label, metrics=metrics))

                results.append(BenchmarkResult(suite=suite_name, engine=label, metrics=metrics))

            logger.info("Suite '%s' evaluated across %d engines (%d cases)", suite_name, len(engines), len(items))

        return results
