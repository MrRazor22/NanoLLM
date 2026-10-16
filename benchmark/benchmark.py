from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable
from benchmark.policies.ledger import DiskLedgerPolicy, ILedgerPolicy
from benchmark.policies.reporter import ConsoleReporterPolicy, IReporterPolicy
from benchmark.policies.scorer import IScorerPolicy, ScorerPolicy
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import IDecisionEngine

@runtime_checkable
class IBenchmark(Protocol):
    """Core Primitive Contract: Sovereign operational boundary for benchmark evaluation and reporting."""

    def run(
        self,
        suites: Sequence[IBenchmarkSuite],
        engines: Mapping[str, IDecisionEngine],
        limit: Optional[int] = None,
        skip_cached: bool = False,
    ) -> None: ...

    def report(self, suites: Optional[Sequence[IBenchmarkSuite]] = None) -> None: ...
    def playground(self, engines: Sequence[tuple[str, IDecisionEngine]]) -> None: ...

class Benchmark(IBenchmark):
    """Core Primitive Implementation: Sovereign coordinator of benchmark suites, engines, and policies."""

    def __init__(
        self,
        scorer: Optional[IScorerPolicy] = None,
        ledger: Optional[ILedgerPolicy] = None,
        reporter: Optional[IReporterPolicy] = None,
    ):
        self.ledger = ledger or DiskLedgerPolicy()
        self.scorer = scorer or ScorerPolicy()
        self.reporter = reporter or ConsoleReporterPolicy()

    def run(
        self,
        suites: Sequence[IBenchmarkSuite],
        engines: Mapping[str, IDecisionEngine],
        limit: Optional[int] = None,
        skip_cached: bool = False,
    ) -> None:
        active_names: list[str] = []
        for suite in suites:
            try:
                items = list(suite.load_cases())
            except FileNotFoundError as e:
                print(f"\n[UNAVAILABLE] Suite '{suite.name.upper()}': {e}")
                continue

            active_names.append(suite.name)
            if limit:
                items = items[:limit]

            print(f"\n{'=' * 65}\nEvaluating Suite: {suite.name.upper()} ({len(items)} items)")
            if suite.published_baselines:
                print("Published Baselines:")
                for b_name, b_score in suite.published_baselines.items():
                    print(f"  {b_name:<24}: {b_score * 100:.1f}%")
            print(f"{'=' * 65}")

            for label, engine in engines.items():
                if skip_cached and self.ledger.get(label, suite=suite.name, expected_count=len(items)) is not None:
                    print(f"[{label}] Cached run found for '{suite.name}'. Skipping.")
                    continue
                try:
                    report = self.scorer.score(
                        engine,
                        items,
                        model_name=label,
                        suite=suite.name,
                    )
                    self.ledger.put(report)
                except Exception as e:
                    print(f"Skipping run for '{label}': {e}")

            self.reporter.suite(self.ledger.all(suite.name), suite.name)

        if len(active_names) > 1:
            self.reporter.summary(self.ledger.all(), suites=active_names)

    def report(self, suites: Optional[Sequence[IBenchmarkSuite]] = None) -> None:
        suite_names = list(dict.fromkeys(s.name for s in suites)) if suites else list(dict.fromkeys(s for s, d in self.ledger.all().items() if isinstance(d, dict)))
        for s_name in suite_names:
            self.reporter.suite(self.ledger.all(s_name), s_name)

        if len(suite_names) > 1:
            self.reporter.summary(self.ledger.all(), suites=suite_names)

    def playground(self, engines: Sequence[tuple[str, IDecisionEngine]]) -> None:
        self.reporter.playground(engines)
