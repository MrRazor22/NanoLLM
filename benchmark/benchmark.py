from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence, Union
from benchmark.policies.ledger import DiskLedgerPolicy, ILedgerPolicy
from benchmark.policies.reporter import ConsoleReporterPolicy, IReporterPolicy
from benchmark.policies.scorer import IScorerPolicy, ScorerPolicy
from benchmark.suites.suite import IBenchmarkSuite
from inference.decision_engine import IDecisionEngine

class IBenchmark(Protocol):
    def run(
        self,
        suites: Mapping[str, IBenchmarkSuite],
        engines: Mapping[str, IDecisionEngine],
        limit: Optional[int] = None,
        skip_cached: bool = False,
    ) -> None: ...

    def report(self, suite_names: Optional[Sequence[str]] = None) -> None: ...

class Benchmark(IBenchmark):
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
        suites: Mapping[str, IBenchmarkSuite],
        engines: Mapping[str, IDecisionEngine],
        limit: Optional[int] = None,
        skip_cached: bool = False,
    ) -> None:
        active_names: list[str] = []
        for suite_name, suite in suites.items():
            try:
                items = list(suite.test_cases())
            except FileNotFoundError as e:
                print(f"\n[UNAVAILABLE] Suite '{suite_name.upper()}': {e}")
                continue

            active_names.append(suite_name)
            if limit:
                items = items[:limit]

            print(f"\n{'=' * 65}\nEvaluating Suite: {suite_name.upper()} ({len(items)} items)")
            print(f"{'=' * 65}")

            for label, engine in engines.items():
                if skip_cached and self.ledger.lookup(label, suite=suite_name, expected_count=len(items)) is not None:
                    print(f"[{label}] Cached run found for '{suite_name}'. Skipping.")
                    continue
                try:
                    report = self.scorer.score(
                        engine,
                        items,
                        model_name=label,
                        suite=suite_name,
                    )
                    self.ledger.record(report)
                except Exception as e:
                    print(f"Skipping run for '{label}': {e}")

            self.reporter.render_suite(self.ledger.entries(suite_name), suite_name)

        if len(active_names) > 1:
            self.reporter.render_summary(self.ledger.entries(), suites=active_names)

    def report(self, suite_names: Optional[Sequence[str]] = None) -> None:
        names = list(dict.fromkeys(suite_names)) if suite_names else list(dict.fromkeys(s for s, d in self.ledger.entries().items() if isinstance(d, dict)))
        for s_name in names:
            self.reporter.render_suite(self.ledger.entries(s_name), s_name)

        if len(names) > 1:
            self.reporter.render_summary(self.ledger.entries(), suites=names)
