from benchmark.suites.fast_decisions import FastDecisionsSuite
from benchmark.suites.jevbench import JevBenchSuite
from benchmark.suites.kev_transfer import KevTransferSuite
from benchmark.suites.laya_typed import LayaTypedDecisionsSuite
from benchmark.suites.suite import IBenchmarkSuite
from benchmark.suites.verdict_nli import VerdictNliSuite

PUBLISHED_SUITES = {
    "laya": LayaTypedDecisionsSuite,
    "kev": KevTransferSuite,
    "jev": JevBenchSuite,
    "gliner": FastDecisionsSuite,
    "verdict": VerdictNliSuite,
}

__all__ = [
    "IBenchmarkSuite",
    "PUBLISHED_SUITES",
    "FastDecisionsSuite",
    "JevBenchSuite",
    "KevTransferSuite",
    "LayaTypedDecisionsSuite",
    "VerdictNliSuite",
]
