from typing import Mapping, Sequence
from data.case import DecisionCase
from data.storage_layer import DiskStorageLayer
from benchmark.suites.suite import IBenchmarkSuite

class FastDecisionsSuite(IBenchmarkSuite):
    """Fastino Labs GLiNER2.5-Decide published 17-task operational benchmark."""

    @property
    def name(self) -> str:
        return "fast_decisions"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "GLiNER2.5-Decide": 0.601,
            "JevK5": 0.575,
            "Laya": 0.570,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        cases = DiskStorageLayer().load("test")
        ops_cats = {"customer_support", "banking77", "function_calling", "phishing_email", "enron_spam", "toxic_chat"}
        selected = [c for c in cases if c.category in ops_cats]
        return selected if selected else cases[:500]

# Backward-compatible alias
GlinerFastDecisionsSuite = FastDecisionsSuite
