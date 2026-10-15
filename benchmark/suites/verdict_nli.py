from typing import Mapping, Sequence
from data.case import DecisionCase
from benchmark.suites.suite import IBenchmarkSuite

class VerdictNliSuite(IBenchmarkSuite):
    """Verdict 2.0 (ModernBERT-151M) multi-candidate decision benchmark."""

    @property
    def name(self) -> str:
        return "verdict_nli"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Verdict-151M": 0.660,
            "GLiClass-ModernBERT": 0.635,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        from data.storage_layer import DiskStorageLayer
        cases = DiskStorageLayer().load("test")
        nli_cats = {"adversarial_reasoning", "causal_reasoning", "deductive_reasoning"}
        selected = [c for c in cases if c.category in nli_cats]
        return selected if selected else cases[:400]
