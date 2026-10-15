from typing import Mapping, Sequence
from data.case import DecisionCase
from data.storage_layer import DiskStorageLayer
from benchmark.suites.suite import IBenchmarkSuite

class KevTransferSuite(IBenchmarkSuite):
    """Kev flagship out-of-domain evaluation suite (SciQ, ANLI, Emotion, ARC)."""

    @property
    def name(self) -> str:
        return "kev_transfer"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Kev-4B": 0.789,
            "Jev (Baseline)": 0.812,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        cases = DiskStorageLayer().load("test")
        transfer_cats = {"deductive_reasoning", "scientific_reasoning", "emotion", "adversarial_reasoning"}
        selected = [c for c in cases if c.category in transfer_cats]
        return selected if selected else cases[:400]
