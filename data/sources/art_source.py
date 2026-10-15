from typing import List, Sequence
from datasets import load_dataset

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import RAW_DIR, read_raw_jsonl

ART_INSTRUCTION = "Which hypothesis best explains what happened between the observations?"

def _load_art_split(split: str, limit: int) -> List[dict]:
    target = RAW_DIR / f"art_{split}.jsonl"

    def _fetch() -> List[dict]:
        rows = list(load_dataset("allenai/art", split=split))
        if limit is not None:
            rows = rows[:limit]
        return rows

    rows = read_raw_jsonl(target, _fetch)
    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
    return rows

class ARTSource(IDataSource):
    """Abductive Reasoning in Natural Language (aNLI / ART) Source: deduces latent unstated hypotheses."""
    name = "allenai_art"

    def __init__(self, train_limit: int = 2000, test_limit: int = 300):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for split_name, tag in [("train", "train"), ("validation", "test")]:
            limit = self.test_limit if split_name == "validation" else self.train_limit
            rows = _load_art_split(split_name, limit)
            for r in rows:
                obs1 = str(r.get("observation_1", "")).strip()
                obs2 = str(r.get("observation_2", "")).strip()
                hyp1 = str(r.get("hypothesis_1", "")).strip()
                hyp2 = str(r.get("hypothesis_2", "")).strip()
                label_raw = r.get("label")

                if not obs1 or not obs2 or not hyp1 or not hyp2 or label_raw is None:
                    continue

                try:
                    raw_idx = int(label_raw)
                    target_idx = (raw_idx - 1) if raw_idx in (1, 2) else raw_idx
                except ValueError:
                    continue

                options = [hyp1, hyp2]
                if target_idx < 0 or target_idx >= len(options):
                    continue

                state = f"Observation 1: {obs1}\nObservation 2: {obs2}"
                target_str = options[target_idx]

                case = DecisionCase(
                    state=state,
                    questions=[
                        Choice(
                            name="explanation",
                            options=options,
                            instruction=ART_INSTRUCTION,
                            target=target_str,
                        )
                    ],
                    category="abductive_reasoning",
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["ARTSource", "ART_INSTRUCTION"]
