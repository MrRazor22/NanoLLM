from typing import List, Sequence
from datasets import load_dataset

from data.case import DecisionCase, SourceRecord
from data.sources.source import IDataSource
from inference.questions import Choice
from data.sources.huggingface_source import RAW_DIR, read_raw_jsonl

NLI_OPTIONS = ["Entailment", "Neutral", "Contradiction"]
NLI_INSTRUCTION = "What is the logical relationship between the context and the statement?"

def _load_anli_split(round_name: str, split_type: str, limit: int) -> List[dict]:
    hf_split = f"{split_type}_{round_name}"
    target = RAW_DIR / f"anli_{hf_split}.jsonl"

    def _fetch() -> List[dict]:
        rows = list(load_dataset("facebook/anli", split=hf_split))
        if limit is not None:
            rows = rows[:limit]
        return rows

    rows = read_raw_jsonl(target, _fetch)
    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
    return rows

class ANLISource(IDataSource):
    """Adversarial NLI (ANLI) dataset source: extracts human-crafted counterfactuals and refutations."""
    name = "facebook_anli"

    def __init__(
        self,
        rounds: Sequence[str] = ("r1", "r2"),
        train_limit_per_round: int = 1500,
        test_limit_per_round: int = 200,
    ):
        self.rounds = list(rounds)
        self.train_limit = train_limit_per_round
        self.test_limit = test_limit_per_round

    def read(self) -> Sequence[SourceRecord]:
        records: List[SourceRecord] = []
        for r_name in self.rounds:
            for split_type, tag in [("train", "train"), ("test", "test")]:
                limit = self.test_limit if split_type == "test" else self.train_limit
                rows = _load_anli_split(r_name, split_type, limit)
                for r in rows:
                    premise = str(r.get("premise", "")).strip()
                    hypothesis = str(r.get("hypothesis", "")).strip()
                    label = r.get("label")

                    if not premise or not hypothesis or label not in (0, 1, 2):
                        continue

                    state = f"Context: {premise}\n\nStatement: {hypothesis}"
                    case = DecisionCase(
                        state=state,
                        questions=[
                            Choice(
                                name="relation",
                                options=list(NLI_OPTIONS),
                                instruction=NLI_INSTRUCTION,
                                target=NLI_OPTIONS[int(label)],
                            )
                        ],
                        category="adversarial_reasoning",
                    )
                    records.append(SourceRecord(sample=case, tag=tag))

        return records
