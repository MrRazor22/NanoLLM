from typing import List, Sequence

from data.case import ChoiceAnswer, ChoiceOption, DecisionCase
from data.sources.source import IDataSource
from data.sources.huggingface_source import load_raw_jsonl

MEDQA_INSTRUCTION = "What is the most likely diagnosis, mechanism, or correct next step?"

class MedQASource(IDataSource):
    """Differential Elimination Source: extracts multi-variable clinical elimination decisions."""
    name = "gbaker_medqa"

    def __init__(self, train_limit: int = 2000, test_limit: int = 300):
        self.train_limit = train_limit
        self.test_limit = test_limit

    def read(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for split_name in ("train", "test"):
            limit = self.test_limit if split_name == "test" else self.train_limit
            rows = load_raw_jsonl(f"medqa_{split_name}", "GBaker/MedQA-USMLE-4-options", split=split_name, limit=limit)
            is_test = split_name == "test"
            for r in rows:
                state = str(r.get("question", "")).strip()
                opts_dict = r.get("options", {})
                answer_str = str(r.get("answer", "")).strip()

                if not state or not opts_dict or not answer_str:
                    continue

                options_list = list(opts_dict.values())
                if len(options_list) < 2 or answer_str not in options_list:
                    continue

                case = DecisionCase(
                    state=state,
                    option=ChoiceOption(
                        options=options_list,
                        answer=ChoiceAnswer(choice=answer_str, confidence=1.0, probabilities={answer_str: 1.0}),
                    ),
                    category="differential_elimination",
                    split="test" if is_test else "train",
                    source=self.name,
                )
                cases.append(case)

        return cases

__all__ = ["MedQASource", "MEDQA_INSTRUCTION"]
