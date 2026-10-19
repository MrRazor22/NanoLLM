from typing import List, Sequence

from data.case import ChoiceAnswer, ChoiceOption, DecisionCase
from data.sources.source import IDataSource
from data.sources.source import load_raw_jsonl

NLI_OPTIONS = ["Entailment", "Neutral", "Contradiction"]
NLI_INSTRUCTION = "What is the logical relationship between the context and the statement?"

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

    def cases(self) -> Sequence[DecisionCase]:
        cases: List[DecisionCase] = []
        for r_name in self.rounds:
            for split_type in ("train", "test"):
                limit = self.test_limit if split_type == "test" else self.train_limit
                hf_split = f"{split_type}_{r_name}"
                rows = load_raw_jsonl(f"anli_{hf_split}", "facebook/anli", split=hf_split, limit=limit)
                is_test = split_type == "test"
                for r in rows:
                    premise = str(r.get("premise", "")).strip()
                    hypothesis = str(r.get("hypothesis", "")).strip()
                    label = r.get("label")

                    if not premise or not hypothesis or label not in (0, 1, 2):
                        continue

                    lbl_str = NLI_OPTIONS[int(label)]
                    state = f"Premise: {premise}\nHypothesis: {hypothesis}"
                    case = DecisionCase(
                        state=state,
                        option=ChoiceOption(
                            options=list(NLI_OPTIONS),
                            answer=ChoiceAnswer(choice=lbl_str, confidence=1.0, probabilities={lbl_str: 1.0}),
                        ),
                        category="adversarial_reasoning",
                        split="test" if is_test else "train",
                        source=self.name,
                    )
                    cases.append(case)

        return cases
