from typing import Mapping, Sequence
from data.case import DecisionCase
from inference.questions import Choice
from benchmark.suites.suite import IBenchmarkSuite, fetch_cached_jsonl

JEVBENCH_URL_BASE = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public"

class JevBenchSuite(IBenchmarkSuite):
    """TypeSafe AI / JevBench official benchmark tasks from Benchmark Heaven (fstandhartinger/jevbench)."""

    def __init__(self, track: str = "hard"):
        self.track = track

    @property
    def name(self) -> str:
        return f"jev_bench_{self.track}"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "Imajev-4B": 0.6737,
            "Plumb-4B": 0.6584,
            "Jev 1.13.0": 0.6329,
            "JevK5 v0.2.0": 0.6204,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        tracks_to_load = ["hard", "original"] if self.track == "all" else [self.track]
        cases: list[DecisionCase] = []

        for tr in tracks_to_load:
            url = f"{JEVBENCH_URL_BASE}/{tr}.jsonl"
            rows = fetch_cached_jsonl(url, f"jevbench_{tr}.jsonl")
            for row in rows:
                state = row.get("state", "")
                q_dict = row.get("question", {})
                expected = row.get("expected")
                labels = row.get("labels", [])
                criteria = q_dict.get("criteria", {})
                instruction = q_dict.get("instructions", "Select the best option:")

                if not criteria and labels:
                    criteria = {lbl: lbl for lbl in labels}

                if criteria and expected is not None:
                    target_key = expected
                    if expected not in criteria:
                        mapping = {"yes": "true", "no": "false", "true": "true", "false": "false"}
                        target_key = mapping.get(str(expected).lower(), expected)
                        if target_key not in criteria:
                            target_key = expected

                    q = Choice(
                        name="decision",
                        options=criteria,
                        instruction=instruction,
                        target=target_key,
                    )
                    family = row.get("family", tr)
                    cases.append(DecisionCase(state=state, questions=[q], category=family))
        return cases
