from typing import Mapping, Sequence
from data.case import DecisionCase
from inference.questions import Choice
from benchmark.suites.suite import IBenchmarkSuite, fetch_cached_jsonl

KEV_TRANSFER_URL = "https://raw.githubusercontent.com/jaredpalmer/kev/main/evals/v4/transfer-v4/test.jsonl"

class KevTransferSuite(IBenchmarkSuite):
    """Jared Palmer's official Kev out-of-domain transfer evaluation suite (jaredpalmer/kev/evals/v4/transfer-v4)."""

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
        rows = fetch_cached_jsonl(KEV_TRANSFER_URL, "kev_transfer_v4_test.jsonl")
        cases: list[DecisionCase] = []

        for row in rows:
            state_raw = row.get("state", "")
            if isinstance(state_raw, dict):
                subj = state_raw.get("subject", "")
                q_text = state_raw.get("question", "")
                state = f"[{subj}] {q_text}" if subj else q_text
            else:
                state = str(state_raw)

            category = row.get("_meta", {}).get("source", "transfer")
            qs: list[Choice] = []

            for q_name, q_spec in row.get("questions", {}).items():
                criteria = q_spec.get("criteria", {})
                instruction = q_spec.get("instructions", "Select the best option:")
                label = q_spec.get("label")
                if criteria and label is not None:
                    qs.append(
                        Choice(
                            name=q_name,
                            options=criteria,
                            instruction=instruction,
                            target=label,
                        )
                    )

            if state and qs:
                cases.append(DecisionCase(state=state, questions=qs, category=category))

        return cases
