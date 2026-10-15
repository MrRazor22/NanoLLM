from typing import Mapping, Sequence
from datasets import load_dataset
from data.case import DecisionCase
from data.sources.huggingface_source import RAW_DIR, read_raw_jsonl
from inference.questions import Choice
from benchmark.suites.suite import IBenchmarkSuite

FAST_DECISIONS_CONFIGS = (
    "support_intent",
    "support_topic",
    "document_type",
    "review_sentiment",
    "agent_handoff",
    "email_triage",
    "ticket_route",
    "product_feedback",
    "banking_intent",
    "clinic_request",
    "travel_request",
    "news_topic",
    "paper_field",
    "sports_recap",
    "restaurant_review",
    "benefits_request",
    "screen_tags",
)

class GlinerFastDecisionsSuite(IBenchmarkSuite):
    """Fastino Labs official published 17-domain operational decision benchmark (fastino/fast-decisions)."""

    def __init__(self, configs: Sequence[str] = FAST_DECISIONS_CONFIGS):
        self.configs = configs

    @property
    def name(self) -> str:
        return "gliner_fast_decisions"

    @property
    def published_baselines(self) -> Mapping[str, float]:
        return {
            "GLiNER2.5-Decide": 0.601,
            "JevK5": 0.575,
            "Laya": 0.570,
        }

    def load_cases(self) -> Sequence[DecisionCase]:
        cases: list[DecisionCase] = []
        for cfg in self.configs:
            target = RAW_DIR / f"fast_decisions_{cfg}.jsonl"
            rows = read_raw_jsonl(
                target,
                lambda c=cfg: list(load_dataset("fastino/fast-decisions", c, split="train")),
            )
            for row in rows:
                state = row.get("input", "")
                classifications = row.get("output", {}).get("classifications", [])
                qs: list[Choice] = []
                for cls in classifications:
                    task = cls.get("task", "decision")
                    labels = cls.get("labels", [])
                    true_labels = cls.get("true_label", [])
                    target_label = true_labels[0] if true_labels else None
                    if labels and target_label is not None:
                        qs.append(
                            Choice(
                                name=task,
                                options=labels,
                                instruction=f"Select the correct {task}:",
                                target=target_label,
                            )
                        )
                if state and qs:
                    cases.append(DecisionCase(state=state, questions=qs, category=cfg))
        return cases
