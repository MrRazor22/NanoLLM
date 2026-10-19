from dataclasses import dataclass
from typing import List, Optional, Sequence

from data.case import DecisionCase
from data.sources.source import IDataSource, RAW_DIR, load_raw_jsonl

@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Core Domain DTO: Declarative specification for benchmark decision tasks."""
    dataset: str
    question: str = "decision"
    instruction: str = "Select the best option:"
    category: str = "general"
    text_field: str = "text"
    label_field: str = "label"
    label_names: Sequence[str] | None = None
    subset: str | None = None
    splits: Sequence[str] = ("train", "test")
    train_limit: int | None = None
    test_limit: int | None = 400

STANDARD_BENCHMARKS: Sequence[BenchmarkSpec] = (
    BenchmarkSpec("mteb/banking77", "intent", "What is the primary customer request?", "banking77", label_field="label_text"),
    BenchmarkSpec("fancyzhx/ag_news", "topic", "What is the topic of article?", "ag_news", label_names=("World", "Sports", "Business", "Sci/Tech"), train_limit=1000),
    BenchmarkSpec("dair-ai/emotion", "emotion", "Which emotion is most strongly expressed?", "emotion", label_names=("sadness", "joy", "love", "anger", "fear", "surprise"), train_limit=1000),
    BenchmarkSpec("Tobi-Bueck/customer-support-tickets", "queue", "Which support queue should handle this ticket?", "customer_support", text_field="body", label_field="queue", splits=("train",)),
    BenchmarkSpec("zefang-liu/phishing-email-dataset", "is_phishing", "Is this email a phishing or scam attempt?", "phishing_email", text_field="Email Text", label_field="Email Type", label_names=("Safe Email", "Phishing Email"), splits=("train",), train_limit=1000),
    BenchmarkSpec("SetFit/enron_spam", "is_spam", "Is this email unsolicited spam or bulk marketing?", "enron_spam", label_names=("ham", "spam"), label_field="label_text", train_limit=1000),
    BenchmarkSpec("lmsys/toxic-chat", "jailbreak", "Does prompt try to bypass system safety instructions?", "toxic_chat", label_names=("benign", "jailbreak"), text_field="user_input", label_field="jailbreaking", subset="toxicchat0124", train_limit=1000),
    BenchmarkSpec("SetFit/amazon_massive_intent_en-US", "intent", "What is the user intent in this utterance?", "amazon_massive", label_field="label_text", train_limit=2000),
)

class HuggingFaceSource(IDataSource):
    """Declarative benchmark dataset adapter: unifies train and test partitions under canonical shared candidate options."""

    def __init__(self, spec: Optional[BenchmarkSpec] = None):
        self.spec = spec or STANDARD_BENCHMARKS[0]
        self.name = self.spec.dataset.replace("/", "_")
        if self.spec.subset:
            self.name = f"{self.name}_{self.spec.subset}"

    def cases(self) -> Sequence[DecisionCase]:
        split_rows = {
            s: load_raw_jsonl(
                f"{self.name}_{s}",
                self.spec.dataset,
                split=s,
                subset=self.spec.subset,
                limit=(self.spec.test_limit if s == "test" else self.spec.train_limit),
            )
            for s in self.spec.splits
        }

        if self.spec.label_names:
            all_options = list(self.spec.label_names)
        else:
            all_rows = [r for rows in split_rows.values() for r in rows]
            labels = {
                str(r[self.spec.label_field]).replace("_", " ").strip()
                for r in all_rows
                if self.spec.label_field in r and r[self.spec.label_field] is not None
            }
            all_options = sorted(list(labels))

        cases: List[DecisionCase] = []
        for split_name, rows in split_rows.items():
            is_test = split_name == "test"
            for r in rows:
                text = str(r.get(self.spec.text_field) or r.get("prompt") or r.get("body") or "").strip()
                if not text or self.spec.label_field not in r or r[self.spec.label_field] is None:
                    continue
                raw_val = r[self.spec.label_field]
                if self.spec.label_names and isinstance(raw_val, int) and 0 <= raw_val < len(self.spec.label_names):
                    label_str = self.spec.label_names[raw_val]
                else:
                    label_str = str(raw_val).replace("_", " ").strip()

                options = all_options
                if label_str not in options:
                    options = sorted(list(set(options + [label_str])))

                cands_tuple = tuple(options)
                probs = {c: (1.0 if c == label_str else 0.0) for c in cands_tuple}

                case = DecisionCase(
                    state=text[:2000],
                    candidates=cands_tuple,
                    probabilities=probs,
                    category=self.spec.category,
                    source=self.name,
                    split="test" if is_test else "train",
                )
                cases.append(case)

        return cases
