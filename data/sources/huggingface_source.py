from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence
from datasets import load_dataset

from data.case import DecisionCase, IDataSource, SourceRecord
from inference.questions import Choice

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

def read_raw_jsonl(path: Path, fetcher: Optional[Callable[[], List[dict]]] = None) -> List[dict]:
    """Reads raw JSONL dataset from disk; fetches from source if absent."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if fetcher is not None:
        rows = list(fetcher())
        if rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
        return rows
    return []

@dataclass(frozen=True)
class BenchmarkSpec:
    """Core Domain DTO: Declarative specification for benchmark decision tasks."""
    dataset: str
    question: str = "decision"
    instruction: str = "Select the best option:"
    category: str = "general"
    text_field: str = "text"
    label_field: str = "label"
    label_names: Optional[Sequence[str]] = None
    subset: Optional[str] = None
    splits: Sequence[str] = ("train", "test")

    @property
    def signature(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:10]

STANDARD_BENCHMARKS: Sequence[BenchmarkSpec] = (
    BenchmarkSpec("mteb/banking77", "intent", "What is the primary customer request?", "banking77", label_field="label_text"),
    BenchmarkSpec("fancyzhx/ag_news", "topic", "What is the topic of article?", "ag_news", label_names=("World", "Sports", "Business", "Sci/Tech")),
    BenchmarkSpec("dair-ai/emotion", "emotion", "Which emotion is most strongly expressed?", "emotion", label_names=("sadness", "joy", "love", "anger", "fear", "surprise")),
    BenchmarkSpec("Tobi-Bueck/customer-support-tickets", "queue", "Which support queue should handle this ticket?", "customer_support", text_field="body", label_field="queue", splits=("train",)),
    BenchmarkSpec("zefang-liu/phishing-email-dataset", "is_phishing", "Is this email a phishing or scam attempt?", "phishing_email", text_field="Email Text", label_field="Email Type", label_names=("Safe Email", "Phishing Email"), splits=("train",)),
    BenchmarkSpec("SetFit/enron_spam", "is_spam", "Is this email unsolicited spam or bulk marketing?", "enron_spam", label_names=("ham", "spam"), label_field="label_text"),
    BenchmarkSpec("lmsys/toxic-chat", "jailbreak", "Does prompt try to bypass system safety instructions?", "toxic_chat", label_names=("benign", "jailbreak"), text_field="user_input", label_field="jailbreaking", subset="toxicchat0124"),
    BenchmarkSpec("SetFit/amazon_massive_intent_en-US", "intent", "What is the user intent in this utterance?", "amazon_massive", label_field="label_text"),
)

class HuggingFaceSource(IDataSource):
    """Declarative benchmark dataset adapter: unifies train and test partitions under canonical shared candidate options."""

    def __init__(self, spec: Optional[BenchmarkSpec] = None):
        self.spec = spec or STANDARD_BENCHMARKS[0]
        self.name = self.spec.dataset.replace("/", "_")
        if self.spec.subset:
            self.name = f"{self.name}_{self.spec.subset}"

    @property
    def signature(self) -> str:
        return self.spec.signature

    def _load_split(self, split: str) -> List[dict]:
        target = RAW_DIR / f"{self.name}_{split}.jsonl"
        def _fetch() -> List[dict]:
            args = [self.spec.dataset]
            if self.spec.subset:
                args.append(self.spec.subset)
            return list(load_dataset(*args, split=split))
        return read_raw_jsonl(target, _fetch)

    def read(self) -> Sequence[SourceRecord]:
        split_rows = {s: self._load_split(s) for s in self.spec.splits}

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

        records: List[SourceRecord] = []
        for split_name, rows in split_rows.items():
            tag = "test" if split_name == "test" else "train"
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

                case = DecisionCase(
                    state=text[:2000],
                    questions=[Choice(name=self.spec.question, options=options, instruction=self.spec.instruction, target=label_str)],
                    category=self.spec.category,
                )
                records.append(SourceRecord(sample=case, tag=tag))

        return records

__all__ = ["BenchmarkSpec", "STANDARD_BENCHMARKS", "HuggingFaceSource", "read_raw_jsonl", "RAW_DIR"]
