from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

from data.case import CaseMeta, DecisionCase
from data.sources.source import IDataSource
from data.sources.hf_cache import HFCache, IRawCachePolicy

@dataclass(frozen=True, slots=True)
class TextClassificationSpec:
    dataset: str
    category: str
    text_field: str = "text"
    label_field: str = "label"
    label_names: Sequence[str] | None = None
    subset: str | None = None
    splits: Sequence[str] = ("train", "test")

STANDARD_CLASSIFICATIONS: Sequence[TextClassificationSpec] = (
    TextClassificationSpec("mteb/banking77", "banking77", label_field="label_text"),
    TextClassificationSpec("fancyzhx/ag_news", "ag_news", label_names=("World", "Sports", "Business", "Sci/Tech")),
    TextClassificationSpec("dair-ai/emotion", "emotion", label_names=("sadness", "joy", "love", "anger", "fear", "surprise")),
    TextClassificationSpec("Tobi-Bueck/customer-support-tickets", "customer_support", text_field="body", label_field="queue", splits=("train",)),
    TextClassificationSpec("zefang-liu/phishing-email-dataset", "phishing_email", text_field="Email Text", label_field="Email Type", label_names=("Safe Email", "Phishing Email"), splits=("train",)),
    TextClassificationSpec("SetFit/enron_spam", "enron_spam", label_names=("ham", "spam"), label_field="label_text"),
    TextClassificationSpec("lmsys/toxic-chat", "toxic_chat", label_names=("benign", "jailbreak"), text_field="user_input", label_field="jailbreaking", subset="toxicchat0124"),
    TextClassificationSpec("SetFit/amazon_massive_intent_en-US", "amazon_massive", label_field="label_text"),
)

class ClassificationSource(IDataSource):
    def __init__(
        self,
        spec: Optional[TextClassificationSpec] = None,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
        cache: Optional[IRawCachePolicy] = None,
    ):
        self.spec = spec or STANDARD_CLASSIFICATIONS[0]
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.cache = cache or HFCache()

    def stream(self) -> Iterator[Sequence[DecisionCase]]:
        source_tag = f"{self.spec.dataset.replace('/', '_')}_{self.spec.subset}" if self.spec.subset else self.spec.dataset.replace("/", "_")

        if self.spec.label_names:
            all_options = tuple(self.spec.label_names)
        else:
            labels: set[str] = set()
            for s in self.spec.splits:
                limit = self.test_limit if s == "test" else self.train_limit
                for r in self.cache.stream(self.spec.dataset, split=s, subset=self.spec.subset, limit=limit):
                    if self.spec.label_field in r and r[self.spec.label_field] is not None:
                        labels.add(str(r[self.spec.label_field]).replace("_", " ").strip())
            all_options = tuple(sorted(labels))

        for split_name in self.spec.splits:
            limit = self.test_limit if split_name == "test" else self.train_limit
            is_test = split_name == "test"
            for r in self.cache.stream(self.spec.dataset, split=split_name, subset=self.spec.subset, limit=limit):
                text = str(r.get(self.spec.text_field, "")).strip()
                if not text or self.spec.label_field not in r or r[self.spec.label_field] is None:
                    continue
                raw_val = r[self.spec.label_field]
                if self.spec.label_names and isinstance(raw_val, int) and 0 <= raw_val < len(self.spec.label_names):
                    label_str = self.spec.label_names[raw_val]
                else:
                    label_str = str(raw_val).replace("_", " ").strip()

                options = all_options
                if label_str not in options:
                    options = tuple(sorted(set(options + (label_str,))))

                probs = {c: (1.0 if c == label_str else 0.0) for c in options}
                yield (
                    DecisionCase(
                        state=text,
                        candidates=options,
                        instruction="",
                        probabilities=probs,
                        meta=CaseMeta(
                            candidate_type="choice",
                            category=self.spec.category,
                            source=source_tag,
                            split="test" if is_test else "train",
                        ),
                    ),
                )

__all__ = ["TextClassificationSpec", "ClassificationSource"]
