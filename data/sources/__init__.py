from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence, runtime_checkable

from inference.case import DecisionCase

DATA_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATA_DIR / "raw"
ADAPTED_DIR = DATA_DIR / "adapted"

@dataclass(frozen=True)
class SourceRecord:
    sample: DecisionCase
    tag: str = "unsplit"

@runtime_checkable
class IDataSource(Protocol):
    name: str
    def read(self) -> Sequence[SourceRecord]: ...

def cached_records(path: Path, fetcher: Callable[[], Sequence[SourceRecord]]) -> Sequence[SourceRecord]:
    if path.exists():
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                records.append(SourceRecord(sample=DecisionCase.from_dict(d), tag=d.get("tag", "unsplit")))
        return records
    records = list(fetcher())
    if records:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({**r.sample.to_dict(), "tag": r.tag}, default=str) for r in records]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records

def cached_jsonl(path: Path, fetcher: Callable[[], List[dict]]) -> List[dict]:
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = fetcher()
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    return rows

from data.sources.glaive_source import GlaiveSource
from data.sources.decisions_source import DecisionsSource
from data.sources.huggingface_source import HuggingFaceSource

def standard_sources(seed: int = 42) -> List[IDataSource]:
    """Default collection of diverse multi-task, multimodal-ready decision sources."""
    ag_news_opts = ["World", "Sports", "Business", "Sci/Tech"]
    ag_news_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech", "0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}
    emotion_opts = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    emotion_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise", "0": "sadness", "1": "joy", "2": "love", "3": "anger", "4": "fear", "5": "surprise"}
    binary_spam_opts = ["ham", "spam"]
    binary_spam_map = {0: "ham", 1: "spam", "0": "ham", "1": "spam"}
    toxic_opts = ["benign", "jailbreak"]
    toxic_map = {0: "benign", 1: "jailbreak", "0": "benign", "1": "jailbreak", False: "benign", True: "jailbreak"}

    return [
        GlaiveSource(limit=5000, seed=seed),
        DecisionsSource(repeat=5),
        HuggingFaceSource(
            "mteb/banking77", "intent", "What is the primary customer request?",
            limit=5000, split="train", label_field="label_text", tag="unsplit", category="banking77",
        ),
        HuggingFaceSource(
            "mteb/banking77", "intent", "What is the primary customer request?",
            limit=400, split="test", label_field="label_text", tag="test", category="banking77",
        ),
        HuggingFaceSource(
            "fancyzhx/ag_news", "topic", "What is the topic of article?",
            limit=5000, split="train", text_field="text", label_field="label", tag="unsplit", category="ag_news",
            candidate_options=ag_news_opts, label_map=ag_news_map,
        ),
        HuggingFaceSource(
            "fancyzhx/ag_news", "topic", "What is the topic of article?",
            limit=400, split="test", text_field="text", label_field="label", tag="test", category="ag_news",
            candidate_options=ag_news_opts, label_map=ag_news_map,
        ),
        HuggingFaceSource(
            "dair-ai/emotion", "emotion", "Which emotion is most strongly expressed?",
            limit=5000, split="train", text_field="text", label_field="label", tag="unsplit", category="emotion",
            candidate_options=emotion_opts, label_map=emotion_map,
        ),
        HuggingFaceSource(
            "dair-ai/emotion", "emotion", "Which emotion is most strongly expressed?",
            limit=400, split="test", text_field="text", label_field="label", tag="test", category="emotion",
            candidate_options=emotion_opts, label_map=emotion_map,
        ),
        HuggingFaceSource(
            "Tobi-Bueck/customer-support-tickets", "queue", "Which support queue should handle this ticket?",
            limit=4000, split="train", text_field="body", label_field="queue", tag="unsplit", category="customer_support",
        ),
        HuggingFaceSource(
            "zefang-liu/phishing-email-dataset", "is_phishing", "Is this email a phishing or scam attempt?",
            limit=4000, split="train", text_field="Email Text", label_field="Email Type", tag="unsplit", category="phishing_email",
            candidate_options=["Safe Email", "Phishing Email"],
        ),
        HuggingFaceSource(
            "SetFit/enron_spam", "is_spam", "Is this email unsolicited spam or bulk marketing?",
            limit=4000, split="train", text_field="text", label_field="label_text", tag="unsplit", category="enron_spam",
            candidate_options=binary_spam_opts, label_map=binary_spam_map,
        ),
        HuggingFaceSource(
            "SetFit/enron_spam", "is_spam", "Is this email unsolicited spam or bulk marketing?",
            limit=400, split="test", text_field="text", label_field="label_text", tag="test", category="enron_spam",
            candidate_options=binary_spam_opts, label_map=binary_spam_map,
        ),
        HuggingFaceSource(
            "lmsys/toxic-chat", "jailbreak", "Does prompt try to bypass system safety instructions?",
            limit=3000, split="train", text_field="user_input", label_field="jailbreaking", tag="unsplit", category="toxic_chat",
            candidate_options=toxic_opts, label_map=toxic_map,
        ),
        HuggingFaceSource(
            "lmsys/toxic-chat", "jailbreak", "Does prompt try to bypass system safety instructions?",
            limit=400, split="test", text_field="user_input", label_field="jailbreaking", tag="test", category="toxic_chat",
            candidate_options=toxic_opts, label_map=toxic_map,
        ),
        HuggingFaceSource(
            "SetFit/amazon_massive_intent_en-US", "intent", "What is the user intent in this utterance?",
            limit=5000, split="train", text_field="text", label_field="label_text", tag="unsplit", category="amazon_massive",
        ),
        HuggingFaceSource(
            "SetFit/amazon_massive_intent_en-US", "intent", "What is the user intent in this utterance?",
            limit=400, split="test", text_field="text", label_field="label_text", tag="test", category="amazon_massive",
        ),
    ]

__all__ = [
    "DATA_DIR", "RAW_DIR", "ADAPTED_DIR",
    "IDataSource", "SourceRecord",
    "cached_jsonl", "cached_records",
    "GlaiveSource", "DecisionsSource", "HuggingFaceSource", "standard_sources",
]
