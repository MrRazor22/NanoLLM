from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union
import torch
from torch.utils.data import DataLoader

from training.dataset.collator_policy import MultiQuestionCollator
from training.dataset.schema import DecisionSample
from training.dataset.sources.generic import GenericChoiceSource, ISourceAdapter
from training.dataset.sources.glaive import GlaiveToolSource
from training.dataset.sources.typed import TypedDecisionsSource
from training.dataset.transforms import (
    inject_abstention,
    load_jsonl,
    save_jsonl,
    split_train_val,
    to_decision_sample,
)

class ITrainingDataPolicy(Protocol):
    def get_loaders(
        self,
        collator: MultiQuestionCollator,
        max_tokens: int = 4000,
        batch_size: int = 16,
        pin_memory: bool = False,
    ) -> Tuple[DataLoader, DataLoader]: ...

class TrainingDataset(ITrainingDataPolicy):
    def __init__(
        self,
        train_path: Optional[Union[str, Path]] = None,
        val_path: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        sources: Optional[Sequence[ISourceAdapter]] = None,
        seed: int = 42,
        abstention_rate: float = 0.15,
        val_ratio: float = 0.08,
    ):
        self.train_path = Path(train_path) if train_path else None
        self.val_path = Path(val_path) if val_path else None
        self.data_dir = Path(data_dir or (self.train_path.parent if self.train_path else Path(".")))
        self.sources = list(sources) if sources is not None else self._default_sources()
        self.rng = random.Random(seed)
        self.abstention_rate = abstention_rate
        self.val_ratio = val_ratio

    def _default_sources(self) -> List[ISourceAdapter]:
        return [
            TypedDecisionsSource(repeat=3),
            GlaiveToolSource(limit=5000, rng=self.rng),
            GenericChoiceSource("mteb/amazon_massive_intent", "en", "train", "text", "label_text", "action", "What is the user's intent in this utterance?", 6000),
            GenericChoiceSource("fancyzhx/ag_news", None, "train", "text", "label", "label", "What is the topic of the article?", 6000, per_class_limit=1500),
            GenericChoiceSource("SetFit/enron_spam", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nMessage: {(r.get('message') or '')[:2000]}", lambda r: "spam" if int(r.get("label", 0)) == 1 else "legitimate", "is_spam", "Is this email unsolicited spam or bulk marketing?", 3000),
            GenericChoiceSource("zefang-liu/phishing-email-dataset", None, "train", lambda r: f"Email:\n{(r.get('Email Text') or '')[:2000]}", lambda r: "phishing" if r.get("Email Type") == "Phishing Email" else "safe", "is_phishing", "Is this email a phishing or scam attempt?", 3000),
            GenericChoiceSource("Tobi-Bueck/customer-support-tickets", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nBody: {(r.get('body') or '')[:2000]}", "queue", "queue", "Which support queue should handle this ticket?", 5000, filter_fn=lambda r: r.get("language") == "en" and r.get("body"), per_class_limit=500),
            GenericChoiceSource("dair-ai/emotion", "split", "train", "text", "label", "emotion", "Which emotion is most strongly expressed in text?", 3000),
            GenericChoiceSource("mteb/banking77", None, "train", "text", lambda r: str(r.get("label_text", "")).replace("_", " "), "intent", "What is the primary customer inquiry or banking request?", 6000),
            GenericChoiceSource("deepset/prompt-injections", None, "train", "text", lambda r: "quarantine_threat" if int(r.get("label", 0)) == 1 else "allow", "guardrail_action", "Determine the safety policy action for this user input.", 1000),
            GenericChoiceSource("jackhhao/jailbreak-classification", None, "train", "prompt", lambda r: "quarantine_threat" if str(r.get("type", "")).lower() == "jailbreak" else "allow", "guardrail_action", "Determine the safety policy action for this user input.", 1500),
        ]

    def load_samples(self) -> Tuple[List[DecisionSample], List[DecisionSample]]:
        if self.train_path and self.train_path.exists() and self.val_path and self.val_path.exists():
            return load_jsonl(self.train_path), load_jsonl(self.val_path)

        all_samples: List[Dict[str, Any]] = []
        for src in self.sources:
            all_samples.extend(src.extract())

        foundation_path = self.data_dir / "train.jsonl"
        if foundation_path.exists():
            with open(foundation_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.rng.shuffle(lines)
                for line in lines[:5000]:
                    all_samples.append(json.loads(line))

        all_samples = inject_abstention(all_samples, rate=self.abstention_rate, rng=self.rng)
        raw_train, raw_val = split_train_val(all_samples, val_ratio=self.val_ratio, rng=self.rng)
        if self.train_path:
            save_jsonl(self.train_path, raw_train)
        if self.val_path:
            save_jsonl(self.val_path, raw_val)
        return [to_decision_sample(r) for r in raw_train], [to_decision_sample(r) for r in raw_val]

    def get_loaders(
        self,
        collator: MultiQuestionCollator,
        max_tokens: int = 4000,
        batch_size: int = 16,
        pin_memory: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        train_data, val_data = self.load_samples()
        train_cache = self.train_path.with_suffix(".train.cache") if self.train_path else None
        val_cache = self.val_path.with_suffix(".val.cache") if self.val_path else None
        train_loader = collator.pack_loader(train_data, max_tokens=max_tokens, batch_size=batch_size, cache_path=train_cache, pin_memory=pin_memory)
        val_loader = collator.pack_loader(val_data, max_tokens=max_tokens, batch_size=batch_size, cache_path=val_cache, shuffle=False, pin_memory=pin_memory)
        return train_loader, val_loader

# Backward compatibility aliases
DatasetBuilder = TrainingDataset
IDatasetBuilder = ITrainingDataPolicy

__all__ = ["TrainingDataset", "ITrainingDataPolicy", "DatasetBuilder", "IDatasetBuilder"]
