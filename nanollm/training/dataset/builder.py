import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from nanollm.training.dataset.sources.generic import GenericChoiceSource, ISourceAdapter
from nanollm.training.dataset.sources.glaive import GlaiveToolSource
from nanollm.training.dataset.sources.typed import TypedDecisionsSource
from nanollm.training.dataset.transforms import inject_abstention, save_jsonl, split_train_val

class IDatasetBuilder(Protocol):
    """The bedrock contract of the Dataset boundary."""
    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ...

class DatasetBuilder(IDatasetBuilder):
    def __init__(
        self,
        data_dir: Union[str, Path],
        sources: Optional[Sequence[ISourceAdapter]] = None,
        seed: int = 42,
        abstention_rate: float = 0.15,
        val_ratio: float = 0.08,
    ):
        self.data_dir = Path(data_dir)
        self.rng = random.Random(seed)
        self.abstention_rate = abstention_rate
        self.val_ratio = val_ratio
        self.sources = list(sources) if sources is not None else self._default_sources()

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

    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        return split_train_val(all_samples, val_ratio=self.val_ratio, rng=self.rng)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build curriculum datasets")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "data"))
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    builder = DatasetBuilder(out_p)
    train_recs, val_recs = builder.build()
    save_jsonl(out_p / "train_adapt.jsonl", train_recs)
    save_jsonl(out_p / "val_adapt.jsonl", val_recs)
    print(f"Built curriculum: {len(train_recs)} train, {len(val_recs)} val -> {out_p}")
