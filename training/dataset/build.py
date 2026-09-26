import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from training.dataset.dataset import ADAPTED_DIR, DATA_DIR, RAW_DIR, IDataSource
from training.dataset.sources.generic import GenericChoiceSource
from training.dataset.sources.glaive import GlaiveToolSource
from training.dataset.sources.typed import TypedDecisionsSource
from training.dataset.transforms import inject_abstention, save_jsonl, split_train_val

SOURCES_ADAPTED_DIR = ADAPTED_DIR / "sources"

def get_default_sources(rng: Optional[random.Random] = None) -> List[IDataSource]:
    gen = rng or random.Random(42)
    return [
        TypedDecisionsSource(repeat=3),
        GlaiveToolSource(limit=5000, rng=gen),
        GenericChoiceSource("mteb/amazon_massive_intent", "en", "train", "text", "label_text", "action", "What is the user's intent in this utterance?", 6000, name="massive_intent"),
        GenericChoiceSource("fancyzhx/ag_news", None, "train", "text", "label", "label", "What is the topic of the article?", 6000, per_class_limit=1500, name="ag_news"),
        GenericChoiceSource("SetFit/enron_spam", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nMessage: {(r.get('message') or '')[:2000]}", lambda r: "spam" if int(r.get("label", 0)) == 1 else "legitimate", "is_spam", "Is this email unsolicited spam or bulk marketing?", 3000, name="enron_spam"),
        GenericChoiceSource("zefang-liu/phishing-email-dataset", None, "train", lambda r: f"Email:\n{(r.get('Email Text') or '')[:2000]}", lambda r: "phishing" if r.get("Email Type") == "Phishing Email" else "safe", "is_phishing", "Is this email a phishing or scam attempt?", 3000, name="phishing_email"),
        GenericChoiceSource("Tobi-Bueck/customer-support-tickets", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nBody: {(r.get('body') or '')[:2000]}", "queue", "queue", "Which support queue should handle this ticket?", 5000, filter_fn=lambda r: r.get("language") == "en" and r.get("body"), per_class_limit=500, name="customer_support"),
        GenericChoiceSource("dair-ai/emotion", "split", "train", "text", "label", "emotion", "Which emotion is most strongly expressed in text?", 3000, name="emotion"),
        GenericChoiceSource("mteb/banking77", None, "train", "text", lambda r: str(r.get("label_text", "")).replace("_", " "), "intent", "What is the primary customer inquiry or banking request?", 6000, name="banking77"),
        GenericChoiceSource("deepset/prompt-injections", None, "train", "text", lambda r: "quarantine_threat" if int(r.get("label", 0)) == 1 else "allow", "guardrail_action", "Determine the safety policy action for this user input.", 1000, name="prompt_injections"),
        GenericChoiceSource("jackhhao/jailbreak-classification", None, "train", "prompt", lambda r: "quarantine_threat" if str(r.get("type", "")).lower() == "jailbreak" else "allow", "guardrail_action", "Determine the safety policy action for this user input.", 1500, name="jailbreak_classification"),
    ]

def build_curriculum(
    sources: Optional[Sequence[IDataSource]] = None,
    seed: int = 42,
    abstention_rate: float = 0.15,
    val_ratio: float = 0.08,
    save_individual_sources: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    active_sources = list(sources) if sources is not None else get_default_sources(rng)

    SPLITS_DIR = DATA_DIR / "splits"
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTED_DIR.mkdir(parents=True, exist_ok=True)

    all_samples: List[Dict[str, Any]] = []

    for src in active_sources:
        src_samples = src.extract()
        print(f"  [Source] Extracted {len(src_samples)} items from {src.name}")
        if save_individual_sources:
            src_path = ADAPTED_DIR / f"{src.name}.jsonl"
            save_jsonl(src_path, src_samples)
        all_samples.extend(src_samples)

    # Check for raw foundation replay
    raw_foundation = RAW_DIR / "foundation_replay.jsonl"
    if not raw_foundation.exists():
        raw_foundation = RAW_DIR / "foundation_train.jsonl"
    if not raw_foundation.exists():
        raw_foundation = RAW_DIR / "train.jsonl"
    if raw_foundation.exists():
        with open(raw_foundation, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        rng.shuffle(lines)
        replay = [json.loads(line) for line in lines[:5000]]
        print(f"  [Foundation Replay] Added {len(replay)} samples from {raw_foundation.name}")
        if save_individual_sources:
            save_jsonl(ADAPTED_DIR / "foundation_replay.jsonl", replay)
        all_samples.extend(replay)

    print(f"Total raw extracted samples: {len(all_samples)}")
    all_samples = inject_abstention(all_samples, rate=abstention_rate, rng=rng)
    train_recs, val_recs = split_train_val(all_samples, val_ratio=val_ratio, rng=rng)

    train_path = SPLITS_DIR / "train.jsonl"
    val_path = SPLITS_DIR / "val.jsonl"
    save_jsonl(train_path, train_recs)
    save_jsonl(val_path, val_recs)

    print(f"Successfully adapted: {len(train_recs)} train -> {train_path}, {len(val_recs)} val -> {val_path}")
    return train_recs, val_recs

def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt raw sources into training datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--abstention-rate", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--no-sources", action="store_true", help="Skip saving individual source jsonl files")
    args = parser.parse_args()

    print(">>> Starting Training Dataset Adaptation...")
    build_curriculum(
        seed=args.seed,
        abstention_rate=args.abstention_rate,
        val_ratio=args.val_ratio,
        save_individual_sources=not args.no_sources,
    )

if __name__ == "__main__":
    main()
