from typing import Dict, List, Tuple
import json
import os
import random
from datasets import load_dataset

SUBJECTS = {
    0: ["credit card", "monthly invoice", "Stripe payment", "wire transfer", "annual subscription"],
    1: ["PostgreSQL replica", "Redis cache", "Kubernetes pod", "Docker container", "Kafka consumer"],
    2: ["enterprise contract", "volume discount", "annual SLA agreement", "procurement review"],
    3: ["phishing email", "brute force login", "compromised API key", "unauthorized access", "ransomware alert"]
}
PREDICATES = {
    0: [("was charged twice unexpectedly", True, (0.8, 0.95)), ("needs to be updated", False, (0.2, 0.45))],
    1: [("crashed with out of memory error", True, (0.85, 1.0)), ("scheduled maintenance window", False, (0.1, 0.3))],
    2: [("ready for legal signature", False, (0.25, 0.5)), ("immediate escalation for renewal", True, (0.7, 0.9))],
    3: [("detected from unknown IP address", True, (0.9, 1.0)), ("blocked automatically by firewall", False, (0.4, 0.65))]
}
DEPT_NAMES = ["Billing", "Infrastructure", "Enterprise Sales", "Security"]

def build_tech_samples(rng: random.Random, count: int) -> List[Dict]:
    records = []
    for _ in range(count):
        dept = rng.randint(0, 3)
        subject = rng.choice(SUBJECTS[dept])
        pred, urgent, (s_min, s_max) = rng.choice(PREDICATES[dept])
        options = list(DEPT_NAMES)
        rng.shuffle(options)
        records.append({
            "state": f"Alert regarding {subject}: {pred}.",
            "questions": [
                ["domain", "choice", options.index(DEPT_NAMES[dept]), options],
                ["is_urgent", "noul", 1.0 if urgent else 0.0],
                ["severity", "score", round(rng.uniform(s_min, s_max), 3)]
            ]
        })
    return records

def build_banking_samples(rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    ds = load_dataset("mteb/banking77")
    labels = sorted(list(set(row["label_text"] for row in ds["train"])))
    splits = {}
    for split_name in ["train", "test"]:
        records = []
        for row in ds[split_name]:
            text, label = row["text"].strip(), row["label_text"]
            if not text:
                continue
            distractors = rng.sample([l for l in labels if l != label], 4)
            options = [label] + distractors
            rng.shuffle(options)
            urgent = 1.0 if any(k in text.lower() for k in ["stolen", "lost", "fraud", "decline", "fail"]) else 0.0
            records.append({
                "state": text,
                "questions": [
                    ["intent", "choice", options.index(label), options],
                    ["is_urgent", "noul", urgent],
                    ["severity", "score", round(rng.uniform(0.7, 0.95) if urgent else rng.uniform(0.1, 0.4), 3)]
                ]
            })
        splits[split_name] = records
    return splits["train"], splits["test"]

def build_massive_samples(rng: random.Random) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    ds = load_dataset("mteb/amazon_massive_intent", "en")
    labels = sorted(list(set(row["label_text"] for row in ds["train"])))
    splits = {}
    for split_name in ["train", "validation", "test"]:
        records = []
        for row in ds[split_name]:
            text, label = row["text"].strip(), row["label_text"]
            if not text:
                continue
            distractors = rng.sample([l for l in labels if l != label], 4)
            options = [label] + distractors
            rng.shuffle(options)
            is_cmd = 1.0 if any(text.lower().startswith(w) for w in ["set", "play", "call", "turn", "open", "mute", "send", "book", "order", "wake"]) else 0.0
            records.append({
                "state": text,
                "questions": [
                    ["action", "choice", options.index(label), options],
                    ["is_command", "noul", is_cmd],
                    ["priority", "score", round(rng.uniform(0.6, 0.9) if is_cmd else rng.uniform(0.2, 0.5), 3)]
                ]
            })
        splits[split_name] = records
    return splits["train"], splits["validation"], splits["test"]

def build_sentiment_samples(rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    ds = load_dataset("mteb/tweet_sentiment_extraction")
    splits = {}
    for split_name in ["train", "test"]:
        records = []
        for row in ds[split_name]:
            text, label = row["text"].strip(), row["label_text"]
            if not text:
                continue
            options = ["positive", "neutral", "negative"]
            rng.shuffle(options)
            is_neg = 1.0 if label == "negative" else 0.0
            score = rng.uniform(0.7, 1.0) if label == "positive" else (rng.uniform(0.4, 0.6) if label == "neutral" else rng.uniform(0.0, 0.3))
            records.append({
                "state": text,
                "questions": [
                    ["sentiment", "choice", options.index(label), options],
                    ["is_negative", "noul", is_neg],
                    ["polarity", "score", round(score, 3)]
                ]
            })
        splits[split_name] = records
    return splits["train"], splits["test"]

def main():
    os.makedirs("data", exist_ok=True)
    rng = random.Random(42)

    b_tr, b_te = build_banking_samples(rng)
    m_tr, m_val, m_te = build_massive_samples(rng)
    s_tr, s_te = build_sentiment_samples(rng)
    t_tr = build_tech_samples(rng, 20000)
    t_val = build_tech_samples(rng, 2000)
    t_te = build_tech_samples(rng, 1000)

    train_all = b_tr + m_tr + s_tr + t_tr
    rng.shuffle(train_all)

    val_all = m_val + s_te[:2000] + t_val
    rng.shuffle(val_all)

    test_all = b_te + m_te + s_te[2000:4000] + t_te
    rng.shuffle(test_all)

    for path, d in [("data/train.jsonl", train_all), ("data/val.jsonl", val_all), ("data/test.jsonl", test_all)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in d:
                f.write(json.dumps(s) + "\n")
        print(f"Wrote {len(d)} records to {path}")

if __name__ == "__main__":
    main()
