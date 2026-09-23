from pathlib import Path
from typing import Dict, List, Tuple
import json
import os
import random
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent

SYNTHETIC_TEMPLATES = [
    ("category", [
        ("Patient presents with sudden onset unilateral facial droop, slurred speech, and right arm weakness", "stroke_emergency", True, 0.95),
        ("Severe itchy red maculopapular rash developing across torso after taking amoxicillin dose", "drug_allergy_reaction", True, 0.85),
        ("Twisted right ankle during basketball, severe localized swelling and inability to bear weight", "orthopedic_injury", False, 0.40),
        ("Blood pressure reading 195/120 with severe headache and blurred vision", "hypertension_crisis", True, 0.90),
        ("Deep laceration on forearm from broken glass with steady venous bleeding", "wound_suture_repair", True, 0.75),
        ("Mild seasonal sneezing, clear rhinorrhea, and itchy watery eyes for past two days", "allergic_rhinitis", False, 0.15),
        ("Neither party shall disclose confidential trade secrets, customer lists, or proprietary source code", "confidentiality_nda", False, 0.30),
        ("This agreement shall be governed by and construed in accordance with the laws of the State of Delaware", "governing_law_jurisdiction", False, 0.20),
        ("Supplier shall defend, indemnify, and hold harmless customer against third-party patent claims", "indemnification_defense", True, 0.80),
        ("If any provision of this agreement is held unenforceable, remaining provisions remain in full effect", "severability_clause", False, 0.10),
        ("Either party may terminate immediately upon written notice if the other party breaches material terms", "termination_for_cause", True, 0.70),
        ("Need to save uncommitted working directory changes temporarily to pull main branch cleanly", "git_stash", False, 0.25),
        ("Apply commit 7a8b9c from develop branch directly onto the release-1.2 maintenance branch", "git_cherry_pick", False, 0.35),
        ("Discard all unstaged and staged changes completely and revert back to commit HEAD", "git_reset_hard", True, 0.85),
        ("Reapply local commits on top of updated upstream main branch for linear commit history", "git_rebase", False, 0.40),
        ("Permanently remove untracked build artifacts and ignore files from the working tree", "git_clean", False, 0.30),
        ("Alert: Kafka consumer lag exceeded 500k messages on events-topic due to crash loop", "kafka_consumer_lag", True, 0.90),
        ("PostgreSQL replica replication delay increased beyond 120 seconds due to long running transaction", "postgres_replication_delay", True, 0.85),
        ("Kubernetes pod evicted due to memory pressure exceeding container limit of 8Gi", "kubernetes_oom_eviction", True, 0.92),
        ("Scheduled SSL certificate rotation completed successfully for api gateway endpoints", "certificate_rotation", False, 0.15),
        ("Redis cluster memory usage reached 88% capacity, triggering key eviction policy", "redis_cache_eviction", False, 0.55),
        ("Detected brute force SSH login attempts from multiple unauthorized external IP addresses", "brute_force_attack", True, 0.95),
        ("AWS IAM root access key utilized from unrecognized geographical location", "compromised_credentials", True, 0.98),
        ("WAF blocked malicious payload matching SQL injection signature on checkout endpoint", "sql_injection_attempt", False, 0.60),
        ("Internal service account token leaked in public GitHub repository commit history", "secret_leak_incident", True, 0.95),
        ("Scheduled vulnerability scan detected unpatched CVE-2026-4412 on base docker image", "vulnerability_patching", False, 0.45)
    ])
]

def build_synthetic_samples(rng: random.Random, count: int) -> List[Dict]:
    rows = SYNTHETIC_TEMPLATES[0][1]
    all_labels = [row[1] for row in rows]
    records = []
    for _ in range(count):
        text, target_label, urgent, sev = rng.choice(rows)
        distractors = rng.sample([l for l in all_labels if l != target_label], 3)
        options = [target_label] + distractors
        rng.shuffle(options)
        noise = rng.uniform(-0.04, 0.04)
        records.append({
            "state": text,
            "questions": [
                ["category", "choice", options.index(target_label), options],
                ["is_urgent", "noul", 1.0 if urgent else 0.0],
                ["severity", "score", round(min(1.0, max(0.0, sev + noise)), 3)]
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
            distractors = rng.sample([l for l in labels if l != label], 3)
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
            distractors = rng.sample([l for l in labels if l != label], 3)
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
    return splits["train"][:3000], splits["test"][:500]

def main():
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    b_tr, b_te = build_banking_samples(rng)
    m_tr, m_val, m_te = build_massive_samples(rng)
    s_tr, s_te = build_sentiment_samples(rng)
    syn_tr = build_synthetic_samples(rng, 12000)
    syn_val = build_synthetic_samples(rng, 1000)
    syn_te = build_synthetic_samples(rng, 1000)

    train_all = b_tr + m_tr + s_tr + syn_tr
    rng.shuffle(train_all)

    val_all = m_val + s_te + syn_val
    rng.shuffle(val_all)

    test_all = b_te + m_te + syn_te
    rng.shuffle(test_all)

    for fname, d in [("train.jsonl", train_all), ("val.jsonl", val_all), ("test.jsonl", test_all)]:
        path = data_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            for s in d:
                f.write(json.dumps(s) + "\n")
        print(f"Wrote {len(d)} records to {path}")

if __name__ == "__main__":
    main()
