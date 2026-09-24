from pathlib import Path
from typing import Any, Dict, List, Tuple
import json, os, random
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent

SYNTHETIC_ROWS = [
    ("Patient presents with sudden onset unilateral facial droop, slurred speech, and right arm weakness", "stroke_emergency", "Acute neurological deficit, hemiparesis, or facial droop requiring emergency CT", True, 0.95),
    ("Severe itchy red maculopapular rash developing across torso after taking amoxicillin dose", "drug_allergy_reaction", "Hypersensitivity drug reaction, allergic rash, or hives", True, 0.85),
    ("Twisted right ankle during basketball, severe localized swelling and inability to bear weight", "orthopedic_injury", "Musculoskeletal trauma, ligament tear, fracture, or joint sprain", False, 0.40),
    ("Blood pressure reading 195/120 with severe headache and blurred vision", "hypertension_crisis", "Severe hypertensive emergency with end-organ risk", True, 0.90),
    ("Deep laceration on forearm from broken glass with steady venous bleeding", "wound_suture_repair", "Open cutaneous laceration requiring primary suture closure", True, 0.75),
    ("Mild seasonal sneezing, clear rhinorrhea, and itchy watery eyes for past two days", "allergic_rhinitis", "Upper respiratory seasonal allergy symptoms", False, 0.15),
    ("Neither party shall disclose confidential trade secrets, customer lists, or proprietary source code", "confidentiality_nda", "Non-disclosure obligations protecting proprietary trade secrets", False, 0.30),
    ("This agreement shall be governed by and construed in accordance with the laws of the State of Delaware", "governing_law_jurisdiction", "Choice of law and legal jurisdiction clause", False, 0.20),
    ("Supplier shall defend, indemnify, and hold harmless customer against third-party patent claims", "indemnification_defense", "Indemnity obligation defending against third-party liabilities", True, 0.80),
    ("If any provision of this agreement is held unenforceable, remaining provisions remain in full effect", "severability_clause", "Severability clause preserving enforceable provisions", False, 0.10),
    ("Either party may terminate immediately upon written notice if the other party breaches material terms", "termination_for_cause", "Immediate contract cancellation due to material contractual breach", True, 0.70),
    ("Need to save uncommitted working directory changes temporarily to pull main branch cleanly", "git_stash", "Shelves and stashes uncommitted local modifications temporarily", False, 0.25),
    ("Apply commit 7a8b9c from develop branch directly onto the release-1.2 maintenance branch", "git_cherry_pick", "Applies specific commit patch onto target branch", False, 0.35),
    ("Discard all unstaged and staged changes completely and revert back to commit HEAD", "git_reset_hard", "Destructively discards local commits and working directory modifications", True, 0.85),
    ("Reapply local commits on top of updated upstream main branch for linear commit history", "git_rebase", "Replays local branch history on top of upstream branch", False, 0.40),
    ("Alert: Kafka consumer lag exceeded 500k messages on events-topic due to crash loop", "kafka_consumer_lag", "Message broker queue backlog exceeding processing thresholds", True, 0.90),
    ("PostgreSQL replica replication delay increased beyond 120 seconds due to long running transaction", "postgres_replication_delay", "Database read-replica delay or replication desynchronization", True, 0.85),
    ("Kubernetes pod evicted due to memory pressure exceeding container limit of 8Gi", "kubernetes_oom_eviction", "Container runtime killed due to out of memory limits", True, 0.92),
    ("Detected brute force SSH login attempts from multiple unauthorized external IP addresses", "brute_force_attack", "Repeated unauthorized authentication attempts from external attackers", True, 0.95),
    ("AWS IAM root access key utilized from unrecognized geographical location", "compromised_credentials", "High-privilege administrative access key anomaly", True, 0.98),
]

def build_synthetic_samples(rng: random.Random, count: int) -> List[Dict]:
    records = []
    for _ in range(count):
        text, target_label, desc, urgent, sev = rng.choice(SYNTHETIC_ROWS)
        distractor_rows = rng.sample([r for r in SYNTHETIC_ROWS if r[1] != target_label], 3)
        use_criteria = rng.random() > 0.4
        if use_criteria:
            crit_dict = {target_label: desc}
            for _, d_label, d_desc, _, _ in distractor_rows:
                crit_dict[d_label] = d_desc
            options_keys = list(crit_dict.keys())
            rng.shuffle(options_keys)
            options = {k: crit_dict[k] for k in options_keys}
            target_idx = list(options.keys()).index(target_label)
        else:
            options = [target_label] + [r[1] for r in distractor_rows]
            rng.shuffle(options)
            target_idx = options.index(target_label)

        records.append({
            "state": text,
            "questions": [
                ["category", "choice", target_idx, options],
                ["is_urgent", "noul", 1.0 if urgent else 0.0],
                ["severity", "score", round(sev + rng.uniform(-0.03, 0.03), 3)]
            ]
        })
    return records

def build_dataset_samples(dataset_name: str, sub: Any, text_k: str, label_k: str, q_name: str, rng: random.Random, limit: int) -> List[Dict]:
    ds = load_dataset(dataset_name, sub, split="train") if sub else load_dataset(dataset_name, split="train")
    labels = sorted(list(set(row[label_k] for row in ds)))
    records = []
    for row in ds:
        text, label = str(row[text_k]).strip(), str(row[label_k])
        if not text: continue
        distractors = rng.sample([l for l in labels if l != label], min(3, len(labels) - 1))
        options = [label] + distractors
        rng.shuffle(options)
        urgent = 1.0 if any(k in text.lower() for k in ["stolen", "lost", "fraud", "decline", "fail", "alert", "error"]) else 0.0
        records.append({
            "state": text,
            "questions": [
                [q_name, "choice", options.index(label), options],
                ["is_urgent", "noul", urgent],
                ["severity", "score", round(rng.uniform(0.7, 0.95) if urgent else rng.uniform(0.1, 0.4), 3)]
            ]
        })
        if len(records) >= limit: break
    return records

def main():
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    banking = build_dataset_samples("mteb/banking77", None, "text", "label_text", "intent", rng, 10000)
    massive = build_dataset_samples("mteb/amazon_massive_intent", "en", "text", "label_text", "action", rng, 12000)
    sentiment = build_dataset_samples("mteb/tweet_sentiment_extraction", None, "text", "label_text", "sentiment", rng, 3000)
    synthetic = build_synthetic_samples(rng, 15000)

    all_samples = banking + massive + sentiment + synthetic
    rng.shuffle(all_samples)

    n_val, n_test = int(0.1 * len(all_samples)), int(0.1 * len(all_samples))
    val_set, test_set, train_set = all_samples[:n_val], all_samples[n_val:n_val + n_test], all_samples[n_val + n_test:]

    for name, data in [("train.jsonl", train_set), ("val.jsonl", val_set), ("test.jsonl", test_set)]:
        path = data_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for item in data: f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(data)} samples to {path}")

if __name__ == "__main__":
    main()
