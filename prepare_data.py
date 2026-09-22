import json
import os
import random
from typing import Dict, List, Tuple

SUBJECTS = {
    0: ["credit card", "monthly invoice", "Stripe payment", "wire transfer", "annual subscription", "debit charge", "VAT receipt", "billing address", "Amex checkout", "refund request"],
    1: ["PostgreSQL replica", "Redis cache", "Kubernetes pod", "Docker container", "Nginx gateway", "Kafka consumer", "API latency", "microservice worker", "database pool", "memory usage"],
    2: ["enterprise contract", "volume discount", "annual SLA agreement", "team of 100 seats", "procurement review", "SOC2 report", "executive demo", "vendor onboarding", "custom pricing", "security review"],
    3: ["phishing email", "brute force login", "compromised API key", "unauthorized access", "SQL injection attempt", "malicious payload", "XSS vulnerability", "ransomware alert", "credential dump", "DDoS attack"]
}

PREDICATES = {
    0: [
        ("was charged twice unexpectedly", True, (80, 95)),
        ("failed with payment declined error", True, (70, 90)),
        ("needs to be updated before next cycle", False, (20, 45)),
        ("receipt is missing from our portal", False, (15, 40)),
        ("chargeback disputed by our bank", True, (85, 100))
    ],
    1: [
        ("crashed with fatal out of memory error", True, (85, 100)),
        ("latency spiked above five seconds", True, (75, 95)),
        ("connection pool exhausted under load", True, (80, 95)),
        ("scheduled maintenance window configuration", False, (10, 30)),
        ("minor warning log regarding deprecation", False, (5, 25))
    ],
    2: [
        ("ready for legal signature and procurement", False, (25, 50)),
        ("requesting immediate executive escalation for renewal", True, (70, 90)),
        ("inquiry about multi-year commitment terms", False, (20, 45)),
        ("evaluating competitive vendor proposals", False, (30, 55)),
        ("formal request for security audit clearance", False, (35, 60))
    ],
    3: [
        ("detected from unknown Russian IP address", True, (90, 100)),
        ("leaked in public GitHub repository", True, (95, 100)),
        ("blocked automatically by firewall rules", False, (40, 65)),
        ("suspicious employee account activity reported", True, (80, 95)),
        ("critical zero-day exploit attempt in logs", True, (95, 100))
    ]
}

TEMPLATES = [
    "Our {subject} {predicate}. Please review immediately.",
    "Notice regarding {subject}: {predicate}.",
    "We have an issue where our {subject} {predicate}.",
    "Urgent inquiry: the {subject} {predicate}.",
    "Can you check why the {subject} {predicate}?"
]

def generate_split(count: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    records = []
    for _ in range(count):
        dept = rng.randint(0, 3)
        subject = rng.choice(SUBJECTS[dept])
        pred_text, is_urgent_default, (min_s, max_s) = rng.choice(PREDICATES[dept])
        template = rng.choice(TEMPLATES)

        state = template.format(subject=subject, predicate=pred_text)
        urgent = 1.0 if is_urgent_default else 0.0
        score = rng.uniform(min_s, max_s) / 100.0

        records.append({
            "state": state,
            "questions": [
                ["department", "choice", dept],
                ["is_urgent", "noul", urgent],
                ["severity", "score", round(score, 3)]
            ]
        })
    return records

def main():
    os.makedirs("data", exist_ok=True)
    
    print("Generating training, validation, and standardized test bed...")
    train_data = generate_split(20000, seed=1001)
    val_data = generate_split(2000, seed=2002)
    test_data = generate_split(1000, seed=9999)

    for path, data in [("data/train.jsonl", train_data), ("data/val.jsonl", val_data), ("data/test.jsonl", test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")
        print(f"  --> Wrote {len(data)} records to {path}")

if __name__ == "__main__":
    main()
