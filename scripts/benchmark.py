from pathlib import Path
from typing import Dict, List, Tuple
import sys
import time
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from nanollm import Choice, DecisionEngine, Noul, Score

BENCHMARK_SUITE: List[Dict] = [
    {
        "domain": "Medical / Clinical Triage",
        "query": "Patient presents with sudden onset unilateral facial droop, slurred speech, and right arm weakness.",
        "options": ["stroke_emergency", "dermatology_rash", "orthopedic_sprain", "routine_dental"],
        "expected": "stroke_emergency",
        "is_urgent": True,
    },
    {
        "domain": "Medical / Clinical Triage",
        "query": "Severe itchy red maculopapular rash developing across torso after taking amoxicillin dose.",
        "options": ["drug_allergy_reaction", "bone_fracture", "concussion", "hypertension"],
        "expected": "drug_allergy_reaction",
        "is_urgent": True,
    },
    {
        "domain": "Medical / Clinical Triage",
        "query": "Twisted right ankle during basketball, severe localized swelling and inability to bear weight.",
        "options": ["orthopedic_injury", "stroke_emergency", "cardiac_arrest", "food_poisoning"],
        "expected": "orthopedic_injury",
        "is_urgent": False,
    },
    {
        "domain": "Legal & Contracts",
        "query": "Neither party shall disclose confidential trade secrets, client lists, or algorithms to third parties.",
        "options": ["confidentiality_nda", "governing_law", "limitation_of_liability", "severability"],
        "expected": "confidentiality_nda",
        "is_urgent": False,
    },
    {
        "domain": "Legal & Contracts",
        "query": "This agreement shall be governed by and construed under the laws of the State of Delaware.",
        "options": ["governing_law_jurisdiction", "force_majeure", "indemnification", "confidentiality_nda"],
        "expected": "governing_law_jurisdiction",
        "is_urgent": False,
    },
    {
        "domain": "Legal & Contracts",
        "query": "Supplier shall indemnify and hold harmless the customer from any third party patent infringement claims.",
        "options": ["indemnification_defense", "payment_terms", "term_and_termination", "severability"],
        "expected": "indemnification_defense",
        "is_urgent": True,
    },
    {
        "domain": "Developer & Git",
        "query": "Need to save my current uncommitted working directory edits temporarily so I can pull origin main.",
        "options": ["git_stash", "git_rebase", "git_cherry_pick", "git_reset_hard"],
        "expected": "git_stash",
        "is_urgent": False,
    },
    {
        "domain": "Developer & Git",
        "query": "Apply commit 7a8b9c from develop branch directly onto release-1.2 branch.",
        "options": ["git_cherry_pick", "git_commit_amend", "git_clean", "git_stash"],
        "expected": "git_cherry_pick",
        "is_urgent": False,
    },
    {
        "domain": "Developer & Git",
        "query": "Discard all unstaged and staged changes completely and revert back to commit HEAD.",
        "options": ["git_reset_hard", "git_merge", "git_stash_pop", "git_branch"],
        "expected": "git_reset_hard",
        "is_urgent": True,
    },
]

def run_benchmark(engine: DecisionEngine):
    print("\n==================================================")
    print("      SCIENTIFIC ZERO-SHOT OOD BENCHMARK SUITE     ")
    print("==================================================")

    latencies = []
    correct_choices = 0
    correct_nouls = 0
    brier_sum = 0.0

    for item in BENCHMARK_SUITE:
        questions = [
            Choice("category", options=item["options"]),
            Noul("is_urgent"),
            Score("severity", min_value=0.0, max_value=100.0),
        ]

        result = engine.decide(state=item["query"], questions=questions)
        latencies.append(result.latency_ms)

        ans_choice = result.answers["category"]
        ans_noul = result.answers["is_urgent"]

        is_choice_ok = ans_choice.choice == item["expected"]
        is_noul_ok = ans_noul.value == item["is_urgent"]

        if is_choice_ok:
            correct_choices += 1
        if is_noul_ok:
            correct_nouls += 1

        brier_sum += (ans_noul.probability - (1.0 if item["is_urgent"] else 0.0)) ** 2

        status_str = "PASS" if is_choice_ok else "FAIL"
        print(f"\n[{status_str}] [{item['domain']}]")
        print(f"  Query:      {item['query'][:80]}...")
        print(f"  Prediction: {ans_choice.choice} ({ans_choice.confidence * 100:.1f}%) | Expected: {item['expected']}")
        print(f"  Urgent:     {ans_noul.value} (p={ans_noul.probability:.3f}) | Expected: {item['is_urgent']}")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    choice_acc = (correct_choices / len(BENCHMARK_SUITE)) * 100.0
    noul_acc = (correct_nouls / len(BENCHMARK_SUITE)) * 100.0
    brier_score = brier_sum / len(BENCHMARK_SUITE)

    print("\n==================================================")
    print("              FINAL BENCHMARK RESULTS             ")
    print("==================================================")
    print(f"Zero-Shot OOD Accuracy: {choice_acc:.1f}% ({correct_choices}/{len(BENCHMARK_SUITE)})")
    print(f"Calibrated Noul Accuracy: {noul_acc:.1f}% ({correct_nouls}/{len(BENCHMARK_SUITE)})")
    print(f"Epistemic Brier Score:   {brier_score:.4f} (0.0 = perfect)")
    print(f"Inference Latency P50:   {p50:.2f} ms")
    print(f"Inference Latency P95:   {p95:.2f} ms")
    print("==================================================\n")

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "checkpoint.pt")
    engine = DecisionEngine.from_checkpoint(checkpoint_path)
    run_benchmark(engine)

if __name__ == "__main__":
    main()
