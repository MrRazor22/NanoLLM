from pathlib import Path
from typing import List
import os
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from nanollm import Choice, DecisionEngine, Noul, Score, load_jsonl

DEFAULT_DEPARTMENTS = ["Billing", "Infrastructure / Tech", "Enterprise / Sales", "Security"]

def evaluate_dataset(engine: DecisionEngine, path: str):
    if not os.path.exists(path):
        print(f"Dataset path does not exist: {path}")
        return

    samples = load_jsonl(path)
    total_samples = len(samples)
    choice_correct = 0
    noul_correct = 0
    noul_brier_sum = 0.0
    score_mae_sum = 0.0

    start_time = time.perf_counter()
    for sample in samples:
        choice_q = next(q for q in sample.questions if q.q_type == "choice")
        urgent_q = next(q for q in sample.questions if q.q_type == "noul")
        score_q = next(q for q in sample.questions if q.q_type == "score")

        options = choice_q.options if choice_q.options else DEFAULT_DEPARTMENTS
        questions = [
            Choice("department", options=options),
            Noul("is_urgent"),
            Score("severity", min_value=0.0, max_value=100.0),
        ]

        result = engine.decide(state=sample.state, questions=questions)
        dept_res = result.answers["department"]
        target_label = options[choice_q.target]
        if dept_res.choice == target_label:
            choice_correct += 1

        urgent_res = result.answers["is_urgent"]
        target_urgent = urgent_q.target >= 0.5
        if urgent_res.value == target_urgent:
            noul_correct += 1
        noul_brier_sum += (urgent_res.probability - float(target_urgent)) ** 2

        score_res = result.answers["severity"]
        score_mae_sum += abs(score_res.score - (score_q.target * 100.0))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency = elapsed_ms / max(1, total_samples)

    print("==================================================")
    print("          STANDARDIZED EVALUATION REPORT          ")
    print("==================================================")
    print(f"Target Dataset:       {path} ({total_samples} samples)")
    print(f"Inference Latency:    {avg_latency:.2f} ms / sample")
    print("--------------------------------------------------")
    print(f"Choice Accuracy:      {(choice_correct / total_samples) * 100:.2f}%")
    print(f"Noul Accuracy:        {(noul_correct / total_samples) * 100:.2f}%")
    print(f"Noul Brier Score:     {noul_brier_sum / total_samples:.4f} (0.0 = perfect)")
    print(f"Score Error (MAE):    {score_mae_sum / total_samples:.2f} points")
    print("==================================================")

def predict_single(engine: DecisionEngine, query: str, options: List[str]):
    questions = [
        Choice("department", options=options),
        Noul("urgent"),
        Score("severity", min_value=0.0, max_value=100.0),
    ]
    result = engine.decide(query, questions)
    dept = result.answers["department"]
    urgent = result.answers["urgent"]
    sev = result.answers["severity"]

    print(f"\nQuery: {query}")
    print(f"  Latency:    {result.latency_ms:.2f} ms")
    print(f"  Selection:  {dept.choice} ({dept.confidence * 100:.1f}%)")
    for opt, prob in dept.probabilities.items():
        print(f"    - {opt}: {prob * 100:.1f}%")
    print(f"  Urgent:     {urgent.value} (p={urgent.probability:.3f})")
    print(f"  Severity:   {sev.score:.1f}/100")

def main():
    checkpoint_path = str(ROOT_DIR / "checkpoint.pt")
    engine = DecisionEngine.from_checkpoint(checkpoint_path)
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        query = " ".join(sys.argv[1:])
        predict_single(engine, query, DEFAULT_DEPARTMENTS)
    else:
        dataset_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--data" else str(ROOT_DIR / "data" / "test.jsonl")
        evaluate_dataset(engine, dataset_path)

if __name__ == "__main__":
    main()
