import argparse
from pathlib import Path
from nanollm.inference.engine import DEFAULT_CHECKPOINT, DecisionEngine
from nanollm.inference.layers.profiling import ProfilingLayer
from nanollm.inference.schema import Choice, Noul, Score

def main() -> None:
    parser = argparse.ArgumentParser(description="NanoLLM User CLI")
    parser.add_argument(
        "state",
        type=str,
        nargs="?",
        default="Our primary Postgres database CPU reached 99% and connection pool is exhausted.",
        help="State / prompt to decide on",
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path")
    args = parser.parse_args()

    engine = DecisionEngine.from_checkpoint(args.checkpoint) | ProfilingLayer

    questions = [
        Choice("department", options=["Billing", "Infrastructure / Tech", "Enterprise / Sales", "Security"]),
        Noul("urgent"),
        Score("severity", min_value=0.0, max_value=100.0),
    ]

    result = engine.decide(state=args.state, questions=questions)
    print(f"\nState: {args.state}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    for name, ans in result.answers.items():
        if hasattr(ans, "choice"):
            print(f"  {name:12s} -> {ans.choice} ({ans.confidence * 100:.1f}%)")
        elif hasattr(ans, "value"):
            print(f"  {name:12s} -> {ans.value} (p={ans.probability:.3f})")
        elif hasattr(ans, "score"):
            print(f"  {name:12s} -> {ans.score:.1f}/100")
    print()

if __name__ == "__main__":
    main()
