import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from nanollm import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINT,
    Choice,
    DecisionEngine,
    DecisionResult,
    Noul,
    ProfilingLayer,
    Question,
    Score,
    get_latest_checkpoint,
    list_all_checkpoints,
)

PRESETS: Dict[str, Dict] = {
    "triage": {
        "title": "Incident Triage",
        "default": "Postgres CPU reached 99% and connection pool is exhausted.",
        "questions": [
            Choice("department", ["Billing", "Infrastructure", "Sales", "Security"]),
            Noul("urgent", instruction="Is immediate incident response required?"),
            Score("severity", min_value=0.0, max_value=100.0),
        ],
    },
    "routing": {
        "title": "Tool Routing",
        "default": "Find all customer orders placed in the last 24 hours.",
        "questions": [
            Choice("tool", ["query_database", "search_docs", "send_notification", "none"]),
            Noul("requires_confirmation"),
            Score("priority", min_value=1.0, max_value=10.0),
        ],
    },
}

def render_decision(result: DecisionResult, state: str) -> None:
    print(f"\nState: {state}\nLatency: {result.latency_ms:.2f} ms\n" + "-" * 50)
    for name, ans in result.answers.items():
        if hasattr(ans, "choice"):
            print(f"  {name:15s} -> {ans.choice:<20s} ({ans.confidence * 100:5.1f}%)")
        elif hasattr(ans, "value"):
            print(f"  {name:15s} -> {'YES' if ans.value else 'NO':<20s} (p={ans.probability:.3f})")
        elif hasattr(ans, "score"):
            print(f"  {name:15s} -> {ans.score:5.1f} / 100.0")
    print()

def parse_question(arg: str, q_type: str) -> Question:
    parts = arg.split(":", 1)
    name, spec = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else None)
    if q_type == "choice":
        opts = [o.strip() for o in spec.split(",")] if spec else ["Yes", "No"]
        return Choice(name=name, options=opts)
    if q_type == "noul":
        return Noul(name=name, instruction=spec)
    return Score(name=name, min_value=0.0, max_value=100.0, instruction=spec)

def main() -> None:
    latest_ckpt, _ = get_latest_checkpoint()
    parser = argparse.ArgumentParser(description="NanoLLM Micro-Decision Engine CLI")
    parser.add_argument("state", type=str, nargs="?", default=None)
    parser.add_argument("--play", "-p", action="store_true")
    parser.add_argument("--models", "-m", action="store_true")
    parser.add_argument("--checkpoint", "-k", type=str, default=str(latest_ckpt or DEFAULT_CHECKPOINT))
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="triage")
    parser.add_argument("--choice", action="append")
    parser.add_argument("--noul", action="append")
    parser.add_argument("--score", action="append")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.models:
        print(f"\n{'Checkpoint':<32} | {'Size':>8} | {'Val Loss':>9}\n" + "-" * 55)
        for c in list_all_checkpoints():
            vl = f"{c.val_loss:.4f}" if c.val_loss is not None else "-"
            print(f"{c.name:<32} | {c.size_mb:>5.1f} MB | {vl:>9}")
        return

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists() and (CHECKPOINTS_DIR / ckpt_path.name).exists():
        ckpt_path = CHECKPOINTS_DIR / ckpt_path.name
    engine = DecisionEngine.from_checkpoint(ckpt_path, device=args.device) | ProfilingLayer()

    if args.play:
        print(f"NanoLLM Playground [{ckpt_path.name}]. Enter context (or 'q' to quit):")
        while True:
            ctx = input("\nContext: ").strip()
            if not ctx or ctx.lower() in ("q", "quit"): break
            opts = input("Choices (comma-separated): ").strip()
            render_decision(engine.decide(ctx, [Choice("decision", [o.strip() for o in opts.split(",")])]), ctx)
        return

    state = args.state or PRESETS[args.preset]["default"]
    questions: List[Question] = []
    if args.choice: questions.extend(parse_question(c, "choice") for c in args.choice)
    if args.noul: questions.extend(parse_question(n, "noul") for n in args.noul)
    if args.score: questions.extend(parse_question(s, "score") for s in args.score)
    render_decision(engine.decide(state, questions or PRESETS[args.preset]["questions"]), state)

if __name__ == "__main__":
    main()
