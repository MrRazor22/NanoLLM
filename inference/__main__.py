import argparse, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Sequence, Union
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from piqo import (
    CHECKPOINTS_DIR, DEFAULT_CHECKPOINT, Choice, ChoiceResult, DecisionEngine,
    DecisionResult, Noul, NoulResult, ProfilingLayer, Question, Score, ScoreResult,
    TorchModel, get_latest_checkpoint, list_all_checkpoints,
)

PRESETS = {
    "triage": {
        "default": "Postgres CPU reached 99% and connection pool is exhausted.",
        "questions": [Choice("department", ["Billing", "Infrastructure", "Sales", "Security"]), Noul("urgent", "Is immediate incident response required?"), Score("severity", min_value=0.0, max_value=100.0)],
    },
    "routing": {
        "default": "Find all customer orders placed in the last 24 hours.",
        "questions": [Choice("tool", ["query_database", "search_docs", "send_notification", "none"]), Noul("requires_confirmation"), Score("priority", min_value=1.0, max_value=10.0)],
    },
}

def render_decision(res: DecisionResult, state: str) -> None:
    print(f"\nState: {state}\nLatency: {res.latency_ms:.2f} ms\n" + "-" * 50)
    for k, v in res.answers.items():
        if isinstance(v, ChoiceResult): print(f"  {k:15s} -> {v.choice:<20s} ({v.confidence * 100:5.1f}%)")
        elif isinstance(v, NoulResult): print(f"  {k:15s} -> {'YES' if v.value else 'NO':<20s} (p={v.probability:.3f})")
        elif isinstance(v, ScoreResult): print(f"  {k:15s} -> {v.score:5.1f} / 100.0")
    print()

def _parse_q(arg: str, kind: str) -> Question:
    name, _, spec = arg.partition(":")
    name, spec = name.strip(), spec.strip() or None
    if kind == "choice": return Choice(name, [o.strip() for o in spec.split(",")] if spec else ["Yes", "No"])
    return Noul(name, instruction=spec) if kind == "noul" else Score(name, instruction=spec, min_value=0.0, max_value=100.0)

def main() -> None:
    latest, _ = get_latest_checkpoint()
    p = argparse.ArgumentParser(description="Piqo Micro-Decision Engine CLI")
    p.add_argument("state", nargs="?", default=None)
    p.add_argument("--play", "-p", action="store_true")
    p.add_argument("--models", "-m", action="store_true")
    p.add_argument("--checkpoint", "-k", default=str(latest or DEFAULT_CHECKPOINT))
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="triage")
    p.add_argument("--choice", action="append"); p.add_argument("--noul", action="append"); p.add_argument("--score", action="append"); p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.models:
        print(f"\n{'Checkpoint':<32} | {'Size':>8} | {'Val Loss':>9}\n" + "-" * 55)
        for c in list_all_checkpoints():
            print(f"{c.name:<32} | {c.size_mb:>5.1f} MB | {f'{c.val_loss:.4f}' if c.val_loss is not None else '-':>9}")
        return

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() and (CHECKPOINTS_DIR / ckpt.name).exists(): ckpt = CHECKPOINTS_DIR / ckpt.name
    import torch
    dev = torch.device(args.device) if args.device else None
    model = TorchModel(checkpoint_path=ckpt, device=dev)
    engine = DecisionEngine(model=model) | ProfilingLayer()

    if args.play:
        print(f"Piqo Playground [{ckpt.name}]. Enter context (or 'q' to quit):")
        while (ctx := input("\nContext: ").strip()) and ctx.lower() not in ("q", "quit"):
            opts = [o.strip() for o in input("Choices (comma-separated): ").strip().split(",")]
            render_decision(engine.decide(ctx, [Choice("decision", opts)]), ctx)
        return

    qs: List[Question] = [_parse_q(c, "choice") for c in (args.choice or [])] + [_parse_q(n, "noul") for n in (args.noul or [])] + [_parse_q(s, "score") for s in (args.score or [])]
    state = args.state or PRESETS[args.preset]["default"]
    render_decision(engine.decide(state, qs or PRESETS[args.preset]["questions"]), state)

if __name__ == "__main__":
    main()
