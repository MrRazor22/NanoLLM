import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from harness.dataset.schema import DecisionSample, QuestionSpec

def to_decision_sample(item: Dict[str, Any]) -> DecisionSample:
    state = item.get("state", "")
    qs = item.get("questions", [])
    gold = item.get("gold", {})
    specs: List[QuestionSpec] = []

    if isinstance(qs, dict):
        for name, spec in qs.items():
            opts = spec.get("options") or spec.get("criteria", {})
            opt_keys = list(opts.keys()) if isinstance(opts, dict) else list(opts)
            g = gold.get(name, {})
            lbl = g.get("label") if isinstance(g, dict) else str(g)
            target = opt_keys.index(lbl) if lbl in opt_keys else 0
            specs.append(
                QuestionSpec(
                    name=name,
                    q_type=spec.get("type", "choice"),
                    target=target,
                    options=opts,
                    instruction=spec.get("instructions"),
                )
            )
    elif isinstance(qs, list):
        for q in qs:
            if isinstance(q, (list, tuple)) and len(q) >= 3:
                specs.append(
                    QuestionSpec(
                        name=q[0],
                        q_type=q[1],
                        target=int(q[2]) if isinstance(q[2], (int, float)) else 0,
                        options=q[3] if len(q) > 3 else None,
                        instruction=q[4] if len(q) > 4 else None,
                    )
                )
            elif isinstance(q, dict):
                opts = q.get("options") or q.get("criteria", {})
                specs.append(
                    QuestionSpec(
                        name=q.get("name", "intent"),
                        q_type=q.get("type", "choice"),
                        target=int(q.get("target", 0)),
                        options=opts,
                        instruction=q.get("instructions"),
                    )
                )

    return DecisionSample(state=state, questions=specs)

def load_raw_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_jsonl(path: Union[str, Path]) -> List[DecisionSample]:
    return [to_decision_sample(row) for row in load_raw_jsonl(path)]

def save_jsonl(path: Union[str, Path], records: Sequence[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

def inject_abstention(
    records: List[Dict[str, Any]],
    rate: float = 0.15,
    token: str = "__insufficient_evidence__",
    desc: str = "insufficient evidence or none of the above options apply",
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    gen = rng if rng is not None else random.Random(42)
    out: List[Dict[str, Any]] = []

    for rec in records:
        if gen.random() >= rate:
            out.append(rec)
            continue

        raw_qs = rec.get("questions")
        gold = rec.get("gold", {})

        if isinstance(raw_qs, dict):
            # Dict questions format: {qid: {type, criteria, instructions}}
            new_qs = {}
            new_gold = dict(gold)
            for qid, qspec in raw_qs.items():
                opts_raw = qspec.get("options") or qspec.get("criteria", {})
                options = dict(opts_raw) if isinstance(opts_raw, dict) else {k: k for k in opts_raw}
                q_type = qspec.get("type", "choice")
                if q_type == "choice" and len(options) >= 2:
                    current_gold = new_gold.get(qid, {}).get("label")
                    if current_gold in options:
                        del options[current_gold]
                    options[token] = desc
                    new_keys = list(options.keys())
                    gen.shuffle(new_keys)
                    new_qs[qid] = {
                        "type": "choice",
                        "instructions": qspec.get("instructions"),
                        "criteria": {k: options[k] for k in new_keys},
                    }
                    new_gold[qid] = {"type": "choice", "label": token}
                else:
                    new_qs[qid] = qspec
            out.append({"state": rec["state"], "questions": new_qs, "gold": new_gold})

        elif isinstance(raw_qs, list):
            # List of tuples format: [[name, q_type, target, options, instr]]
            qs = []
            for q in raw_qs:
                if isinstance(q, (list, tuple)) and len(q) >= 3:
                    opts_raw = q[3] if len(q) > 3 and q[3] is not None else None
                    options = dict(opts_raw) if isinstance(opts_raw, dict) else (list(opts_raw) if opts_raw is not None else None)
                    name, q_type, target, instr = q[0], q[1], q[2], q[4] if len(q) > 4 else None
                    if q_type == "choice" and isinstance(options, dict) and len(options) >= 2:
                        keys = list(options.keys())
                        target_key = keys[target] if isinstance(target, int) and target < len(keys) else None
                        if target_key:
                            del options[target_key]
                            options[token] = desc
                            new_keys = list(options.keys())
                            gen.shuffle(new_keys)
                            qs.append([name, "choice", new_keys.index(token), {k: options[k] for k in new_keys}, instr])
                            continue
                    qs.append(q)
                else:
                    qs.append(q)
            out.append({"state": rec["state"], "questions": qs})
        else:
            out.append(rec)

    return out

def split_train_val(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.08,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    gen = rng if rng is not None else random.Random(42)
    shuffled = list(records)
    gen.shuffle(shuffled)
    n_val = max(10, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]

__all__ = [
    "to_decision_sample",
    "load_raw_jsonl",
    "load_jsonl",
    "save_jsonl",
    "inject_abstention",
    "split_train_val",
]
