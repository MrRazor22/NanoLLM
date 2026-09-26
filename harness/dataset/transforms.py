import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from harness.dataset.schema import DecisionSample, QuestionSpec

def to_decision_sample(item: Dict[str, Any]) -> DecisionSample:
    specs = [
        QuestionSpec(
            name=q[0],
            q_type=q[1],
            target=q[2],
            options=q[3] if len(q) > 3 else None,
            instruction=q[4] if len(q) > 4 else None,
        )
        for q in item["questions"]
    ]
    return DecisionSample(state=item["state"], questions=specs)

def load_jsonl(path: Union[str, Path]) -> List[DecisionSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(to_decision_sample(json.loads(line)))
    return samples

def save_jsonl(path: Union[str, Path], records: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def inject_abstention(
    records: List[Dict[str, Any]],
    rate: float = 0.15,
    token: str = "__insufficient_evidence__",
    desc: str = "insufficient evidence or none of the above options apply",
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    gen = rng if rng is not None else random.Random(42)
    out = []
    for rec in records:
        if gen.random() >= rate:
            out.append(rec)
            continue
        qs = []
        for q in rec.get("questions", []):
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
        out.append({"state": rec["state"], "questions": qs})
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
