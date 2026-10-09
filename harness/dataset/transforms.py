import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from harness.dataset.training_dataset import TrainingDataset, to_decision_sample

def load_raw_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists(): return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path: Union[str, Path], records: Sequence[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, default=str) + "\n")

def inject_abstention(records: List[Dict[str, Any]], rate: float = 0.15, rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
    gen = rng or random.Random(42)
    out = []
    for r in records:
        if gen.random() >= rate:
            out.append(r)
            continue
        new_r = dict(r)
        new_qs = {}
        for qid, qspec in r.get("questions", {}).items():
            opts = dict(qspec.get("options") or {})
            if len(opts) >= 2:
                opts["__none__"] = "none of the above"
                new_qs[qid] = {**qspec, "options": opts}
            else:
                new_qs[qid] = qspec
        new_r["questions"] = new_qs
        new_r["gold"] = {qid: "__none__" for qid in new_qs}
        out.append(new_r)
    return out

def split_train_val(records: List[Dict[str, Any]], val_ratio: float = 0.08, rng: Optional[random.Random] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return TrainingDataset.split_records(records, val_ratio=val_ratio, rng=rng)

__all__ = ["load_raw_jsonl", "save_jsonl", "inject_abstention", "split_train_val", "to_decision_sample"]
