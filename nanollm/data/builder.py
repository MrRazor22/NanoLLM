import json, random
from typing import Any, Dict, List, Optional, Tuple

def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def split_train_val(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.08,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    generator = rng if rng is not None else random.Random(42)
    shuffled = list(records)
    generator.shuffle(shuffled)
    n_val = max(10, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]

def choice_question(name: str, target: Any, options: Any, instruction: str) -> List[Any]:
    return [name, "choice", target, options, instruction]
