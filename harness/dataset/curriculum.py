from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from harness.dataset.sources import Sources
from harness.dataset.transforms import inject_abstention, load_raw_jsonl, save_jsonl, split_train_val

DATA_DIR = Path(__file__).resolve().parent / "data"
SPLITS_DIR = DATA_DIR / "splits"

def build_curriculum(
    limit_per_source: int = 5000,
    seed: int = 42,
    abstention_rate: float = 0.15,
    val_ratio: float = 0.08,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    samples.extend(Sources.load_typed(repeat=500))
    samples.extend(Sources.load_glaive(limit=limit_per_source, seed=seed))
    samples.extend(Sources.load_generic("mteb/banking77", limit=limit_per_source))

    samples = inject_abstention(samples, rate=abstention_rate, rng=rng)
    train_recs, val_recs = split_train_val(samples, val_ratio=val_ratio, rng=rng)

    save_jsonl(SPLITS_DIR / "train.jsonl", train_recs)
    save_jsonl(SPLITS_DIR / "val.jsonl", val_recs)
    print(f"Curriculum built: {len(train_recs)} train, {len(val_recs)} val.")
    return train_recs, val_recs

__all__ = ["build_curriculum"]
