from pathlib import Path
import random
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable
from piqo import DecisionSample

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

@runtime_checkable
class ISource(Protocol):
    """Sovereign primitive contract for training dataset source extractors."""
    name: str
    def load(self, limit: Optional[int] = None) -> Sequence[DecisionSample]: ...

def cached_jsonl(path: Path, fetcher) -> List[dict]:
    import json
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = fetcher()
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows

from harness.sources.glaive_source import GlaiveSource
from harness.sources.decisions_source import DecisionsSource
from harness.sources.generic_source import GenericSource

def build_curriculum(
    dest_dir: Path,
    sources: Optional[Sequence[ISource]] = None,
    seed: int = 42,
    abstention_rate: float = 0.15,
    val_ratio: float = 0.08,
    holdouts: Optional[Sequence[DecisionSample]] = None,
    policy: Optional[ILeakagePolicy] = None,
) -> Tuple[List[DecisionSample], List[DecisionSample]]:
    from harness.policies.leakage import ILeakagePolicy, NGramLeakagePolicy
    from harness.training_dataset import inject_abstention, save_jsonl, split_train_val

    rng = random.Random(seed)
    dest_dir.mkdir(parents=True, exist_ok=True)

    active_sources = list(sources) if sources else [
        GlaiveSource(limit=5000, seed=seed),
        DecisionsSource(repeat=5),
        GenericSource("mteb/banking77", "intent", "What is the primary customer request?", limit=5000),
    ]

    samples: List[DecisionSample] = []
    for src in active_sources:
        loaded = list(src.load())
        print(f"  [Source: {src.name}] Loaded {len(loaded)} real records.")
        samples.extend(loaded)

    samples = inject_abstention(samples, rate=abstention_rate, rng=rng)
    train_recs, val_recs = split_train_val(samples, val_ratio=val_ratio, rng=rng)

    active_policy: ILeakagePolicy = policy or NGramLeakagePolicy(13)
    if holdouts:
        if active_policy.find_leaks(train_recs, holdouts) or active_policy.find_leaks(val_recs, holdouts):
            raise ValueError("Contamination invariant violated: curriculum leaks holdouts.")
    if active_policy.find_leaks(train_recs, val_recs):
        raise ValueError("Contamination invariant violated: train split leaks validation.")

    save_jsonl(dest_dir / "train.jsonl", train_recs)
    save_jsonl(dest_dir / "val.jsonl", val_recs)
    print(f"Curriculum adapted: {len(train_recs)} train, {len(val_recs)} val records written (zero leakage verified).")
    return train_recs, val_recs

__all__ = [
    "DATA_DIR",
    "RAW_DIR",
    "ISource",
    "GlaiveSource",
    "DecisionsSource",
    "GenericSource",
    "build_curriculum",
]
