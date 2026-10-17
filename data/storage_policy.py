import orjson
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable
from data.case import DataSplits, DecisionCase, SplitTag, case_from_dict, case_to_dict

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

@runtime_checkable
class IStoragePolicy(Protocol):
    """Sub-Primitive Policy: Manages persistence and retrieval of partitioned dataset splits."""
    def save(self, splits: DataSplits) -> None: ...
    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]: ...

class DiskStoragePolicy(IStoragePolicy):
    """Default disk persistence policy for JSONL corpus storage."""

    def __init__(self, corpus_dir: Optional[Union[str, Path]] = None):
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR

    def save(self, splits: DataSplits) -> None:
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        for tag, items in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
            target = self.corpus_dir / f"{tag}.jsonl"
            payload = b"\n".join(orjson.dumps(case_to_dict(s), default=str) for s in items)
            target.write_bytes(payload + (b"\n" if payload else b""))

    def load(self, split: SplitTag = "train") -> Sequence[DecisionCase]:
        target = self.corpus_dir / f"{split}.jsonl"
        if not target.exists():
            raise FileNotFoundError(
                f"Corpus partition '{split}' not found at {target}. "
                f"Explicitly prepare the corpus first (run 'python -m data' or invoke dataset.prepare())."
            )
        return [case_from_dict(orjson.loads(line)) for line in target.read_bytes().splitlines() if line.strip()]

__all__ = ["IStoragePolicy", "DiskStoragePolicy"]
