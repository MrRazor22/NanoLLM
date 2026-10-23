import logging
import orjson
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable
from data.case import DecisionCase

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

@runtime_checkable
class IStoragePolicy(Protocol):
    def save(self, cases: Sequence[DecisionCase]) -> None: ...
    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]: ...

class DiskStoragePolicy(IStoragePolicy):
    def __init__(self, corpus_dir: Optional[Union[str, Path]] = None):
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR

    def save(self, cases: Sequence[DecisionCase]) -> None:
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for tag in ("train", "val", "test"):
            items = [c for c in cases if c.meta and c.meta.split == tag]
            target = self.corpus_dir / f"{tag}.jsonl"
            payload = b"\n".join(orjson.dumps(s.to_dict(), default=str) for s in items)
            target.write_bytes(payload + (b"\n" if payload else b""))
            counts[tag] = len(items)
        logger.info("Persisted corpus to %s (train: %d, val: %d, test: %d)", self.corpus_dir, counts["train"], counts["val"], counts["test"])

    def load(self, split: Optional[str] = None) -> Sequence[DecisionCase]:
        if not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"Corpus directory not found at {self.corpus_dir}. "
                "Explicitly prepare the corpus first (run 'python -m data' or invoke dataset.prepare())."
            )
        tags = (split,) if split is not None else ("train", "val", "test")
        cases: list[DecisionCase] = []
        for tag in tags:
            target = self.corpus_dir / f"{tag}.jsonl"
            if target.exists():
                cases.extend(DecisionCase.from_dict(orjson.loads(line)) for line in target.read_bytes().splitlines() if line.strip())
        return cases

__all__ = ["IStoragePolicy", "DiskStoragePolicy"]
