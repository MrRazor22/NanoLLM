import json
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable
from inference.case import DecisionCase

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

@runtime_checkable
class ICachePolicy(Protocol):
    """Sovereign opinion sub-primitive contract: manages storage and retrieval of prepared dataset partitions."""
    def get(self, key: str) -> Optional[Sequence[DecisionCase]]: ...
    def put(self, key: str, samples: Sequence[DecisionCase]) -> None: ...

class DiskCachePolicy(ICachePolicy):
    """Local JSONL filesystem cache implementation."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_CORPUS_DIR

    def get(self, key: str) -> Optional[Sequence[DecisionCase]]:
        target = self.root_dir / f"{key}.jsonl"
        if not target.exists():
            return None
        return [DecisionCase.from_dict(json.loads(line)) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]

    def put(self, key: str, samples: Sequence[DecisionCase]) -> None:
        target = self.root_dir / f"{key}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(s.to_dict(), default=str) for s in samples]
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def load_raw_jsonl(path: Path) -> Sequence[DecisionCase]:
    p = Path(path)
    if not p.exists():
        return []
    return [DecisionCase.from_dict(json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

__all__ = ["ICachePolicy", "DiskCachePolicy", "load_raw_jsonl"]
