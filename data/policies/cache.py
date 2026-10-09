import json
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable
from inference.questions import DecisionSample

@runtime_checkable
class ICachePolicy(Protocol):
    """Sovereign opinion sub-primitive contract: manages storage and retrieval of prepared dataset partitions."""
    def get(self, key: str) -> Optional[Sequence[DecisionSample]]: ...
    def put(self, key: str, samples: Sequence[DecisionSample]) -> None: ...

class DiskCachePolicy(ICachePolicy):
    """Local JSONL filesystem cache implementation."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def get(self, key: str) -> Optional[Sequence[DecisionSample]]:
        target = self.root_dir / f"{key}.jsonl"
        if not target.exists():
            return None
        return [DecisionSample.from_dict(json.loads(line)) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]

    def put(self, key: str, samples: Sequence[DecisionSample]) -> None:
        target = self.root_dir / f"{key}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(s.to_dict(), default=str) for s in samples]
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def load_raw_jsonl(path: Path) -> Sequence[DecisionSample]:
    p = Path(path)
    if not p.exists():
        return []
    return [DecisionSample.from_dict(json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

__all__ = ["ICachePolicy", "DiskCachePolicy", "load_raw_jsonl"]
