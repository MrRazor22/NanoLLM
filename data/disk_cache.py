import json
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable
from data.case import DecisionCase

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

@runtime_checkable
class ICachePolicy(Protocol):
    """Sovereign opinion sub-primitive contract: manages storage and retrieval of prepared dataset partitions."""
    @property
    def signature(self) -> str: ...
    def get(self, key: str, signature: Optional[str] = None) -> Optional[Sequence[DecisionCase]]: ...
    def put(self, key: str, samples: Sequence[DecisionCase], signature: Optional[str] = None) -> None: ...

class DiskCachePolicy(ICachePolicy):
    """Local JSONL filesystem cache implementation with automatic signature validation."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_CORPUS_DIR

    @property
    def signature(self) -> str:
        return "disk_jsonl_v1"

    def get(self, key: str, signature: Optional[str] = None) -> Optional[Sequence[DecisionCase]]:
        if signature is not None:
            sig_file = self.root_dir / ".signature"
            if not sig_file.exists() or sig_file.read_text(encoding="utf-8").strip() != signature:
                return None
        target = self.root_dir / f"{key}.jsonl"
        if not target.exists():
            return None
        return [DecisionCase.from_dict(json.loads(line)) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]

    def put(self, key: str, samples: Sequence[DecisionCase], signature: Optional[str] = None) -> None:
        target = self.root_dir / f"{key}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(s.to_dict(), default=str) for s in samples]
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if signature is not None:
            (self.root_dir / ".signature").write_text(signature, encoding="utf-8")
