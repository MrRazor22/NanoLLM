import json
import logging
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, runtime_checkable
import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
from datasets import load_dataset

logger = logging.getLogger(__name__)
DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

@runtime_checkable
class IRawCachePolicy(Protocol):
    """Sub-Primitive Policy: loads raw dataset records, caching from remote origin on miss."""

    def load(
        self,
        dataset: str,
        split: str,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Sequence[dict]: ...

class HFCache(IRawCachePolicy):
    """Disk-backed caching policy fetching from HuggingFace on cache miss."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_RAW_DIR

    def load(
        self,
        dataset: str,
        split: str,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Sequence[dict]:
        clean_ds = dataset.replace("/", "_")
        key = f"{clean_ds}_{subset}_{split}" if subset else f"{clean_ds}_{split}"
        target = self.cache_dir / f"{key}.jsonl"
        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
                logger.info("Cache hit: '%s' (%d rows)", key, len(rows))
                return rows[:limit] if limit is not None else rows

        args = [dataset]
        if subset:
            args.append(subset)
        logger.info("Fetching '%s' from HuggingFace (%s)...", key, dataset)
        rows = list(load_dataset(*args, split=split))
        if limit is not None:
            rows = rows[:limit]
        if rows:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
            logger.info("Cached '%s' (%d rows) to disk", key, len(rows))
        return rows

__all__ = ["IRawCachePolicy", "HFCache", "DEFAULT_RAW_DIR"]
