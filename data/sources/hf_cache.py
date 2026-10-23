import json
import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Protocol, runtime_checkable

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
from datasets import load_dataset

logger = logging.getLogger(__name__)
DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

@runtime_checkable
class IRawCachePolicy(Protocol):
    def stream(
        self,
        dataset: str,
        split: str,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[dict]: ...

class HFCache(IRawCachePolicy):
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_RAW_DIR

    def stream(
        self,
        dataset: str,
        split: str,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[dict]:
        clean_ds = dataset.replace("/", "_")
        key = f"{clean_ds}_{subset}_{split}" if subset else f"{clean_ds}_{split}"
        target = self.cache_dir / f"{key}.jsonl"

        if not target.exists():
            args = [dataset]
            if subset:
                args.append(subset)
            logger.info("Fetching '%s' from HuggingFace (%s)...", key, dataset)
            rows = list(load_dataset(*args, split=split))
            if rows:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                target.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
                logger.info("Cached '%s' (%d rows) to disk", key, len(rows))

        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                count = 0
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    yield json.loads(stripped)
                    count += 1
                    if limit is not None and count >= limit:
                        break

__all__ = ["IRawCachePolicy", "HFCache", "DEFAULT_RAW_DIR"]
