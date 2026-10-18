import json
import logging
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Protocol, Sequence, runtime_checkable
from datasets import load_dataset
from data.case import DecisionCase

logger = logging.getLogger("piqo.sources")

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

def load_raw_jsonl(
    target_name: str,
    dataset: str,
    split: str,
    subset: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    """Loads dataset rows from local disk cache, fetching and persisting from HuggingFace if absent."""
    target = RAW_DIR / f"{target_name}.jsonl"
    if target.exists():
        logger.info("Loading cached %s (%s)", target_name, target.name)
        with open(target, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
            return rows[:limit] if limit is not None else rows

    logger.info("Fetching %s (%s, split=%s) from HuggingFace...", target_name, dataset, split)
    args = [dataset]
    if subset:
        args.append(subset)
    rows = list(load_dataset(*args, split=split))
    if limit is not None:
        rows = rows[:limit]
    if rows:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    return rows

@runtime_checkable
class IDataSource(Protocol):
    """Sovereign policy contract: yields domain records from external data origins."""
    def cases(self) -> Sequence[DecisionCase]: ...

def get_available_sources() -> Sequence[IDataSource]:
    """Lazy catalog of all available decision sources."""
    from data.sources.anli_source import ANLISource
    from data.sources.art_source import ARTSource
    from data.sources.copa_source import COPASource
    from data.sources.decisions_source import DecisionsSource
    from data.sources.glaive_source import GlaiveSource
    from data.sources.hellaswag_source import HellaSwagSource
    from data.sources.huggingface_source import HuggingFaceSource, STANDARD_BENCHMARKS
    from data.sources.medqa_source import MedQASource
    from data.sources.multichoice_source import MultiChoiceReasoningSource, MULTICHOICE_REASONING_BENCHMARKS
    from data.sources.sciq_source import SciQSource
    from data.sources.reclor_source import ReClorSource

    return [
        GlaiveSource(),
        DecisionsSource(),
        ANLISource(),
        HellaSwagSource(),
        MedQASource(),
        ARTSource(),
        COPASource(),
        SciQSource(),
        ReClorSource(),
        *(HuggingFaceSource(s) for s in STANDARD_BENCHMARKS),
        *(MultiChoiceReasoningSource(s) for s in MULTICHOICE_REASONING_BENCHMARKS),
    ]

__all__ = ["IDataSource", "get_available_sources", "load_raw_jsonl", "RAW_DIR"]
