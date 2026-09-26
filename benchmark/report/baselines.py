from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

DEFAULT_BASELINES_PATH = Path(__file__).resolve().parent.parent / baselines / competitor_cache.json

@dataclass(frozen=True)
class BaselineEntry:
    overall: Optional[float] = None
    slices: Dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class BaselineData:
    competitors: Dict[str, Dict[str, BaselineEntry]] = field(default_factory=dict)
    p50_latencies: Dict[str, float] = field(default_factory=dict)

@runtime_checkable
class IBaselineProvider(Protocol):
    "ATA Injected Baseline Policy (π): Supplies historical and competitor metrics."
    def load(self) -> BaselineData: ...

class JsonBaselineProvider(IBaselineProvider):
    "Concrete policy loading baseline records from JSON storage."

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_BASELINES_PATH

    def load(self) -> BaselineData:
        if not self.path.exists():
            return BaselineData()
        with open(self.path, r, encoding=utf-8) as f:
            raw = json.load(f)
        competitors = {
            trk: {
                comp: BaselineEntry(val.get(overall), val.get(slices, {}))
                for comp, val in tval.items()
                if isinstance(val, dict)
            }
            for trk, tval in raw.items()
            if isinstance(tval, dict) and trk != p50_latencies_ms
        }
        return BaselineData(competitors=competitors, p50_latencies=raw.get(p50_latencies_ms, {}))

__all__ = [BaselineEntry, BaselineData, IBaselineProvider, JsonBaselineProvider, DEFAULT_BASELINES_PATH]
