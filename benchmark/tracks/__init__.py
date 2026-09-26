from typing import Any, Dict, List, Optional
from benchmark.tracks.base import ITrackPolicy
from benchmark.tracks.json_tracks import AgenticTrackPolicy, LayaTrackPolicy
from benchmark.tracks.abstention import AbstentionTrackPolicy
from benchmark.tracks.typed_decisions import TypedDecisionsTrackPolicy

_TRACKS: Dict[str, ITrackPolicy] = {
    "agentic": AgenticTrackPolicy(),
    "abstention": AbstentionTrackPolicy(),
    "typed_decisions": TypedDecisionsTrackPolicy(),
    "typed": TypedDecisionsTrackPolicy(),
    "verdict": TypedDecisionsTrackPolicy(),
    "laya": LayaTrackPolicy(),
}

def load_benchmark_items(track: str = "agentic", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    policy = _TRACKS.get(track)
    if policy is None:
        raise ValueError(f"Unknown benchmark track: {track}. Available: {list(_TRACKS.keys())}")
    return policy.load(limit=limit)

__all__ = [
    "ITrackPolicy",
    "AgenticTrackPolicy",
    "AbstentionTrackPolicy",
    "TypedDecisionsTrackPolicy",
    "LayaTrackPolicy",
    "load_benchmark_items",
]
