from typing import Any, Dict, List, Optional
from benchmark.tracks.base_policy import ITrackPolicy
from benchmark.tracks.agentic_policy import AgenticPolicy
from benchmark.tracks.abstention_policy import AbstentionPolicy
from benchmark.tracks.typed_decisions_policy import TypedDecisionsPolicy
from benchmark.tracks.laya_policy import LayaPolicy

_TRACKS: Dict[str, ITrackPolicy] = {
    "agentic": AgenticPolicy(),
    "abstention": AbstentionPolicy(),
    "typed_decisions": TypedDecisionsPolicy(),
    "typed": TypedDecisionsPolicy(),
    "verdict": TypedDecisionsPolicy(),
    "laya": LayaPolicy(),
}

def load_benchmark_items(track: str = "agentic", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    policy = _TRACKS.get(track)
    if policy is None:
        raise ValueError(f"Unknown benchmark track: {track}. Available: {list(_TRACKS.keys())}")
    return policy.load(limit=limit)

__all__ = [
    "ITrackPolicy",
    "AgenticPolicy",
    "AbstentionPolicy",
    "TypedDecisionsPolicy",
    "LayaPolicy",
    "load_benchmark_items",
]
