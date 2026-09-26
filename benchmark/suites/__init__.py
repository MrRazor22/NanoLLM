from pathlib import Path
from typing import Any, Dict, List, Optional
from benchmark.suites.base_policy import ISuitePolicy, ITrackPolicy
from benchmark.suites.agentic_policy import AgenticPolicy
from benchmark.suites.abstention_policy import AbstentionPolicy
from benchmark.suites.typed_decisions_policy import TypedDecisionsPolicy
from benchmark.suites.laya_policy import LayaPolicy
from benchmark.suites.caching_policy import CachingSuitePolicy, CachingTrackPolicy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SUITES: Dict[str, ISuitePolicy] = {
    "agentic": CachingSuitePolicy(AgenticPolicy(), DATA_DIR / "benchmark.cache.json"),
    "laya": CachingSuitePolicy(LayaPolicy(), DATA_DIR / "laya_benchmark.json"),
    "abstention": CachingSuitePolicy(AbstentionPolicy(), DATA_DIR / "abstention.cache.json"),
    "typed_decisions": CachingSuitePolicy(TypedDecisionsPolicy(), DATA_DIR / "typed_decisions.cache.json"),
    "typed": CachingSuitePolicy(TypedDecisionsPolicy(), DATA_DIR / "typed_decisions.cache.json"),
    "verdict": CachingSuitePolicy(TypedDecisionsPolicy(), DATA_DIR / "typed_decisions.cache.json"),
}

def get_suite(name: str = "agentic") -> ISuitePolicy:
    suite = SUITES.get(name)
    if suite is None:
        raise ValueError(f"Unknown benchmark suite: '{name}'. Available: {list(SUITES.keys())}")
    return suite

def load_benchmark_items(suite_name: str = "agentic", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    return get_suite(suite_name).load(limit=limit)

__all__ = [
    "ISuitePolicy",
    "ITrackPolicy",
    "AgenticPolicy",
    "AbstentionPolicy",
    "TypedDecisionsPolicy",
    "LayaPolicy",
    "CachingSuitePolicy",
    "CachingTrackPolicy",
    "SUITES",
    "get_suite",
    "load_benchmark_items",
]
