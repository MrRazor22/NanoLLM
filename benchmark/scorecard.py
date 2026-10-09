import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence
from nanollm import Choice, IDecisionEngine

DEFAULT_BASELINES = Path(__file__).resolve().parent / "baselines" / "competitor_cache.json"

class Scorecard:
    """Evaluates candidate engines against benchmark tracks and renders scorecards."""

    def __init__(self, baselines_path: Optional[Path] = None):
        p = baselines_path or DEFAULT_BASELINES
        raw = json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}
        self.baselines = {k: v for k, v in raw.items() if k != "p50_latencies_ms"}
        self.p50s = raw.get("p50_latencies_ms", {})

    def evaluate(
        self,
        engine: IDecisionEngine,
        items: Sequence[Mapping[str, Any]],
        log_interval: int = 100,
    ) -> Dict[str, Any]:
        stats: Dict[str, Dict[str, int]] = {}
        latencies, hits = [], []
        t0 = time.perf_counter()

        for i, item in enumerate(items):
            cat = item.get("category", "general")
            if cat not in stats: stats[cat] = {"correct": 0, "total": 0}
            qs = [Choice(qid, spec.get("options") or spec.get("criteria", {}), instruction=spec.get("instructions")) for qid, spec in item["questions"].items()]
            res = engine.decide(item["state"], qs)
            if hasattr(res, "latency_ms") and res.latency_ms: latencies.append(res.latency_ms)

            match = all(str(res.answers[qid].choice) == str(gold.get("label") if isinstance(gold, dict) else gold) for qid, gold in item.get("gold", {}).items() if qid in res.answers)
            hits.append(1.0 if match else 0.0)
            stats[cat]["total"] += 1
            if match: stats[cat]["correct"] += 1

        overall = sum(hits) / max(1, len(hits))
        p50 = float(sorted(latencies)[len(latencies) // 2]) if latencies else 0.0
        return {
            "overall": overall,
            "count": len(items),
            "p50_ms": p50,
            "slices": {cat: d["correct"] / max(1, d["total"]) for cat, d in stats.items()},
        }

    def render(self, reports: Dict[str, Any], model_name: str = "NanoLLM") -> None:
        hdr = f"{'Benchmark Track':<32} | {'Count':>6} | {model_name:>12} | {'Verdict 2.0':>12} | {'Laya (421M)':>12}"
        print("\n" + "=" * len(hdr) + "\n" + hdr + "\n" + "-" * len(hdr))
        for track, rep in reports.items():
            base = self.baselines.get(track, {})
            v_val = base.get("Verdict 2.0 (151M)", {}).get("overall")
            l_val = base.get("Laya (421M)", {}).get("overall")
            v_str = f"{v_val*100:11.1f}%" if v_val is not None else "-"
            l_str = f"{l_val*100:11.1f}%" if l_val is not None else "-"
            print(f"{track:<32} | {rep['count']:>6} | {rep['overall']*100:11.1f}% | {v_str:>12} | {l_str:>12}")
        print("=" * len(hdr) + "\n")

__all__ = ["Scorecard"]
