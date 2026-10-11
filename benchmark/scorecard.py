from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable
from inference import DecisionCase, IDecisionEngine

BenchmarkItem = DecisionCase
SCORES_FILE = Path(__file__).resolve().parent / "scores.json"

@dataclass
class CategoryHits:
    correct: int = 0
    total: int = 0
    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

@dataclass(frozen=True)
class BenchmarkReport:
    model: str
    overall: float
    count: int
    p50_ms: float = 0.0
    slices: Mapping[str, float] = ()
    counts: Mapping[str, int] = ()

@runtime_checkable
class IScorecard(Protocol):
    def evaluate(self, engine: IDecisionEngine, items: Sequence[BenchmarkItem], model_name: str = "Piqo") -> BenchmarkReport: ...
    def record(self, report: BenchmarkReport) -> None: ...
    def render(self) -> None: ...

class Scorecard(IScorecard):
    """Sovereign benchmark runner: dynamically evaluates engines across all tracks and maintains ledger."""

    def __init__(
        self,
        scores_path: Optional[Path] = None,
        show_progress: bool = True,
        log_interval_sec: float = 3.0,
    ):
        self.scores_path = Path(scores_path or SCORES_FILE)
        self.show_progress = show_progress
        self.log_interval_sec = log_interval_sec

    def load_ledger(self) -> Dict[str, dict]:
        if not self.scores_path.exists(): return {}
        try: return json.loads(self.scores_path.read_text(encoding="utf-8"))
        except Exception: return {}

    def record(self, report: BenchmarkReport) -> None:
        ledger = self.load_ledger()
        for cat, acc in report.slices.items():
            entry = ledger.setdefault(cat, {})
            entry["count"], entry[report.model] = report.counts.get(cat, 0), round(acc, 4)
        overall = ledger.setdefault("OVERALL", {})
        overall["count"], overall[report.model] = report.count, round(report.overall, 4)
        lat = ledger.setdefault("LATENCY_P50_MS", {})
        lat["count"], lat[report.model] = "-", round(report.p50_ms, 2)
        self.scores_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    def evaluate(self, engine: IDecisionEngine, items: Sequence[BenchmarkItem], model_name: str = "Piqo") -> BenchmarkReport:
        stats: Dict[str, CategoryHits] = {}
        latencies, hits = [], []
        total_items = len(items)
        start_time = time.perf_counter()
        last_log = start_time

        for idx, item in enumerate(items, 1):
            cat = item.category or "general"
            if cat not in stats: stats[cat] = CategoryHits()
            res = engine.decide(item.state, item.questions)
            if res.latency_ms: latencies.append(res.latency_ms)

            q_by_name = {q.name: q for q in item.questions}
            def _matches(qid: str, target) -> bool:
                if qid not in res.answers: return False
                ans_val = str(res.answers[qid].value)
                if ans_val == str(target): return True
                q = q_by_name.get(qid)
                if hasattr(q, "options"):
                    keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                    if isinstance(target, int) and 0 <= target < len(keys): return ans_val == str(keys[target])
                    if str(target).isdigit() and 0 <= int(target) < len(keys): return ans_val == str(keys[int(target)])
                return False

            match = bool(item.gold) and all(_matches(qid, tgt) for qid, tgt in item.gold.items())
            hits.append(1.0 if match else 0.0)
            stats[cat].total += 1
            if match: stats[cat].correct += 1

            now = time.perf_counter()
            if self.show_progress and (now - last_log >= self.log_interval_sec or idx == total_items):
                p50 = float(sorted(latencies)[len(latencies) // 2]) if latencies else 0.0
                acc = (sum(hits) / max(1, len(hits))) * 100.0
                speed = idx / max(0.001, now - start_time)
                print(f"[{model_name}] {idx}/{total_items} ({idx*100.0/total_items:.1f}%) | Acc: {acc:.1f}% | P50: {p50:.1f}ms | {speed:.1f} it/s", flush=True)
                last_log = now

        overall = sum(hits) / max(1, len(hits))
        p50 = float(sorted(latencies)[len(latencies) // 2]) if latencies else 0.0
        return BenchmarkReport(
            model=model_name,
            overall=overall,
            count=len(items),
            p50_ms=p50,
            slices={c: h.accuracy for c, h in stats.items()},
            counts={c: h.total for c, h in stats.items()},
        )

    def render(self) -> None:
        ledger = self.load_ledger()
        if not ledger:
            print("\nLeaderboard is empty. Run: python -m benchmark\n")
            return

        models = sorted({m for entry in ledger.values() for m in entry if m != "count"})
        if "Piqo" in models:
            models.remove("Piqo")
            models.insert(0, "Piqo")

        tracks = [t for t in sorted(ledger.keys()) if t not in ("OVERALL", "LATENCY_P50_MS")]
        if "OVERALL" in ledger:
            tracks.append("OVERALL")
        if "LATENCY_P50_MS" in ledger:
            tracks.append("LATENCY_P50_MS")

        hdr = f"{'Benchmark Track / Metric':<32} | {'Count':>6} | " + " | ".join(f"{m:>14}" for m in models)
        sep = "-" * len(hdr)
        print("\n" + "=" * len(hdr) + "\n" + hdr + "\n" + sep)
        for track in tracks:
            entry = ledger.get(track, {})
            cnt = entry.get("count", "-")
            if track == "LATENCY_P50_MS":
                cols = [f"{entry[m]:12.2f} ms" if m in entry else f"{'-':>14}" for m in models]
            else:
                cols = [f"{entry[m]*100:13.1f}%" if m in entry else f"{'-':>14}" for m in models]
            if track in ("OVERALL", "LATENCY_P50_MS"):
                print(sep)
            print(f"{track:<32} | {str(cnt):>6} | " + " | ".join(cols))
        print("=" * len(hdr) + "\n")

__all__ = ["CategoryHits", "BenchmarkItem", "BenchmarkReport", "IScorecard", "Scorecard", "SCORES_FILE"]
