from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable
from data import DecisionCase
from inference import IDecisionEngine, ProfilingLayer

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
        ledger_policy: Optional["ILedgerPolicy"] = None,
        show_progress: bool = True,
        log_interval_sec: float = 3.0,
    ):
        from benchmark.ledger import DiskLedgerPolicy
        self.ledger_policy = ledger_policy or DiskLedgerPolicy()
        self.show_progress = show_progress
        self.log_interval_sec = log_interval_sec

    def signature_for(self, items: Sequence[BenchmarkItem]) -> str:
        import hashlib
        raw = f"{len(items)}|{items[0].state[:30] if items else ''}|{items[-1].state[:30] if items else ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def record(self, report: BenchmarkReport, dataset_signature: Optional[str] = None) -> None:
        self.ledger_policy.put(report, signature=dataset_signature)

    def evaluate(self, engine: IDecisionEngine, items: Sequence[BenchmarkItem], model_name: str = "Piqo", force: bool = False) -> BenchmarkReport:
        engine = engine if isinstance(engine, ProfilingLayer) else (engine | ProfilingLayer())
        sig = self.signature_for(items)
        if not force:
            cached = self.ledger_policy.get(model_name, signature=sig)
            if cached is not None:
                print(f"Competitor '{model_name}' already evaluated on this dataset (reusing cached scores; pass --force to re-run).", flush=True)
                return BenchmarkReport(
                    model=model_name,
                    overall=cached.get("OVERALL", 0.0),
                    count=len(items),
                    p50_ms=cached.get("LATENCY_P50_MS", 0.0),
                    slices={k: v for k, v in cached.items() if k not in ("OVERALL", "LATENCY_P50_MS")},
                    counts={},
                )

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
            def _matches(qid: str, target: Union[str, int, float, bool]) -> bool:
                if qid not in res.answers: return False
                ans_val = str(res.answers[qid].value)
                if ans_val == str(target): return True
                q = q_by_name.get(qid)
                if hasattr(q, "options"):
                    keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                    if isinstance(target, int) and 0 <= target < len(keys): return ans_val == str(keys[target])
                    if str(target).isdigit() and 0 <= int(target) < len(keys): return ans_val == str(keys[int(target)])
                return False

            if item.gold:
                for qid, tgt in item.gold.items():
                    m = _matches(qid, tgt)
                    hits.append(1.0 if m else 0.0)
                    stats[cat].total += 1
                    if m: stats[cat].correct += 1

            now = time.perf_counter()
            if self.show_progress and (now - last_log >= self.log_interval_sec or idx == total_items):
                p50 = float(sorted(latencies)[len(latencies) // 2]) if latencies else 0.0
                acc = (sum(hits) / max(1, len(hits))) * 100.0
                speed = idx / max(0.001, now - start_time)
                print(f"[{model_name}] {idx}/{total_items} ({idx*100.0/total_items:.1f}%) | Acc: {acc:.1f}% | P50: {p50:.1f}ms | {speed:.1f} it/s", flush=True)
                last_log = now

        overall = sum(hits) / max(1, len(hits))
        p50 = float(sorted(latencies)[len(latencies) // 2]) if latencies else 0.0
        report = BenchmarkReport(
            model=model_name,
            overall=overall,
            count=len(hits),
            p50_ms=p50,
            slices={c: h.accuracy for c, h in stats.items()},
            counts={c: h.total for c, h in stats.items()},
        )
        self.record(report, dataset_signature=sig)
        return report

    def render(self) -> None:
        ledger = self.ledger_policy.all()
        if not ledger:
            print("\nLeaderboard is empty. Run: python -m benchmark\n")
            return

        models = sorted({m for entry in ledger.values() for m in entry if m not in ("count", "signature")})
        if "Piqo" in models: models = ["Piqo"] + [m for m in models if m != "Piqo"]

        tracks = [t for t in sorted(ledger.keys()) if t not in ("OVERALL", "LATENCY_P50_MS", "DATASET")]
        for m in ("OVERALL", "LATENCY_P50_MS"):
            if m in ledger: tracks.append(m)

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
