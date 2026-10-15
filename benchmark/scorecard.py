import statistics
import time
from typing import Dict, Optional, Protocol, Sequence, runtime_checkable
from benchmark.ledger import DiskLedgerPolicy, ILedgerPolicy
from benchmark.report import BenchmarkReport
from data.case import DecisionCase
from inference.decision_engine import IDecisionEngine

@runtime_checkable
class IScorecard(Protocol):
    """Core Primitive: Evaluates decision engines on benchmark items to produce verified score reports."""
    def evaluate(
        self,
        engine: IDecisionEngine,
        items: Sequence[DecisionCase],
        model_name: str = "Piqo",
        suite: str = "default",
        force: bool = False,
    ) -> BenchmarkReport: ...

class Scorecard(IScorecard):
    """Core Primitive Implementation: Evaluates decision engines on benchmark items."""

    def __init__(
        self,
        ledger: Optional[ILedgerPolicy] = None,
        show_progress: bool = True,
        log_interval_sec: float = 3.0,
    ):
        self.ledger = ledger if ledger is not None else DiskLedgerPolicy()
        self.show_progress = show_progress
        self.log_interval_sec = log_interval_sec

    def evaluate(
        self,
        engine: IDecisionEngine,
        items: Sequence[DecisionCase],
        model_name: str = "Piqo",
        suite: str = "default",
        force: bool = False,
    ) -> BenchmarkReport:
        if not force and self.ledger is not None:
            cached = self.ledger.get(model_name, suite=suite)
            if cached is not None:
                return BenchmarkReport(
                    model=model_name,
                    overall=cached.get("OVERALL", 0.0),
                    count=len(items),
                    p50_ms=cached.get("LATENCY_P50_MS", 0.0),
                    slices={k: v for k, v in cached.items() if k not in ("OVERALL", "LATENCY_P50_MS")},
                    suite=suite,
                )

        cat_correct: Dict[str, int] = {}
        cat_total: Dict[str, int] = {}
        latencies, hits = [], []
        total_items = len(items)
        start_time = time.perf_counter()
        last_log = start_time

        for idx, item in enumerate(items, 1):
            cat = item.category or "general"
            t_call = time.perf_counter()
            res = engine.decide(item.state, item.questions)
            dt_ms = res.latency_ms if res.latency_ms > 0 else (time.perf_counter() - t_call) * 1000.0
            latencies.append(dt_ms)

            for q in item.questions:
                if q.target is not None and q.name in res.answers:
                    is_hit = q.matches(res.answers[q.name])
                    hits.append(1.0 if is_hit else 0.0)
                    cat_total[cat] = cat_total.get(cat, 0) + 1
                    if is_hit:
                        cat_correct[cat] = cat_correct.get(cat, 0) + 1

            now = time.perf_counter()
            if self.show_progress and (now - last_log >= self.log_interval_sec or idx == total_items):
                p50 = float(statistics.median(latencies)) if latencies else 0.0
                acc = (sum(hits) / max(1, len(hits))) * 100.0
                speed = idx / max(0.001, now - start_time)
                print(f"[{model_name}] {idx}/{total_items} ({idx*100.0/total_items:.1f}%) | Acc: {acc:.1f}% | P50: {p50:.1f}ms | {speed:.1f} it/s", flush=True)
                last_log = now

        overall = sum(hits) / max(1, len(hits))
        p50 = float(statistics.median(latencies)) if latencies else 0.0
        report = BenchmarkReport(
            model=model_name,
            overall=overall,
            count=len(hits),
            p50_ms=p50,
            slices={c: cat_correct.get(c, 0) / cat_total[c] for c in cat_total},
            counts=cat_total,
            suite=suite,
        )
        if self.ledger is not None:
            self.ledger.put(report)
        return report
