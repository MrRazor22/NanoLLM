from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union
import numpy as np
from nanollm.inference.schema import Choice
from benchmark.suites.base_policy import ISuitePolicy

DecideFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]

class IEvaluator(Protocol):
    def evaluate(self, source: Union[ISuitePolicy, Sequence[Dict[str, Any]]], limit: Optional[int] = None) -> Dict[str, Any]: ...

class ModelEvaluator:
    def __init__(self, name: str, decide_fn: DecideFn, log_interval: int = 0):
        self.name = name
        self.decide_fn = decide_fn
        self.log_interval = log_interval

    @classmethod
    def from_engine(cls, name: str, engine: Any, log_interval: int = 0) -> "ModelEvaluator":
        def decide(state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
            qs: List[Choice] = []
            for qid, spec in questions.items():
                opts = spec.get("options") or spec.get("criteria", {})
                qs.append(Choice(qid, opts, instruction=spec.get("instructions")))
            res = engine.decide(state, qs)
            return {
                qid: {
                    "choice": str(res.answers[qid].choice),
                    "confidence": getattr(res.answers[qid], "confidence", 0.0),
                    "probabilities": getattr(res.answers[qid], "probabilities", {}),
                }
                for qid in questions if qid in res.answers
            }
        return cls(name, decide, log_interval=log_interval)

    def __or__(self, layer: Any) -> Any:
        return layer.attach(self) if hasattr(layer, "attach") else layer(self)

    def evaluate(self, source: Union[ISuitePolicy, Sequence[Dict[str, Any]]], limit: Optional[int] = None) -> Dict[str, Any]:
        items = source.load(limit=limit) if hasattr(source, "load") else (source[:limit] if limit else source)
        stats: Dict[str, Dict[str, int]] = {}
        all_hits: List[float] = []
        all_confs: List[float] = []
        all_briers: List[float] = []

        for i, item in enumerate(items):
            cat = item.get("category", "general")
            if cat not in stats:
                stats[cat] = {"correct": 0, "total": 0}
            preds = self.decide_fn(item["state"], item["questions"])

            for qid, gold in item["gold"].items():
                if qid not in preds:
                    continue
                stats[cat]["total"] += 1
                pred_obj = preds[qid]
                pred_choice = pred_obj["choice"] if isinstance(pred_obj, dict) else str(pred_obj)
                gold_label = str(gold.get("label", ""))
                hit = 1.0 if pred_choice == gold_label else 0.0
                stats[cat]["correct"] += int(hit)
                all_hits.append(hit)

                if isinstance(pred_obj, dict):
                    all_confs.append(float(pred_obj.get("confidence", 0.0)))
                    gold_probs, pred_probs = gold.get("probabilities", {}), pred_obj.get("probabilities", {})
                    if gold_probs and pred_probs:
                        keys = list(gold_probs.keys())
                        pv = np.array([pred_probs.get(k, 0.0) for k in keys], dtype=float)
                        gv = np.array([gold_probs.get(k, 0.0) for k in keys], dtype=float)
                        if pv.sum() > 0: pv /= pv.sum()
                        if gv.sum() > 0: gv /= gv.sum()
                        all_briers.append(float(np.sum((pv - gv) ** 2)))

            if self.log_interval > 0 and ((i + 1) % self.log_interval == 0 or (i + 1) == len(items)):
                tot_c = sum(s["correct"] for s in stats.values())
                tot_q = sum(s["total"] for s in stats.values())
                print(f"[{self.name}] [{i+1:5d}/{len(items)}] Acc: {(tot_c / max(1, tot_q)) * 100.0:.1f}%", flush=True)

        tot_c = sum(s["correct"] for s in stats.values())
        tot_q = sum(s["total"] for s in stats.values())
        report: Dict[str, Any] = {
            "name": self.name,
            "overall_acc": tot_c / max(1, tot_q) if tot_q else 0.0,
            "total_correct": tot_c,
            "total_questions": tot_q,
            "by_cat": {c: s["correct"] / max(1, s["total"]) for c, s in stats.items()},
            "cat_counts": {c: s["total"] for c, s in stats.items()},
        }
        if all_briers:
            report["brier_score"] = float(np.mean(all_briers))
        return report

__all__ = ["DecideFn", "IEvaluator", "ModelEvaluator"]
