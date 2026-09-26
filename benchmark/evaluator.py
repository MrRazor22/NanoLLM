from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union
import numpy as np
from nanollm.inference.schema import Choice
from benchmark.dataset import IBenchmarkDataset

DecideFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]

class IEvaluator(Protocol):
    def evaluate(self, source: Union[IBenchmarkDataset, Sequence[Dict[str, Any]]], limit: Optional[int] = None) -> Dict[str, Any]: ...

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

    def evaluate(self, source: Union[IBenchmarkDataset, Sequence[Dict[str, Any]]], limit: Optional[int] = None) -> Dict[str, Any]:
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

                conf = pred_obj.get("confidence", 0.0) if isinstance(pred_obj, dict) else 0.0
                all_confs.append(conf)
                all_briers.append((conf - hit) ** 2)

            if self.log_interval > 0 and (i + 1) % self.log_interval == 0:
                cur_acc = 100.0 * np.mean(all_hits) if all_hits else 0.0
                print(f"[{self.name}] [{i+1:5d}/{len(items)}] Acc: {cur_acc:.1f}%", flush=True)

        total_correct = sum(s["correct"] for s in stats.values())
        total_eval = sum(s["total"] for s in stats.values())
        overall_acc = (total_correct / total_eval) if total_eval > 0 else 0.0

        return {
            "evaluator": self.name,
            "overall_acc": overall_acc,
            "total_correct": total_correct,
            "total_questions": total_eval,
            "mean_confidence": float(np.mean(all_confs)) if all_confs else 0.0,
            "brier_score": float(np.mean(all_briers)) if all_briers else 0.0,
            "by_cat": {c: (s["correct"] / s["total"]) if s["total"] > 0 else 0.0 for c, s in stats.items()},
            "cat_counts": {c: s["total"] for c, s in stats.items()},
            "categories": {
                c: {
                    "accuracy": (s["correct"] / s["total"]) if s["total"] > 0 else 0.0,
                    "total": s["total"]
                }
                for c, s in stats.items()
            }
        }

__all__ = ["IEvaluator", "ModelEvaluator", "DecideFn"]
