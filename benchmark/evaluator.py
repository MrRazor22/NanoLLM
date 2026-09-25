from typing import Any, Callable, Dict, List, Protocol, Sequence
from nanollm.inference.schema import Choice

DecideFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]

class IEvaluator(Protocol):
    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...

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
                crit = spec.get("criteria", {})
                opts = {str(i): c for i, c in enumerate(crit)} if isinstance(crit, list) else crit
                qs.append(Choice(qid, opts, instruction=spec.get("instructions")))
            res = engine.decide(state, qs)
            return {qid: res.answers[qid].choice for qid in questions}
        return cls(name, decide, log_interval=log_interval)

    def add(self, layer: Any, **kwargs: Any) -> Any:
        if isinstance(layer, type):
            return layer(self, **kwargs)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> Any:
        return self.add(layer)

    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        stats: Dict[str, Dict[str, int]] = {}
        total_items = len(items)
        for i, item in enumerate(items):
            cat = item.get("category", "general")
            if cat not in stats:
                stats[cat] = {"correct": 0, "total": 0}
            preds = self.decide_fn(item["state"], item["questions"])
            for qid, gold in item["gold"].items():
                stats[cat]["total"] += 1
                if preds.get(qid) == gold["label"]:
                    stats[cat]["correct"] += 1
            if self.log_interval > 0 and ((i + 1) % self.log_interval == 0 or (i + 1) == total_items):
                tot_c = sum(s["correct"] for s in stats.values())
                tot_q = sum(s["total"] for s in stats.values())
                acc = (tot_c / max(1, tot_q)) * 100.0
                print(f"[{self.name}] Evaluated [{i+1:5d}/{total_items}] Acc: {acc:.1f}%", flush=True)

        total_correct = sum(s["correct"] for s in stats.values())
        total_q = sum(s["total"] for s in stats.values())
        return {
            "name": self.name,
            "overall_acc": total_correct / max(1, total_q),
            "total_correct": total_correct,
            "total_questions": total_q,
            "by_cat": {c: s["correct"] / s["total"] for c, s in stats.items()},
        }
