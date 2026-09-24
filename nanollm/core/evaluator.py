from typing import Any, Callable, Dict, List, Protocol

DecideFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]

class IEvaluator(Protocol):
    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...

class ModelEvaluator:
    def __init__(self, name: str, decide_fn: DecideFn):
        self.name = name
        self.decide_fn = decide_fn

    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        stats: Dict[str, Dict[str, int]] = {}
        for item in items:
            cat = item.get("category", "general")
            if cat not in stats:
                stats[cat] = {"correct": 0, "total": 0}
            preds = self.decide_fn(item["state"], item["questions"])
            for qid, gold in item["gold"].items():
                stats[cat]["total"] += 1
                if preds.get(qid) == gold["label"]:
                    stats[cat]["correct"] += 1

        total_correct = sum(s["correct"] for s in stats.values())
        total_q = sum(s["total"] for s in stats.values())
        return {
            "name": self.name,
            "overall_acc": total_correct / max(1, total_q),
            "total_correct": total_correct,
            "total_questions": total_q,
            "by_cat": {c: s["correct"] / s["total"] for c, s in stats.items()},
        }
