import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence
import numpy as np
from nanollm.inference.schema import Choice

DecideFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]

class IEvaluator(Protocol):
    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, Any]: ...

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

    def add(self, layer: Any, **kwargs: Any) -> Any:
        if isinstance(layer, type):
            return layer(self, **kwargs)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> Any:
        return self.add(layer)

    def evaluate(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        stats: Dict[str, Dict[str, Any]] = {}
        all_hits: List[float] = []
        all_confs: List[float] = []
        all_briers: List[float] = []
        total_items = len(items)

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
                    conf = float(pred_obj.get("confidence", 0.0))
                    all_confs.append(conf)
                    gold_probs = gold.get("probabilities", {})
                    pred_probs = pred_obj.get("probabilities", {})
                    if gold_probs and pred_probs:
                        keys = list(gold_probs.keys())
                        pv = np.array([pred_probs.get(k, 0.0) for k in keys], dtype=float)
                        gv = np.array([gold_probs.get(k, 0.0) for k in keys], dtype=float)
                        if pv.sum() > 0:
                            pv /= pv.sum()
                        if gv.sum() > 0:
                            gv /= gv.sum()
                        all_briers.append(float(np.sum((pv - gv) ** 2)))

            if self.log_interval > 0 and ((i + 1) % self.log_interval == 0 or (i + 1) == total_items):
                tot_c = sum(s["correct"] for s in stats.values())
                tot_q = sum(s["total"] for s in stats.values())
                acc = (tot_c / max(1, tot_q)) * 100.0
                print(f"[{self.name}] [{i+1:5d}/{total_items}] Acc: {acc:.1f}%", flush=True)

        tot_c = sum(s["correct"] for s in stats.values())
        tot_q = sum(s["total"] for s in stats.values())
        arr_hits = np.array(all_hits, dtype=float)
        arr_confs = np.array(all_confs, dtype=float)

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
        if len(arr_confs) > 0 and np.any(arr_confs > 0):
            report["ece"] = _compute_ece(arr_confs, arr_hits)
            report["selective_classification"] = _compute_selective(arr_confs, arr_hits)
        return report

def _compute_ece(confidences: np.ndarray, hits: np.ndarray, num_bins: int = 10) -> float:
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece_val = 0.0
    total = len(confidences)
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(hits[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece_val += (bin_count / total) * abs(bin_acc - bin_conf)
    return float(ece_val)

def _compute_selective(confidences: np.ndarray, hits: np.ndarray, coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> List[Dict[str, float]]:
    n = len(confidences)
    if n == 0:
        return []
    order = np.argsort(-confidences)
    sorted_hits = hits[order]
    sorted_conf = confidences[order]
    return [
        {
            "coverage": float(cov),
            "retained_accuracy": round(float(np.mean(sorted_hits[:max(1, int(round(cov * n)))])), 4),
            "selective_risk": round(1.0 - float(np.mean(sorted_hits[:max(1, int(round(cov * n)))])), 4),
            "threshold": round(float(sorted_conf[max(1, int(round(cov * n))) - 1]), 4),
        }
        for cov in coverages
    ]

def load_benchmark_items(track: str = "typed_decisions", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data_dir = Path(__file__).resolve().parent / "data"
    if track in ("typed_decisions", "typed", "verdict"):
        from datasets import load_dataset
        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
        n = len(ds) if limit is None else min(len(ds), limit)
        items = []
        for i in range(n):
            row = ds[i]
            q_defs = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
            norm_qs = {}
            for qid, spec in q_defs.items():
                t, ins, crit = spec.get("type"), spec.get("instructions"), spec.get("criteria")
                if t == "choice":
                    opts = crit if isinstance(crit, dict) else {str(j): c for j, c in enumerate(crit or [])}
                elif t == "noul":
                    opts = crit if isinstance(crit, dict) and crit else {"false": "no, condition does not hold", "true": "yes, condition holds"}
                elif t == "score":
                    opts = {str(j): c for j, c in enumerate(crit)} if isinstance(crit, list) else (crit or {})
                else:
                    opts = crit or {"false": "no", "true": "yes"}
                norm_qs[qid] = {"instructions": ins, "options": opts}
            items.append({
                "id": row.get("id", f"case_{i}"),
                "category": row.get("workflow", "general"),
                "state": str(row["state"]),
                "questions": norm_qs,
                "gold": json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"],
            })
        return items
    elif track == "abstention":
        items = []
        for fn in ("slice_missing_option.jsonl", "slice_distant_oos.jsonl"):
            fp = data_dir / fn
            if not fp.exists():
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    items.append({
                        "id": r.get("id", ""),
                        "category": fp.stem,
                        "state": r.get("text", ""),
                        "questions": {
                            "intent": {
                                "type": "choice",
                                "instructions": r.get("question", ""),
                                "criteria": {c["id"]: c["description"] for c in r.get("candidates", [])},
                            }
                        },
                        "gold": {"intent": {"label": r.get("target_id", "__insufficient_evidence__")}},
                    })
        return items[:limit] if limit else items
    else:
        fn = "benchmark.json" if track == "agentic" else "laya_benchmark.json"
        fp = data_dir / fn
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[:limit] if limit else data
