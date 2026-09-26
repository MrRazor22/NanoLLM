import json
from typing import Any, Dict, List
from benchmark.suites.base_policy import BaseSuitePolicy

class TypedDecisionsPolicy(BaseSuitePolicy):
    name = "typed_decisions"

    def extract(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset
        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
        items = []
        for i in range(len(ds)):
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

__all__ = ["TypedDecisionsPolicy"]
