import json
from pathlib import Path
from typing import Any, Dict, List
from benchmark.dataset.dataset import ADAPTED_DIR, IDataSource

class TypedDecisionsSource(IDataSource):
    name = "typed_decisions"
    display_name = "3. Canonical Typed Decisions"
    category_labels: Dict[str, str] = {
        "agent_trace_observability": "Agent Observability",
        "customer_service": "Customer Service",
        "invoice_processing": "Invoice Processing",
        "security_incidents": "Security Incidents",
    }

    def extract(self) -> List[Dict[str, Any]]:
        local_file = ADAPTED_DIR / "typed_decisions.json"
        if local_file.exists():
            with open(local_file, "r", encoding="utf-8") as f:
                return json.load(f)

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

        if items:
            ADAPTED_DIR.mkdir(parents=True, exist_ok=True)
            with open(local_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)

        return items

__all__ = ["TypedDecisionsSource"]
