import json
from typing import Any, Dict, List
from datasets import load_dataset
from nanollm.training.dataset.sources.generic import ISourceAdapter

class TypedDecisionsSource(ISourceAdapter):
    def __init__(self, repeat: int = 3):
        self.repeat = repeat

    def extract(self) -> List[Dict[str, Any]]:
        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
        records = []
        for row in ds:
            q_dict, g_dict = json.loads(row["questions"]), json.loads(row["gold"])
            qs = []
            for name, spec in q_dict.items():
                t, ins, crit = spec.get("type"), spec.get("instructions"), spec.get("criteria", {})
                gold = g_dict.get(name, {})
                if t == "choice":
                    opts = crit if isinstance(crit, dict) else list(crit)
                    opt_keys = list(opts.keys()) if isinstance(opts, dict) else opts
                    lbl = gold.get("label")
                    if lbl in opt_keys: qs.append([name, "choice", opt_keys.index(lbl), opts, ins])
                elif t == "noul":
                    crit_dict = crit if isinstance(crit, dict) and crit else {"false": "no, condition does not hold", "true": "yes, condition holds"}
                    target_bool = gold.get("label") == "true" or gold.get("noul", 0) >= 0.5
                    qs.append([name, "choice", 1 if target_bool else 0, crit_dict, ins])
                elif t == "score":
                    opts = {str(i): c for i, c in enumerate(crit)} if isinstance(crit, list) else crit
                    lbl = str(gold.get("label"))
                    if lbl in opts: qs.append([name, "choice", list(opts.keys()).index(lbl), opts, ins])
            if qs: records.append({"state": row["state"], "questions": qs})
        return records * self.repeat
