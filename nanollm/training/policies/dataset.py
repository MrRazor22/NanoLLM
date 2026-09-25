from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import json
from nanollm.inference.policies.assembler import SlotAssembler

@dataclass(frozen=True)
class QuestionSpec:
    name: str
    q_type: str
    target: Any
    options: Optional[Any] = None
    instruction: Optional[str] = None

@dataclass(frozen=True)
class DecisionSample:
    state: str
    questions: List[QuestionSpec]

class MultiQuestionCollator:
    def __init__(self, tokenizer_or_assembler: Any):
        if hasattr(tokenizer_or_assembler, "assemble_batch"):
            self.assembler = tokenizer_or_assembler
        else:
            self.assembler = SlotAssembler(tokenizer_or_assembler)

    def __call__(self, batch: Sequence[DecisionSample]) -> Dict[str, Any]:
        return self.assembler.assemble_batch(batch)

def to_decision_sample(item: Dict[str, Any]) -> DecisionSample:
    specs = [
        QuestionSpec(
            name=q[0],
            q_type=q[1],
            target=q[2],
            options=q[3] if len(q) > 3 else None,
            instruction=q[4] if len(q) > 4 else None,
        )
        for q in item["questions"]
    ]
    return DecisionSample(state=item["state"], questions=specs)

def load_jsonl(path: str) -> List[DecisionSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(to_decision_sample(json.loads(line)))
    return samples

def parse_typed_decisions(ds: Any) -> List[Dict[str, Any]]:
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
    return records

def extract_glaive_tools(streaming_ds: Any, limit: int = 5000, rng: Optional[Any] = None) -> List[Dict[str, Any]]:
    import random, re
    generator = rng if rng is not None else random.Random(42)
    raw, tool_registry = [], {}
    for row in streaming_ds:
        system, chat = row["system"], row["chat"]
        fn_match = re.search(r'<functioncall>\s*\{\s*"name":\s*"([^"]+)"', chat)
        user_match = re.search(r'USER:\s*(.*?)(?=\n\s*(?:ASSISTANT:|<\|endoftext\|>|$))', chat, re.DOTALL)
        if not (fn_match and user_match): continue
        fn_name, user_query = fn_match.group(1), user_match.group(1).strip()
        tools = dict(re.findall(r'\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]+)"', system))
        if fn_name not in tools: continue
        tool_registry.update(tools)
        raw.append((user_query, fn_name, tools))
        if len(raw) >= limit * 2: break

    records = []
    all_names = list(tool_registry.keys())
    for query, fn, sample_tools in raw:
        candidates = dict(sample_tools)
        if len(candidates) < 5 and len(all_names) >= 5:
            distractors = [n for n in all_names if n != fn and n not in candidates]
            for d in generator.sample(distractors, min(len(distractors), 5 - len(candidates))):
                candidates[d] = tool_registry[d]
        opt_keys = list(candidates.keys())
        generator.shuffle(opt_keys)
        records.append({
            "state": query,
            "questions": [["tool", "choice", opt_keys.index(fn), {k: candidates[k] for k in opt_keys}, "Which function or tool should be invoked to handle this request?"]]
        })
        if len(records) >= limit: break
    return records
