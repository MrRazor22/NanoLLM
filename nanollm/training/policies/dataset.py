from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
        self._cache: Dict[int, Tuple[List[int], List[Any]]] = {}

    def __call__(self, batch: Sequence[Any]) -> Dict[str, Any]:
        items = []
        for s in batch:
            k = id(s)
            cached = self._cache.get(k)
            if cached is None:
                cached = self._cache[k] = (
                    s if isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], list)
                    else self.assembler.render_sample(s.state, s.questions)
                )
            items.append(cached)
        return self.assembler.assemble_batch(items)

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

def inject_abstention(
    records: List[Dict[str, Any]],
    rate: float = 0.15,
    token: str = "__insufficient_evidence__",
    desc: str = "insufficient evidence or none of the above options apply",
    rng: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    import random
    gen = rng if rng is not None else random.Random(42)
    out = []
    for rec in records:
        if gen.random() >= rate:
            out.append(rec)
            continue
        qs = []
        for q in rec.get("questions", []):
            opts_raw = q[3] if len(q) > 3 and q[3] is not None else None
            options = dict(opts_raw) if isinstance(opts_raw, dict) else (list(opts_raw) if opts_raw is not None else None)
            name, q_type, target, instr = q[0], q[1], q[2], q[4] if len(q) > 4 else None
            if q_type == "choice" and isinstance(options, dict) and len(options) >= 2:
                keys = list(options.keys())
                target_key = keys[target] if isinstance(target, int) and target < len(keys) else None
                if target_key:
                    del options[target_key]
                    options[token] = desc
                    new_keys = list(options.keys())
                    gen.shuffle(new_keys)
                    qs.append([name, "choice", new_keys.index(token), {k: options[k] for k in new_keys}, instr])
                    continue
            qs.append(q)
        out.append({"state": rec["state"], "questions": qs})
    return out

