import json, random, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from datasets import load_dataset
from nanollm.data.builder import split_train_val

class ICurriculum(Protocol):
    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ...

class AdaptationCurriculum:
    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.rng = random.Random(seed)

    def _build_typed_decisions(self) -> List[Dict]:
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
        return records

    def _build_choice_set(self, path: str, sub: Any, split: str, tcol: str, lcol: str, qname: str, instr: str, limit: int, opts_dict: Any = None) -> List[Dict]:
        ds = load_dataset(path, sub, split=split) if sub else load_dataset(path, split=split)
        labels = opts_dict if opts_dict else sorted(list(set(ds[lcol])))
        opt_keys = list(labels.keys()) if isinstance(labels, dict) else labels
        records = []
        for row in ds:
            text, raw = str(row[tcol]).strip(), row[lcol]
            if not text: continue
            lbl_str = opt_keys[raw] if isinstance(raw, int) and isinstance(labels, dict) else (labels[raw] if isinstance(raw, int) else str(raw))
            if lbl_str not in opt_keys: continue
            records.append({"state": text, "questions": [[qname, "choice", opt_keys.index(lbl_str), labels, instr]]})
            if len(records) >= limit: break
        return records

    def _build_glaive_tools(self, limit: int = 5000) -> List[Dict]:
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
        raw, tool_registry = [], {}
        for row in ds:
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
                for d in self.rng.sample(distractors, min(len(distractors), 5 - len(candidates))):
                    candidates[d] = tool_registry[d]
            opt_keys = list(candidates.keys())
            self.rng.shuffle(opt_keys)
            records.append({
                "state": query,
                "questions": [["tool", "choice", opt_keys.index(fn), {k: candidates[k] for k in opt_keys}, "Which function or tool should be invoked to handle this request?"]]
            })
            if len(records) >= limit: break
        return records

    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_td = self._build_typed_decisions()
        td = raw_td * 3
        tools = self._build_glaive_tools(5000)
        massive = self._build_choice_set("mteb/amazon_massive_intent", "en", "train", "text", "label_text", "action", "What is the user's intent in this utterance?", 8000)
        ag_opts = {
            "World": "international news, politics, conflicts", "Sports": "sports, games, athletes",
            "Business": "companies, markets, economy", "Sci/Tech": "science, technology, software, space"
        }
        ag_news = self._build_choice_set("fancyzhx/ag_news", None, "train", "text", "label", "label", "What is the topic of the article?", 6000, opts_dict=ag_opts)

        foundation_path = self.data_dir / "train.jsonl"
        replay = []
        if foundation_path.exists():
            with open(foundation_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.rng.shuffle(lines)
                for line in lines[:6000]: replay.append(json.loads(line))

        all_samples = td + tools + massive + ag_news + replay
        return split_train_val(all_samples, 0.08, self.rng)
