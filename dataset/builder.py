from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union
from datasets import load_dataset

class IDatasetBuilder(Protocol):
    """The bedrock contract of the Dataset boundary."""
    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ...

def inject_abstention(
    records: List[Dict[str, Any]],
    rate: float = 0.15,
    token: str = "__insufficient_evidence__",
    desc: str = "insufficient evidence or none of the above options apply",
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
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

def split_train_val(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.08,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    gen = rng if rng is not None else random.Random(42)
    shuffled = list(records)
    gen.shuffle(shuffled)
    n_val = max(10, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]

def save_jsonl(path: Union[str, Path], records: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

class DatasetBuilder(IDatasetBuilder):
    def __init__(self, data_dir: Union[str, Path], seed: int = 42, blacklist: Optional[set] = None):
        self.data_dir = Path(data_dir)
        self.rng = random.Random(seed)
        self.blacklist = {b.strip().lower() for b in blacklist} if blacklist else set()

    def _build_choice_set(
        self, path: str, sub: Any, split: str, formatter: Any, label_extractor: Any,
        qname: str, instr: str, limit: int, opts_dict: Any = None,
        filter_fn: Optional[Any] = None, per_class_limit: int = 0
    ) -> List[Dict]:
        ds = load_dataset(path, sub, split=split) if sub else load_dataset(path, split=split)
        feat = ds.features.get(label_extractor) if isinstance(label_extractor, str) else None
        if opts_dict:
            labels = opts_dict
        elif hasattr(feat, "names") and feat.names:
            labels = list(feat.names)
        elif isinstance(label_extractor, str):
            labels = sorted(list(set(ds[label_extractor])))
        else:
            labels = sorted(list(set(label_extractor(row) for row in ds.select(range(min(len(ds), 500))))))
        opt_keys = list(labels.keys()) if isinstance(labels, dict) else labels
        records, counts = [], {}
        for row in ds:
            if filter_fn and not filter_fn(row):
                continue
            text = formatter(row).strip() if callable(formatter) else str(row.get(formatter, "")).strip()
            if not text or (self.blacklist and text.lower() in self.blacklist):
                continue
            raw = label_extractor(row) if callable(label_extractor) else row.get(label_extractor)
            lbl_str = opt_keys[raw] if isinstance(raw, int) and isinstance(labels, dict) else (labels[raw] if isinstance(raw, int) else str(raw))
            if lbl_str not in opt_keys:
                continue
            if per_class_limit > 0:
                if counts.get(lbl_str, 0) >= per_class_limit:
                    continue
                counts[lbl_str] = counts.get(lbl_str, 0) + 1
            records.append({"state": text, "questions": [[qname, "choice", opt_keys.index(lbl_str), labels, instr]]})
            if limit > 0 and len(records) >= limit:
                break
        return records

    def _parse_typed_decisions(self, ds: Any) -> List[Dict[str, Any]]:
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

    def _extract_glaive_tools(self, streaming_ds: Any, limit: int = 5000) -> List[Dict[str, Any]]:
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
        td = self._parse_typed_decisions(load_dataset("LocalLLaMA/typed-decisions", "all", split="train")) * 3
        tools = self._extract_glaive_tools(load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True), 5000)
        massive = self._build_choice_set("mteb/amazon_massive_intent", "en", "train", "text", "label_text", "action", "What is the user's intent in this utterance?", 6000)
        ag_news = self._build_choice_set("fancyzhx/ag_news", None, "train", "text", "label", "label", "What is the topic of the article?", 6000, per_class_limit=1500)
        spam = self._build_choice_set("SetFit/enron_spam", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nMessage: {(r.get('message') or '')[:2000]}", lambda r: "spam" if int(r.get("label", 0)) == 1 else "legitimate", "is_spam", "Is this email unsolicited spam or bulk marketing?", 3000)
        phish = self._build_choice_set("zefang-liu/phishing-email-dataset", None, "train", lambda r: f"Email:\n{(r.get('Email Text') or '')[:2000]}", lambda r: "phishing" if r.get("Email Type") == "Phishing Email" else "safe", "is_phishing", "Is this email a phishing or scam attempt?", 3000)
        support = self._build_choice_set("Tobi-Bueck/customer-support-tickets", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nBody: {(r.get('body') or '')[:2000]}", "queue", "queue", "Which support queue should handle this ticket?", 5000, filter_fn=lambda r: r.get("language") == "en" and r.get("body"), per_class_limit=500)
        emotion = self._build_choice_set("dair-ai/emotion", "split", "train", "text", "label", "emotion", "Which emotion is most strongly expressed in text?", 3000)
        banking = self._build_choice_set("mteb/banking77", None, "train", "text", lambda r: str(r.get("label_text", "")).replace("_", " "), "intent", "What is the primary customer inquiry or banking request?", 6000)
        safety_deepset = self._build_choice_set(
            "deepset/prompt-injections", None, "train", "text",
            lambda r: "quarantine_threat" if int(r.get("label", 0)) == 1 else "allow",
            "guardrail_action", "Determine the safety policy action for this user input.",
            1000
        )
        safety_jailbreak = self._build_choice_set(
            "jackhhao/jailbreak-classification", None, "train", "prompt",
            lambda r: "quarantine_threat" if str(r.get("type", "")).lower() == "jailbreak" else "allow",
            "guardrail_action", "Determine the safety policy action for this user input.",
            1500
        )
        safety = safety_deepset + safety_jailbreak

        foundation_path = self.data_dir / "train.jsonl"
        replay = []
        if foundation_path.exists():
            with open(foundation_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.rng.shuffle(lines)
                for line in lines[:5000]: replay.append(json.loads(line))

        all_samples = td + tools + massive + ag_news + spam + phish + support + emotion + banking + safety + replay
        all_samples = inject_abstention(all_samples, rate=0.15, rng=self.rng)
        return split_train_val(all_samples, 0.08, self.rng)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build curriculum datasets")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "nanollm" / "training" / "data"))
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    builder = DatasetBuilder(out_p)
    train_recs, val_recs = builder.build()
    save_jsonl(out_p / "train_adapt.jsonl", train_recs)
    save_jsonl(out_p / "val_adapt.jsonl", val_recs)
    print(f"Built curriculum: {len(train_recs)} train, {len(val_recs)} val -> {out_p}")
