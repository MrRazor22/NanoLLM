import json, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from datasets import load_dataset
from nanollm.training.policies.builder import split_train_val
from nanollm.training.policies.dataset import extract_glaive_tools, inject_abstention, parse_typed_decisions

class ICurriculum(Protocol):
    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ...

class AdaptationCurriculum:
    def __init__(self, data_dir: str, seed: int = 42, blacklist: Optional[set] = None):
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

    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        td = parse_typed_decisions(load_dataset("LocalLLaMA/typed-decisions", "all", split="train")) * 3
        tools = extract_glaive_tools(load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True), 5000, self.rng)
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
    from nanollm.training.policies.builder import save_jsonl
    parser = argparse.ArgumentParser(description="Build curriculum datasets")
    parser.add_argument("--curriculum", type=str, default="adaptation", choices=["adaptation", "foundation"])
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data"))
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    cur = AdaptationCurriculum(str(out_p)) if args.curriculum == "adaptation" else FoundationCurriculum()
    train_recs, val_recs = cur.build()
    save_jsonl(str(out_p / f"train_{args.curriculum}.jsonl"), train_recs)
    save_jsonl(str(out_p / f"val_{args.curriculum}.jsonl"), val_recs)
    print(f"Built {args.curriculum} curriculum: {len(train_recs)} train, {len(val_recs)} val -> {out_p}")

