import json, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from datasets import load_dataset
from nanollm.training.policies.builder import split_train_val
from nanollm.training.policies.dataset import extract_glaive_tools, parse_typed_decisions
from nanollm.training.policies.taxonomies import (
    AG_NEWS_TOPICS,
    EMOTION_CRITERIA,
    PHISHING_CRITERIA,
    SPAM_CRITERIA,
    SUPPORT_QUEUES,
)

class ICurriculum(Protocol):
    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ...

class AdaptationCurriculum:
    def __init__(self, data_dir: str, seed: int = 42, blacklist: Optional[set] = None):
        self.data_dir = Path(data_dir)
        self.rng = random.Random(seed)
        self.blacklist = {b.strip().lower() for b in blacklist} if blacklist else set()

    def _build_choice_set(
        self, path: str, sub: Any, split: str, formatter: Any, label_extractor: Any,
        qname: str, instr: str, limit: int, opts_dict: Any = None
    ) -> List[Dict]:
        ds = load_dataset(path, sub, split=split) if sub else load_dataset(path, split=split)
        labels = opts_dict if opts_dict else sorted(list(set(ds[label_extractor])))
        opt_keys = list(labels.keys()) if isinstance(labels, dict) else labels
        records = []
        for row in ds:
            text = formatter(row).strip() if callable(formatter) else str(row.get(formatter, "")).strip()
            if not text or (self.blacklist and text.lower() in self.blacklist):
                continue
            raw = label_extractor(row) if callable(label_extractor) else row.get(label_extractor)
            lbl_str = opt_keys[raw] if isinstance(raw, int) and isinstance(labels, dict) else (labels[raw] if isinstance(raw, int) else str(raw))
            if lbl_str not in opt_keys:
                continue
            records.append({"state": text, "questions": [[qname, "choice", opt_keys.index(lbl_str), labels, instr]]})
            if len(records) >= limit:
                break
        return records

    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        td = parse_typed_decisions(load_dataset("LocalLLaMA/typed-decisions", "all", split="train")) * 3
        tools = extract_glaive_tools(load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True), 5000, self.rng)
        massive = self._build_choice_set("mteb/amazon_massive_intent", "en", "train", "text", "label_text", "action", "What is the user's intent in this utterance?", 6000)
        ag_news = self._build_choice_set("fancyzhx/ag_news", None, "train", "text", "label", "label", "What is the topic of the article?", 4000, opts_dict=AG_NEWS_TOPICS)
        spam = self._build_choice_set("SetFit/enron_spam", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nMessage: {(r.get('message') or '')[:2000]}", lambda r: "true" if int(r.get("label", 0)) == 1 else "false", "is_spam", "Is this email unsolicited spam or bulk marketing?", 3000, opts_dict=SPAM_CRITERIA)
        phish = self._build_choice_set("zefang-liu/phishing-email-dataset", None, "train", lambda r: f"Email:\n{(r.get('Email Text') or '')[:2000]}", lambda r: "true" if r.get("Email Type") == "Phishing Email" else "false", "is_phishing", "Is this email a phishing or scam attempt?", 3000, opts_dict=PHISHING_CRITERIA)
        support = self._build_choice_set("Tobi-Bueck/customer-support-tickets", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nBody: {(r.get('body') or '')[:2000]}", "queue", "queue", "Which support queue should handle this ticket?", 3000, opts_dict=SUPPORT_QUEUES)
        emotion = self._build_choice_set("dair-ai/emotion", "split", "train", "text", "label", "emotion", "Which emotion is most strongly expressed in text?", 3000, opts_dict=EMOTION_CRITERIA)

        foundation_path = self.data_dir / "train.jsonl"
        replay = []
        if foundation_path.exists():
            with open(foundation_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.rng.shuffle(lines)
                for line in lines[:5000]: replay.append(json.loads(line))

        all_samples = td + tools + massive + ag_news + spam + phish + support + emotion + replay
        return split_train_val(all_samples, 0.08, self.rng)

