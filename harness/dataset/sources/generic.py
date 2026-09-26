import json
from pathlib import Path
from datasets import load_dataset
from harness.dataset.training_dataset import RAW_DIR, IDataSource
from harness.dataset.transforms import load_raw_jsonl, save_jsonl

class GenericChoiceSource(IDataSource):
    name = "generic_choice"
    def __init__(
        self,
        path: str,
        sub: Any,
        split: str,
        formatter: Any,
        label_extractor: Any,
        qname: str,
        instr: str,
        limit: int,
        opts_dict: Any = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        per_class_limit: int = 0,
        blacklist: Optional[Set[str]] = None,
        name: Optional[str] = None,
    ):
        if name:
            self.name = name
        self.path = path
        self.sub = sub
        self.split = split
        self.formatter = formatter
        self.label_extractor = label_extractor
        self.qname = qname
        self.instr = instr
        self.limit = limit
        self.opts_dict = opts_dict
        self.filter_fn = filter_fn
        self.per_class_limit = per_class_limit
        self.blacklist = blacklist or set()

    def extract(self) -> List[Dict[str, Any]]:
        raw_path = RAW_DIR / f"{self.name}.jsonl"
        if raw_path.exists():
            ds, feat = load_raw_jsonl(raw_path), None
        else:
            ds = load_dataset(self.path, self.sub, split=self.split) if self.sub else load_dataset(self.path, split=self.split)
            feat = ds.features.get(self.label_extractor) if isinstance(self.label_extractor, str) else None
            save_jsonl(raw_path, ds)
        if self.opts_dict:
            criteria = self.opts_dict
        elif feat and hasattr(feat, "names"):
            criteria = {name.replace("_", " "): name.replace("_", " ") for name in feat.names if name not in self.blacklist}
        else:
            criteria = None

        count, per_class_counts = 0, {}
        records = []
        for row in ds:
            if self.limit and count >= self.limit:
                break
            if self.filter_fn and not self.filter_fn(row):
                continue
            text = self.formatter(row) if callable(self.formatter) else str(row.get(self.formatter, ""))
            if not text.strip():
                continue

            if callable(self.label_extractor):
                gold_label = self.label_extractor(row)
            else:
                raw_val = row.get(self.label_extractor)
                gold_label = feat.names[raw_val].replace("_", " ") if feat and hasattr(feat, "names") and isinstance(raw_val, int) else str(raw_val)

            if gold_label in self.blacklist:
                continue

            if self.per_class_limit:
                c = per_class_counts.get(gold_label, 0)
                if c >= self.per_class_limit:
                    continue
                per_class_counts[gold_label] = c + 1

            records.append({
                "category": self.path,
                "state": text.strip(),
                "questions": {
                    self.qname: {
                        "type": "choice",
                        "instructions": self.instr,
                        "criteria": criteria or {gold_label: gold_label},
                    }
                },
                "gold": {self.qname: {"type": "choice", "label": gold_label}},
            })
            count += 1

        return records

__all__ = ["GenericChoiceSource"]
