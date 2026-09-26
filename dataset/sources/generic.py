from typing import Any, Callable, Dict, List, Optional, Protocol, Set
from datasets import load_dataset

class ISourceAdapter(Protocol):
    """The bedrock contract for an injected data source policy."""
    def extract(self) -> List[Dict[str, Any]]: ...

class GenericChoiceSource(ISourceAdapter):
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
    ):
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
        ds = load_dataset(self.path, self.sub, split=self.split) if self.sub else load_dataset(self.path, split=self.split)
        feat = ds.features.get(self.label_extractor) if isinstance(self.label_extractor, str) else None
        if self.opts_dict:
            labels = self.opts_dict
        elif hasattr(feat, "names") and feat.names:
            labels = list(feat.names)
        elif isinstance(self.label_extractor, str):
            labels = sorted(list(set(ds[self.label_extractor])))
        else:
            labels = sorted(list(set(self.label_extractor(row) for row in ds.select(range(min(len(ds), 500))))))
        opt_keys = list(labels.keys()) if isinstance(labels, dict) else labels
        records, counts = [], {}
        for row in ds:
            if self.filter_fn and not self.filter_fn(row):
                continue
            text = self.formatter(row).strip() if callable(self.formatter) else str(row.get(self.formatter, "")).strip()
            if not text or (self.blacklist and text.lower() in self.blacklist):
                continue
            raw = self.label_extractor(row) if callable(self.label_extractor) else row.get(self.label_extractor)
            lbl_str = opt_keys[raw] if isinstance(raw, int) and isinstance(labels, dict) else (labels[raw] if isinstance(raw, int) else str(raw))
            if lbl_str not in opt_keys:
                continue
            if self.per_class_limit > 0:
                if counts.get(lbl_str, 0) >= self.per_class_limit:
                    continue
                counts[lbl_str] = counts.get(lbl_str, 0) + 1
            records.append({"state": text, "questions": [[self.qname, "choice", opt_keys.index(lbl_str), labels, self.instr]]})
            if self.limit > 0 and len(records) >= self.limit:
                break
        return records
