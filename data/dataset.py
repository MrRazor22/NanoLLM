import json
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Union, runtime_checkable

from inference.questions import Choice, DecisionSample, Noul, Score
from data.policies.leakage import ILeakagePolicy, NGramLeakagePolicy
from data.policies.split import BudgetSplitPolicy, DataSplits, ISplitPolicy
from data.sources.source import IDataSource, SourceRecord

DATA_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DATA_DIR / "corpus"

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract: supplies partitioned decision samples to callers."""
    def load(self, split: str = "train") -> Sequence[DecisionSample]: ...

class Dataset(IDataset):
    """Sovereign Core Primitive: manages source registration, split preparation, zero leakage, and cached sample loading."""

    def __init__(
        self,
        split_policy: Optional[ISplitPolicy] = None,
        leakage_policy: Optional[ILeakagePolicy] = None,
        corpus_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        self.split_policy = split_policy or BudgetSplitPolicy(seed=seed)
        self.leakage_policy = leakage_policy or NGramLeakagePolicy(13)
        self.corpus_dir = corpus_dir or CORPUS_DIR
        self.seed = seed
        self._sources: List[IDataSource] = []

    def register(self, source: IDataSource) -> "Dataset":
        """Fluent extension to register data sources."""
        self._sources.append(source)
        return self

    def prepare(self, sources: Optional[Sequence[IDataSource]] = None) -> DataSplits:
        """Extracts all sources, partitions splits, audits for leakage, and caches to disk."""
        src_list = list(sources) if sources is not None else self._sources
        if not src_list:
            raise ValueError("No data sources registered for dataset preparation.")

        records: List[SourceRecord] = []
        for src in src_list:
            records.extend(src.read())

        splits = self.split_policy.partition(records)

        # Invariant verification: zero leakage
        if self.leakage_policy.find_leaks(splits.train, splits.test):
            raise ValueError("Contamination invariant violated: train split leaks test holdout.")
        if self.leakage_policy.find_leaks(splits.train, splits.val):
            raise ValueError("Contamination invariant violated: train split leaks validation.")

        # Persist to data/corpus/ cache
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(self.corpus_dir / "train.jsonl", splits.train)
        save_jsonl(self.corpus_dir / "val.jsonl", splits.val)
        save_jsonl(self.corpus_dir / "test.jsonl", splits.test)

        return splits

    def load(self, split: str = "train") -> Sequence[DecisionSample]:
        """Loads samples directly for callers from corpus cache, automatically preparing if absent."""
        split_file = self.corpus_dir / f"{split}.jsonl"
        if not split_file.exists():
            if self._sources:
                self.prepare()
            else:
                return []
        return load_raw_jsonl(split_file)

def to_decision_sample(raw: dict) -> DecisionSample:
    state = str(raw.get("state", ""))
    qs, gold = raw.get("questions", {}), raw.get("gold", {}) or {}
    specs = []
    if isinstance(qs, dict):
        for name, spec in qs.items():
            opts, ins = spec.get("options") or spec.get("criteria", {}), spec.get("instructions")
            q_type = spec.get("type", "choice" if opts else "noul")
            g = gold.get(name, {})
            lbl = g.get("label") if isinstance(g, dict) else g
            if q_type == "choice" and opts:
                keys = list(opts.keys()) if isinstance(opts, dict) else list(opts)
                specs.append(Choice(name=name, options=opts, instruction=ins, target=keys.index(str(lbl)) if str(lbl) in keys else 0))
            elif q_type == "score":
                specs.append(Score(name=name, instruction=ins, target=float(lbl) if lbl is not None else 0.0))
            else:
                specs.append(Noul(name=name, instruction=ins, target=bool(lbl) if lbl is not None else False))
    return DecisionSample(state=state, questions=specs)

def load_raw_jsonl(path: Union[str, Path]) -> List[DecisionSample]:
    p = Path(path)
    return [to_decision_sample(json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.exists() else []

def save_jsonl(path: Union[str, Path], samples: Sequence[DecisionSample]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "state": s.state,
            "questions": {q.name: {"type": "choice" if isinstance(q, Choice) else ("score" if isinstance(q, Score) else "noul"), "options": getattr(q, "options", None), "instructions": q.instruction} for q in s.questions},
            "gold": {q.name: q.target for q in s.questions},
        }, default=str)
        for s in samples
    ]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

__all__ = ["DATA_DIR", "CORPUS_DIR", "IDataset", "Dataset", "DataSplits", "to_decision_sample", "load_raw_jsonl", "save_jsonl"]
