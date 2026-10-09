from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterator, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
from torch.utils.data import DataLoader, Dataset
from nanollm import Choice, DecisionSample, Noul, Score
from harness.policies.slot_assembler import SlotAssembler

DATA_DIR = Path(__file__).resolve().parent / "data"

@runtime_checkable
class IDataset(Protocol):
    """Sovereign primitive contract for decision dataset collections."""
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> DecisionSample: ...
    def get_loader(self, batch_size: int = 16, shuffle: bool = True) -> DataLoader: ...

class TrainingDataset(Dataset, IDataset):
    """Sovereign core primitive: encapsulates decision samples and native batch collation."""

    def __init__(self, samples: Sequence[DecisionSample], assembler: Optional[SlotAssembler] = None):
        self.samples = list(samples)
        self.assembler = assembler

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DecisionSample:
        return self.samples[idx]

    @classmethod
    def from_jsonl(cls, path: Union[str, Path], assembler: Optional[SlotAssembler] = None) -> "TrainingDataset":
        p = Path(path)
        return cls(load_raw_jsonl(p) if p.exists() else [], assembler=assembler)

    def get_loader(self, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
        if self.assembler is None:
            raise ValueError("TrainingDataset requires an assembler policy to construct DataLoader.")
        collator = lambda batch: self.assembler.assemble_batch([self.assembler.render_sample(s.state, s.questions) for s in batch])
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

    @classmethod
    def loaders(
        cls, train_data: str, val_data: Optional[str] = None, backbone: str = "answerdotai/ModernBERT-base",
        batch_size: int = 16, seed: int = 42,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        assembler = SlotAssembler(backbone)
        t_loader = cls.from_jsonl(train_data, assembler=assembler).get_loader(batch_size=batch_size, shuffle=True)
        v_loader = cls.from_jsonl(val_data, assembler=assembler).get_loader(batch_size=batch_size, shuffle=False) if val_data else None
        return t_loader, v_loader

def to_decision_sample(raw: Mapping[str, object]) -> DecisionSample:
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

def inject_abstention(samples: List[DecisionSample], rate: float = 0.15, rng: Optional[random.Random] = None) -> List[DecisionSample]:
    gen = rng or random.Random(42)
    out: List[DecisionSample] = []
    for s in samples:
        if gen.random() >= rate:
            out.append(s)
            continue
        new_qs = []
        for q in s.questions:
            if isinstance(q, Choice):
                opts = dict(q.options) if isinstance(q.options, dict) else {o: None for o in q.options}
                if len(opts) >= 2:
                    opts["__none__"] = "none of the above"
                    new_qs.append(Choice(name=q.name, options=opts, instruction=q.instruction, target="__none__"))
                else: new_qs.append(q)
            else: new_qs.append(q)
        out.append(DecisionSample(state=s.state, questions=new_qs))
    return out

def split_train_val(samples: Sequence[DecisionSample], val_ratio: float = 0.08, rng: Optional[random.Random] = None) -> Tuple[List[DecisionSample], List[DecisionSample]]:
    gen = rng or random.Random(42)
    indices = list(range(len(samples)))
    gen.shuffle(indices)
    n_val = int(len(samples) * val_ratio)
    val_set = set(indices[:n_val])
    return [samples[i] for i in range(len(samples)) if i not in val_set], [samples[i] for i in range(len(samples)) if i in val_set]

__all__ = ["DATA_DIR", "IDataset", "TrainingDataset", "to_decision_sample", "load_raw_jsonl", "save_jsonl", "inject_abstention", "split_train_val"]
