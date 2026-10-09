from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch
from torch.utils.data import DataLoader, Dataset
from harness.dataset.slot_assembler import SlotAssembler
from nanollm import Choice, DecisionSample, Noul, Question, Score

DATA_DIR = Path(__file__).resolve().parent / "data"

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()

def to_decision_sample(item: Dict[str, Any]) -> DecisionSample:
    state = item.get("state", "")
    qs = item.get("questions", {})
    gold = item.get("gold", {})
    specs = []
    if isinstance(qs, dict):
        for name, spec in qs.items():
            opts = spec.get("options") or spec.get("criteria", {})
            ins = spec.get("instructions")
            q_type = spec.get("type", "choice" if opts else "noul")
            g = gold.get(name, {})
            lbl = g.get("label") if isinstance(g, dict) else g
            if q_type == "choice" and opts:
                opt_keys = list(opts.keys()) if isinstance(opts, dict) else list(opts)
                target = opt_keys.index(str(lbl)) if str(lbl) in opt_keys else 0
                specs.append(Choice(name=name, options=opts, instruction=ins, target=target))
            elif q_type == "score":
                specs.append(Score(name=name, instruction=ins, target=float(lbl) if lbl is not None else 0.0))
            else:
                specs.append(Noul(name=name, instruction=ins, target=bool(lbl) if lbl is not None else False))
    return DecisionSample(state=state, questions=specs)

class TrainingDataset(Dataset):
    """The single root primitive of the dataset boundary: holds samples and builds DataLoaders."""

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
        if not p.exists(): return cls([], assembler=assembler)
        import json
        with open(p, "r", encoding="utf-8") as f:
            samples = [to_decision_sample(json.loads(line)) for line in f if line.strip()]
        return cls(samples, assembler=assembler)

    def get_loader(self, batch_size: int = 16, shuffle: bool = True, max_tokens: int = 3600) -> DataLoader:
        if self.assembler is None:
            raise ValueError("TrainingDataset requires an assembler to construct DataLoader.")
        collator = lambda batch: self.assembler.assemble_batch([self.assembler.render_sample(s.state, s.questions) for s in batch])
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

    @classmethod
    def loaders(cls, train_data: str, val_data: Optional[str] = None, backbone: str = "answerdotai/ModernBERT-base", batch_size: int = 16, max_tokens: int = 3600, seed: int = 42) -> Tuple[DataLoader, Optional[DataLoader]]:
        assembler = SlotAssembler(backbone)
        train_loader = cls.from_jsonl(train_data, assembler=assembler).get_loader(batch_size=batch_size, shuffle=True, max_tokens=max_tokens)
        val_loader = cls.from_jsonl(val_data, assembler=assembler).get_loader(batch_size=batch_size, shuffle=False, max_tokens=max_tokens) if val_data else None
        return train_loader, val_loader

    @staticmethod
    def split_records(records: Sequence[Dict[str, Any]], val_ratio: float = 0.08, rng: Optional[random.Random] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        gen = rng or random.Random(42)
        n_val = int(len(records) * val_ratio)
        indices = list(range(len(records)))
        gen.shuffle(indices)
        val_set = set(indices[:n_val])
        return [r for i, r in enumerate(records) if i not in val_set], [r for i, r in enumerate(records) if i in val_set]

__all__ = ["TrainingDataset", "normalize_text", "to_decision_sample", "DATA_DIR"]
