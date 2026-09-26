from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline import PipelineComposable
from harness.dataset.collator import IBatchCollator, MultiQuestionCollator
from harness.dataset.schema import DecisionSample
from harness.dataset.transforms import load_jsonl

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
ADAPTED_DIR = DATA_DIR / "adapted"
SPLITS_DIR = DATA_DIR / "splits"

@runtime_checkable
class IDataSource(Protocol):
    """Universal strategy contract for dataset extraction."""
    name: str
    def extract(self) -> List[Dict[str, Any]]: ...

@runtime_checkable
class ITrainingDataset(Protocol):
    """Primitive contract for training dataset."""
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> DecisionSample: ...
    def get_loader(
        self,
        batch_size: int = 16,
        max_tokens: int = 4000,
        shuffle: bool = True,
        pin_memory: bool = False,
        seed: int = 42,
        collator: Optional[IBatchCollator] = None,
    ) -> DataLoader: ...

class TrainingDataset(PipelineComposable, Dataset, ITrainingDataset):
    """The single root primitive of the dataset boundary: holds samples and builds DataLoaders."""

    def __init__(
        self,
        samples: Sequence[DecisionSample],
        assembler: Optional[Any] = None,
        collator: Optional[IBatchCollator] = None,
    ):
        self.samples = list(samples)
        self.assembler = assembler
        self.collator = collator or (MultiQuestionCollator(assembler) if assembler is not None else None)
        if self.collator is not None and not isinstance(self.collator, IBatchCollator):
            raise TypeError(f"collator must implement IBatchCollator, got {type(self.collator).__name__}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DecisionSample:
        return self.samples[idx]

    def get_loader(
        self,
        batch_size: int = 16,
        max_tokens: int = 4000,
        shuffle: bool = True,
        pin_memory: bool = False,
        seed: int = 42,
        collator: Optional[IBatchCollator] = None,
    ) -> DataLoader:
        active_collator = collator or self.collator
        if active_collator is None:
            raise ValueError("TrainingDataset requires an assembler or collator to construct DataLoader.")

        assembler = getattr(active_collator, "assembler", self.assembler)
        if assembler is not None and max_tokens > 0:
            lens = [len(assembler.render_sample(s.state, s.questions)[0]) for s in self.samples]
            indices = sorted(range(len(self.samples)), key=lambda i: lens[i])
            batches, cur_b, cur_toks = [], [], 0
            for i in indices:
                if (cur_toks + lens[i] > max_tokens and cur_b) or (batch_size > 0 and len(cur_b) >= batch_size):
                    batches.append(cur_b)
                    cur_b, cur_toks = [], 0
                cur_b.append(i)
                cur_toks += lens[i]
            if cur_b:
                batches.append(cur_b)
        else:
            batches = [list(range(i, min(i + batch_size, len(self.samples)))) for i in range(0, len(self.samples), batch_size)]

        if shuffle:
            random.Random(seed).shuffle(batches)

        return DataLoader(self, batch_sampler=batches, collate_fn=active_collator, pin_memory=pin_memory)

    @classmethod
    def from_jsonl(cls, path: Union[str, Path], assembler: Optional[Any] = None) -> "TrainingDataset":
        if isinstance(assembler, str):
            from nanollm.inference.assembler import SlotAssembler
            assembler = SlotAssembler(assembler)
        return cls(load_jsonl(path), assembler=assembler)

    @classmethod
    def from_source(cls, source: IDataSource, assembler: Optional[Any] = None) -> "TrainingDataset":
        if isinstance(assembler, str):
            from nanollm.inference.assembler import SlotAssembler
            assembler = SlotAssembler(assembler)
        from harness.dataset.transforms import to_decision_sample
        return cls([to_decision_sample(item) for item in source.extract()], assembler=assembler)

    @classmethod
    def loaders(
        cls,
        train_data: Union[str, Path],
        val_data: Optional[Union[str, Path]] = None,
        backbone: Union[str, Any] = "answerdotai/ModernBERT-base",
        batch_size: int = 16,
        max_tokens: int = 4000,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        pin_memory: Optional[bool] = None,
    ) -> Union[DataLoader, Tuple[DataLoader, Optional[DataLoader]]]:
        from nanollm.inference.assembler import SlotAssembler
        from harness.dataset.cached_dataset_layer import CachedDatasetLayer

        assembler = SlotAssembler(backbone) if isinstance(backbone, str) else backbone
        pin = pin_memory if pin_memory is not None else torch.cuda.is_available()

        train_ds: ITrainingDataset = cls.from_jsonl(train_data, assembler=assembler)
        if cache_dir:
            train_ds = train_ds | CachedDatasetLayer(cache_dir=cache_dir)
        train_loader = train_ds.get_loader(
            batch_size=batch_size, max_tokens=max_tokens, shuffle=True, pin_memory=pin, seed=seed
        )

        if not val_data:
            return train_loader

        val_ds: ITrainingDataset = cls.from_jsonl(val_data, assembler=assembler)
        if cache_dir:
            val_ds = val_ds | CachedDatasetLayer(cache_dir=cache_dir)
        val_loader = val_ds.get_loader(
            batch_size=batch_size, max_tokens=max_tokens, shuffle=False, pin_memory=pin, seed=seed
        )
        return train_loader, val_loader

__all__ = ["IDataSource", "ITrainingDataset", "TrainingDataset", "DATA_DIR", "RAW_DIR", "ADAPTED_DIR", "SPLITS_DIR"]
