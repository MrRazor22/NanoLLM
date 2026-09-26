from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union
import torch
from torch.utils.data import DataLoader, Dataset

from training.dataset.collator_policy import IBatchCollator, MultiQuestionCollator
from training.dataset.schema import DecisionSample
from training.dataset.transforms import load_jsonl

class IDataSourcePolicy(Protocol):
    """Policy contract for raw data source extraction."""
    def extract(self) -> List[Dict[str, Any]]: ...

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

class TrainingDataset(Dataset, ITrainingDataset):
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
        cache_path: Optional[Union[str, Path]] = None,
    ) -> DataLoader:
        active_collator = collator or self.collator
        if active_collator is None:
            raise ValueError("TrainingDataset requires an assembler or collator to construct DataLoader.")

        if cache_path and Path(cache_path).exists():
            batches = torch.load(cache_path, weights_only=False)
        else:
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

            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(batches, str(cache_path))

        if shuffle:
            random.Random(seed).shuffle(batches)

        return DataLoader(self, batch_sampler=batches, collate_fn=active_collator, pin_memory=pin_memory)

    @classmethod
    def from_jsonl(cls, path: Union[str, Path], assembler: Optional[Any] = None) -> "TrainingDataset":
        return cls(load_jsonl(path), assembler=assembler)

__all__ = ["IDataSourcePolicy", "ITrainingDataset", "TrainingDataset"]
