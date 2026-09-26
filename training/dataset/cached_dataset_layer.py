from pathlib import Path
from typing import Any, Optional, Union
import torch
from torch.utils.data import DataLoader

from training.dataset.collator_policy import IBatchCollator
from training.dataset.dataset import ITrainingDataset
from training.dataset.schema import DecisionSample

class CachedDatasetLayer(ITrainingDataset):
    """ATA Composable Layer: transparently decorates ITrainingDataset to cache batch groupings to disk."""

    def __init__(
        self,
        inner: Optional[ITrainingDataset] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.inner = inner
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def attach(self, inner: ITrainingDataset) -> "CachedDatasetLayer":
        self.inner = inner
        return self

    def add(self, layer: Any, **kwargs: Any) -> "ITrainingDataset":
        if isinstance(layer, type):
            return layer(self, **kwargs)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> "ITrainingDataset":
        return self.add(layer)

    def __len__(self) -> int:
        if self.inner is None:
            raise RuntimeError("CachedDatasetLayer is not attached to an inner dataset.")
        return len(self.inner)

    def __getitem__(self, idx: int) -> DecisionSample:
        if self.inner is None:
            raise RuntimeError("CachedDatasetLayer is not attached to an inner dataset.")
        return self.inner[idx]

    def get_loader(
        self,
        batch_size: int = 16,
        max_tokens: int = 4000,
        shuffle: bool = True,
        pin_memory: bool = False,
        seed: int = 42,
        collator: Optional[IBatchCollator] = None,
    ) -> DataLoader:
        if self.inner is None:
            raise RuntimeError("CachedDatasetLayer is not attached to an inner dataset.")

        if not self.cache_dir:
            return self.inner.get_loader(
                batch_size=batch_size,
                max_tokens=max_tokens,
                shuffle=shuffle,
                pin_memory=pin_memory,
                seed=seed,
                collator=collator,
            )

        active_collator = collator or getattr(self.inner, "collator", None)
        assembler = getattr(active_collator, "assembler", getattr(self.inner, "assembler", None))
        backbone_id = getattr(assembler, "backbone", "") if assembler is not None else ""
        fingerprint = f"{len(self.inner)}:{batch_size}:{max_tokens}:{backbone_id}"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"batches_{abs(hash(fingerprint))}.pt"

        if cache_file.exists():
            try:
                cached = torch.load(cache_file, weights_only=False)
                if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
                    batches = cached.get("batches")
                    if shuffle:
                        import random
                        random.Random(seed).shuffle(batches)
                    return DataLoader(
                        self.inner,
                        batch_sampler=batches,
                        collate_fn=active_collator,
                        pin_memory=pin_memory,
                    )
            except Exception:
                pass

        loader = self.inner.get_loader(
            batch_size=batch_size,
            max_tokens=max_tokens,
            shuffle=False,
            pin_memory=pin_memory,
            seed=seed,
            collator=collator,
        )
        try:
            torch.save({"fingerprint": fingerprint, "batches": list(loader.batch_sampler)}, str(cache_file))
        except Exception:
            pass

        if shuffle:
            import random
            batches = list(loader.batch_sampler)
            random.Random(seed).shuffle(batches)
            return DataLoader(
                self.inner,
                batch_sampler=batches,
                collate_fn=active_collator,
                pin_memory=pin_memory,
            )

        return loader

__all__ = ["CachedDatasetLayer"]
