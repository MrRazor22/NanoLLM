from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union
import torch
from torch.utils.data import DataLoader
from nanollm.inference.assembler_policy import ISlotAssembler, SlotAssembler

class IBatchCollator(Protocol):
    def __call__(self, batch: Sequence[Any]) -> Dict[str, Any]: ...

class MultiQuestionCollator(IBatchCollator):
    def __init__(self, tokenizer_or_assembler: Any):
        if hasattr(tokenizer_or_assembler, "assemble_batch"):
            self.assembler = tokenizer_or_assembler
        else:
            self.assembler = SlotAssembler(tokenizer_or_assembler)
        self._cache: Dict[int, Tuple[List[int], List[Any]]] = {}

    def __call__(self, batch: Sequence[Any]) -> Dict[str, Any]:
        items = []
        for s in batch:
            k = id(s)
            cached = self._cache.get(k)
            if cached is None:
                cached = self._cache[k] = (
                    s if isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], list)
                    else self.assembler.render_sample(s.state, s.questions)
                )
            items.append(cached)
        return self.assembler.assemble_batch(items)

    def pack_loader(
        self,
        dataset: Sequence[Any],
        max_tokens: int = 4000,
        batch_size: int = 16,
        cache_path: Optional[Union[str, Path]] = None,
        shuffle: bool = True,
        pin_memory: bool = False,
    ) -> DataLoader:
        if cache_path and Path(cache_path).exists():
            batches = torch.load(cache_path, weights_only=False)
        else:
            lens = [len(self.assembler.render_sample(s.state, s.questions)[0]) for s in dataset]
            indices = sorted(range(len(dataset)), key=lambda i: lens[i])
            batches, cur_b, cur_toks = [], [], 0
            for i in indices:
                if (max_tokens > 0 and cur_toks + lens[i] > max_tokens and cur_b) or (batch_size > 0 and len(cur_b) >= batch_size):
                    batches.append(cur_b); cur_b, cur_toks = [], 0
                cur_b.append(i); cur_toks += lens[i]
            if cur_b: batches.append(cur_b)
            if cache_path:
                torch.save(batches, str(cache_path))
        if shuffle:
            random.Random(42).shuffle(batches)
        return DataLoader(dataset, batch_sampler=batches, collate_fn=self, pin_memory=pin_memory)
