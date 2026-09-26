from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable
from nanollm.inference.assembler import ISlotAssembler, SlotAssembler

@runtime_checkable
class IBatchCollator(Protocol):
    """Collation policy contract: collates a batch of dataset samples into tensors."""
    def __call__(self, batch: Sequence[Any]) -> Dict[str, Any]: ...

class MultiQuestionCollator(IBatchCollator):
    """Pure collation policy that renders and packs token slots into a batch tensor dict."""
    def __init__(self, tokenizer_or_assembler: Any):
        if isinstance(tokenizer_or_assembler, ISlotAssembler):
            self.assembler = tokenizer_or_assembler
        elif hasattr(tokenizer_or_assembler, "assemble_batch"):
            self.assembler = tokenizer_or_assembler
        else:
            self.assembler = SlotAssembler(tokenizer_or_assembler)
        if not isinstance(self.assembler, ISlotAssembler):
            raise TypeError(f"assembler must implement ISlotAssembler, got {type(self.assembler).__name__}")
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

__all__ = ["IBatchCollator", "MultiQuestionCollator"]
