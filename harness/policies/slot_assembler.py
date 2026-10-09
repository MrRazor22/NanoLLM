from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch
from transformers import AutoTokenizer
from nanollm import DecisionSample, DecisionState, Question
from nanollm.engine.policies.slot_encoder import ITokenizer, SlotEncoder
from nanollm.trainer.batches import TargetSlot, TrainingBatch

class SlotAssembler:
    """Renders domain state and question slots into token indices and builds collated batch tensors."""
    def __init__(self, tokenizer_or_name: Union[str, ITokenizer]):
        tok = AutoTokenizer.from_pretrained(tokenizer_or_name) if isinstance(tokenizer_or_name, str) else tokenizer_or_name
        self.encoder = SlotEncoder(tok)
        self.pad_id = getattr(tok, "pad_token_id", 50283)

    def render_sample(self, state: DecisionState, questions: Sequence[Question]) -> Tuple[List[int], List[TargetSlot]]:
        ids, _, targets = self.encoder.encode_with_targets(state, questions)
        return ids, targets

    def assemble_batch(self, items: Sequence[Tuple[List[int], List[TargetSlot]]], device: Optional[torch.device] = None) -> Dict[str, Union[torch.Tensor, Sequence[Sequence[int]], Sequence[Sequence[TargetSlot]]]]:
        dev = device or torch.device("cpu")
        all_ids, batch_meta = [it[0] for it in items], [it[1] for it in items]
        max_len, bsz = max(len(ids) for ids in all_ids), len(all_ids)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((bsz, max_len), dtype=torch.float, device=dev)
        for i, ids in enumerate(all_ids):
            sz = len(ids)
            input_ids[i, :sz] = torch.as_tensor(ids, dtype=torch.long)
            mask[i, :sz] = 1.0
        return {"input_ids": input_ids, "mask": mask, "slots": [[m.indices for m in row] for row in batch_meta], "targets": batch_meta, "meta": batch_meta}

@runtime_checkable
class IBatchCollator(Protocol):
    """Collation policy contract: collates a batch of dataset samples into a structured training batch."""
    def __call__(self, batch: Sequence[DecisionSample]) -> Union[TrainingBatch, Dict[str, object]]: ...

class MultiQuestionCollator(IBatchCollator):
    """Pure collation policy that renders and packs token slots into a batch tensor dict."""
    def __init__(self, assembler: SlotAssembler):
        self.assembler, self._cache = assembler, {}

    def __call__(self, batch: Sequence[DecisionSample]) -> Dict[str, Union[torch.Tensor, Sequence[Sequence[int]], Sequence[Sequence[TargetSlot]]]]:
        items = []
        for s in batch:
            k = id(s)
            cached = self._cache.get(k)
            if cached is None:
                cached = self._cache[k] = self.assembler.render_sample(s.state, s.questions)
            items.append(cached)
        return self.assembler.assemble_batch(items)

__all__ = ["SlotAssembler", "IBatchCollator", "MultiQuestionCollator"]
