from typing import List, Optional, Protocol, Sequence, runtime_checkable
from inference.case import DecisionCase
from inference.policies.slot_encoder import ILayoutEncoder
from training.batches import SampleTargets, TrainingBatch

@runtime_checkable
class IBatchCollator(Protocol):
    """Injected Opinion Sub-Primitive: Collates variable-length decision cases into padded training batches."""
    def collate(self, cases: Sequence[DecisionCase], encoder: ILayoutEncoder) -> TrainingBatch: ...

class PaddedBatchCollator(IBatchCollator):
    """Pad-to-longest batch collator policy."""

    def __init__(self, pad_token_id: Optional[int] = None):
        self.pad_token_id = pad_token_id

    def collate(self, cases: Sequence[DecisionCase], encoder: ILayoutEncoder) -> TrainingBatch:
        pad_id = self.pad_token_id if self.pad_token_id is not None else getattr(encoder, "pad_token_id", 0)
        layouts = [encoder.encode(c.state, c.questions) for c in cases]
        all_ids = [list(l.input_ids) for l in layouts]
        targets = [SampleTargets(l.targets) for l in layouts]
        max_len = max((len(ids) for ids in all_ids), default=0)
        padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1.0] * len(ids) + [0.0] * (max_len - len(ids)) for ids in all_ids]
        return TrainingBatch(input_ids=padded_ids, mask=masks, targets=targets)

__all__ = ["IBatchCollator", "PaddedBatchCollator"]
