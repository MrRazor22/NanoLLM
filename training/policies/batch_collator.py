from typing import Optional, Protocol, Sequence, runtime_checkable
from inference import DecisionCase, ILayoutEncoder, SlotEncoder
from inference.questions import Choice, Score
from training.batches import Batch, SampleTargets, TargetSlot

@runtime_checkable
class IBatchCollator(Protocol):
    """Injected Opinion Sub-Primitive: Collates decision cases into a padded training batch."""
    def collate(self, cases: Sequence[DecisionCase]) -> Batch: ...

class PaddedBatchCollator(IBatchCollator):
    """Encodes decision cases using slot encoder and pads tokens to the longest batch sequence."""

    def __init__(self, encoder: Optional[ILayoutEncoder] = None):
        self.encoder = encoder or SlotEncoder()
        self.pad_id = getattr(self.encoder, "pad_token_id", 0)

    def collate(self, cases: Sequence[DecisionCase]) -> Batch:
        all_ids, targets = [], []
        for c in cases:
            layout = self.encoder.encode(c.state, c.questions)
            sample_targets = []
            for q, pos in zip(c.questions, layout.slots):
                if isinstance(q, Choice):
                    keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                    if isinstance(q.target, int):
                        idx = q.target
                    elif q.target in keys:
                        idx = keys.index(q.target)
                    elif str(q.target).isdigit():
                        idx = int(q.target)
                    else:
                        idx = 0
                    sample_targets.append(TargetSlot("choice", pos, idx))
                else:
                    q_type = "score" if isinstance(q, Score) else "noul"
                    sample_targets.append(TargetSlot(q_type, pos, float(q.target) if q.target is not None else 0.0))
            all_ids.append(list(layout.input_ids))
            targets.append(SampleTargets(sample_targets))

        max_len = max((len(ids) for ids in all_ids), default=0)
        padded_ids = [ids + [self.pad_id] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1.0] * len(ids) + [0.0] * (max_len - len(ids)) for ids in all_ids]
        return Batch(input_ids=padded_ids, mask=masks, targets=targets)

__all__ = ["IBatchCollator", "PaddedBatchCollator"]
