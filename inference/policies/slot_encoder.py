from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable
from inference.case import DecisionCase
from inference.questions import Choice, DecisionState, Question, Score, TokenLayout, format_state
from training.batches import SampleTargets, TargetSlot, TrainingBatch

@runtime_checkable
class ITokenizer(Protocol):
    """Protocol for string tokenizers with slot masking support."""
    mask_token_id: int
    sep_token_id: int
    eos_token_id: int
    pad_token_id: int
    def encode(self, text: str) -> Sequence[int]: ...

@runtime_checkable
class ILayoutEncoder(Protocol):
    """Injected Opinion Sub-Primitive: Maps domain state and questions into discrete token slot layouts."""
    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout: ...
    def encode_case(self, case: DecisionCase) -> Tuple[List[int], List[TargetSlot]]: ...
    def encode_batch(self, cases: Sequence[DecisionCase]) -> TrainingBatch: ...

class SlotEncoder(ILayoutEncoder):
    """Encodes state and questions into token layout with masked slots, optionally extracting target annotations."""
    def __init__(self, tokenizer: Optional[ITokenizer] = None, max_state_tokens: int = 384):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens
        self.mask_id = tokenizer.mask_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout:
        ids, slots, _ = self.encode_with_targets(state, questions)
        return TokenLayout(input_ids=ids, mask=[1.0] * len(ids), slots=slots)

    def encode_case(self, case: DecisionCase) -> Tuple[List[int], List[TargetSlot]]:
        ids, _, targets = self.encode_with_targets(case.state, case.questions)
        return ids, targets

    def encode_with_targets(self, state: DecisionState, questions: Sequence[Question]) -> Tuple[List[int], List[List[int]], List[TargetSlot]]:
        state_ids = self.tokenizer.encode(format_state(state))
        ids = state_ids[:self.max_state_tokens] if len(state_ids) > self.max_state_tokens else list(state_ids)
        slots, targets = [], []
        for q in questions:
            ids.append(self.sep_id)
            ids.extend(self.tokenizer.encode(f" {q.name}: {q.instruction}" if q.instruction else f" {q.name}:"))
            if isinstance(q, Choice):
                pos = []
                opt_items = q.options.items() if isinstance(q.options, dict) else [(o, None) for o in q.options]
                for k, v in opt_items:
                    ids.append(self.mask_id)
                    pos.append(len(ids) - 1)
                    ids.extend(self.tokenizer.encode(f" {k}: {v}" if v else f" {k}"))
                slots.append(pos)
                targets.append(TargetSlot("choice", pos, int(q.target) if q.target is not None else 0))
            else:
                ids.append(self.mask_id)
                pos = [len(ids) - 1]
                slots.append(pos)
                q_type = "score" if isinstance(q, Score) else "noul"
                targets.append(TargetSlot(q_type, pos, float(q.target) if q.target is not None else 0.0))
        return ids, slots, targets

    def encode_batch(self, cases: Sequence[DecisionCase]) -> TrainingBatch:
        encoded = [self.encode_case(c) for c in cases]
        all_ids = [e[0] for e in encoded]
        targets = [SampleTargets(e[1]) for e in encoded]
        max_len = max((len(ids) for ids in all_ids), default=0)
        padded_ids = [ids + [self.pad_id] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1.0] * len(ids) + [0.0] * (max_len - len(ids)) for ids in all_ids]
        return TrainingBatch(input_ids=padded_ids, mask=masks, targets=targets)

__all__ = ["ITokenizer", "ILayoutEncoder", "SlotEncoder"]
