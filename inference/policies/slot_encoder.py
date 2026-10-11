from typing import Optional, Protocol, Sequence, runtime_checkable
from inference.questions import Choice, DecisionState, Question, format_state
from model import TokenLayout

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
    """Injected Opinion Sub-Primitive: Maps domain state and questions into token slot layouts."""
    pad_token_id: int
    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout: ...

class SlotEncoder(ILayoutEncoder):
    """Encodes state and questions into token layout with masked slots for inference."""

    def __init__(self, tokenizer: Optional[ITokenizer] = None, max_state_tokens: int = 384):
        self._tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens

    @property
    def tokenizer(self) -> ITokenizer:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        return self._tokenizer

    @property
    def mask_id(self) -> int:
        return self.tokenizer.mask_token_id

    @property
    def sep_id(self) -> int:
        return self.tokenizer.sep_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout:
        state_ids = self.tokenizer.encode(format_state(state))
        ids = state_ids[:self.max_state_tokens] if len(state_ids) > self.max_state_tokens else list(state_ids)
        slots = []
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
            else:
                ids.append(self.mask_id)
                slots.append([len(ids) - 1])
        return TokenLayout(input_ids=ids, mask=[1.0] * len(ids), slots=slots)

__all__ = ["ITokenizer", "ILayoutEncoder", "SlotEncoder"]
