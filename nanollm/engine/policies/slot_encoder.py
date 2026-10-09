from typing import List, Protocol, Sequence, runtime_checkable
from nanollm.engine.questions import DecisionState, Question, TokenLayout, format_state

@runtime_checkable
class ITokenizer(Protocol):
    """Protocol for string tokenizers with slot masking support."""

    mask_token_id: int
    sep_token_id: int
    eos_token_id: int

    def encode(self, text: str) -> Sequence[int]:
        ...

@runtime_checkable
class ILayoutEncoder(Protocol):
    """Injected Opinion Sub-Primitive: Maps domain state and questions into discrete token slot layouts."""

    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout:
        ...

class SlotEncoder(ILayoutEncoder):
    """Encodes state and questions into token layout with masked slots."""

    def __init__(self, tokenizer: ITokenizer, max_state_tokens: int = 384):
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens
        self.mask_id = getattr(tokenizer, "mask_token_id", 50281)
        self.sep_id = getattr(tokenizer, "sep_token_id", getattr(tokenizer, "eos_token_id", 50282))

    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout:
        text = format_state(state)
        state_ids = self.tokenizer.encode(text)
        ids = state_ids[:self.max_state_tokens] if len(state_ids) > self.max_state_tokens else list(state_ids)
        slots: List[List[int]] = []

        for q in questions:
            ids.append(self.sep_id)
            ins = getattr(q, "instruction", None)
            header = f" {q.name}: {ins}" if ins else f" {q.name}:"
            ids.extend(self.tokenizer.encode(header))
            opts = getattr(q, "options", None)
            if opts:
                pos = []
                opt_items = opts.items() if isinstance(opts, dict) else [(o, None) for o in opts]
                for k, v in opt_items:
                    ids.append(self.mask_id)
                    pos.append(len(ids) - 1)
                    ids.extend(self.tokenizer.encode(f" {k}: {v}" if v else f" {k}"))
                slots.append(pos)
            else:
                ids.append(self.mask_id)
                slots.append([len(ids) - 1])

        return TokenLayout(
            input_ids=[ids],
            mask=[[1.0] * len(ids)],
            slot_indices=slots,
        )

__all__ = ["ITokenizer", "ILayoutEncoder", "SlotEncoder"]
