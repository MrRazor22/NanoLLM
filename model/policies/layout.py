from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable
from transformers import PreTrainedTokenizerBase

from data.case import DecisionCase

@dataclass(frozen=True, slots=True)
class Layout:
    tokens: Sequence[int]
    slots: Sequence[int]
    pad_id: int = 0

@runtime_checkable
class ILayout(Protocol):
    def layout(self, cases: Sequence[DecisionCase]) -> Sequence[Layout]: ...

class MaskedLayout(ILayout):
    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None, max_state_tokens: int = 8192):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id.")
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens

    def layout_case(self, case: DecisionCase) -> Layout:
        cls_id = self.tokenizer.cls_token_id   # [CLS] ID
        sep_id = self.tokenizer.sep_token_id   # [SEP] ID
        mask_id = self.tokenizer.mask_token_id # [MASK] ID

        ids = [cls_id, sep_id]
        slots = []
        for cand in case.candidates:
            ids.append(mask_id)
            slots.append(len(ids) - 1)
            ids.extend(self.tokenizer.encode(f" {cand}", add_special_tokens=False))

        ids.append(sep_id)
        state_tokens = self.tokenizer.encode(case.state, add_special_tokens=False)
        if len(state_tokens) > self.max_state_tokens:
            state_tokens = state_tokens[:self.max_state_tokens]
        ids.extend(state_tokens)
        ids.append(sep_id)

        return Layout(tokens=ids, slots=slots, pad_id=self.tokenizer.pad_token_id)

    def layout(self, cases: Sequence[DecisionCase]) -> Sequence[Layout]:
        return [self.layout_case(c) for c in cases]

    #Start[CLS] [SEP]​​  Candidate anchors and text[MASK] cand1​  [MASK] cand2​​​  Delimiter[SEP]​​  Decision contextstate tokens​​  End[SEP]​​
