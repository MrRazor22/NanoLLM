from typing import Optional, Protocol, Sequence, runtime_checkable
import random
import torch

import json
from data.case import DecisionCase
from model.token_layout import Batch, SampleTargets, TargetSlot, TokenLayout, TokenSlot

def format_state(state: str) -> str:
    if state.startswith("{") and state.endswith("}"):
        try:
            if isinstance(parsed := json.loads(state), dict):
                return " | ".join(f"{k}: {v}" for k, v in parsed.items())
        except json.JSONDecodeError:
            pass
    return state

@runtime_checkable
class ITokenizer(Protocol):
    """Sub-Primitive Protocol: Tokenizes text into integer token ids."""
    cls_token_id: int
    sep_token_id: int
    mask_token_id: int
    pad_token_id: int
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

@runtime_checkable
class ILayout(Protocol):
    """Opinion Sub-Primitive: Formats and tokenizes decision states and candidate lists into model layouts."""
    def encode(self, state: str, candidates: Sequence[str]) -> TokenLayout: ...
    def collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> Batch: ...

class MaskedLayout(ILayout):
    """Standard masked token layout policy encoding candidate options directly into marker slots."""

    def __init__(self, tokenizer: Optional[ITokenizer] = None, max_state_tokens: int = 8192):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens

    def encode(self, state: str, candidates: Sequence[str]) -> TokenLayout:
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id

        ids = [cls_id, sep_id]
        pos = []

        for cand in candidates:
            ids.append(mask_id)
            pos.append(len(ids) - 1)
            ids.extend(self.tokenizer.encode(f" {cand}", add_special_tokens=False))
        slots = [TokenSlot(pos)]

        ids.append(sep_id)
        state_tokens = self.tokenizer.encode(format_state(state), add_special_tokens=False)
        if len(state_tokens) > self.max_state_tokens:
            state_tokens = state_tokens[:self.max_state_tokens]
        ids.extend(state_tokens)
        ids.append(sep_id)

        return TokenLayout(input_ids=ids, mask=[1.0] * len(ids), slots=slots)

    def collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> Batch:
        pad_id = self.tokenizer.pad_token_id
        padded_layouts: list[TokenLayout] = []
        all_targets: list[SampleTargets] = []

        for c in cases:
            cands = list(c.candidates)
            target_label = c.target_index()
            if shuffle and len(cands) > 1:
                perm = list(range(len(cands)))
                random.shuffle(perm)
                cands = [cands[p] for p in perm]
                target_label = perm.index(target_label) if target_label < len(perm) else 0

            layout = self.encode(c.state, cands)
            padded_layouts.append(layout)

            pos = layout.slots[0]
            slots = [TargetSlot("candidate", pos, target_label)]
            all_targets.append(SampleTargets(slots))

        max_len = max((len(l.input_ids) for l in padded_layouts), default=0)
        padded_ids = torch.full((len(padded_layouts), max_len), pad_id, dtype=torch.long)
        masks = torch.zeros((len(padded_layouts), max_len), dtype=torch.float)

        for i, layout in enumerate(padded_layouts):
            sz = len(layout.input_ids)
            padded_ids[i, :sz] = torch.as_tensor(layout.input_ids, dtype=torch.long)
            masks[i, :sz] = 1.0

        return Batch(input_ids=padded_ids, mask=masks, targets=all_targets)
