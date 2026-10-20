from typing import Optional, Protocol, Sequence, runtime_checkable
import random
import torch

import json
from data.case import DecisionCase
from model.token_layout import Batch

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
    """Opinion Sub-Primitive: Formats and tokenizes decision states and candidate lists into model batches."""
    def encode(self, state: str, candidates: Sequence[str]) -> Batch: ...
    def collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> Batch: ...

class MaskedLayout(ILayout):
    """Standard masked token layout policy encoding candidate options directly into marker slots."""

    def __init__(self, tokenizer: Optional[ITokenizer] = None, max_state_tokens: int = 8192):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens

    def encode(self, state: str, candidates: Sequence[str]) -> Batch:
        case = DecisionCase(state=state, candidates=candidates, probabilities={}, category="", source="")
        return self.collate([case], shuffle=False)

    def collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> Batch:
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id

        all_ids: list[list[int]] = []
        all_candidate_indices: list[list[int]] = []
        all_targets: list[int] = []
        has_targets = False

        for c in cases:
            cands = list(c.candidates)
            if c.probabilities:
                has_targets = True
                target_label = c.target_index()
                if shuffle and len(cands) > 1:
                    perm = list(range(len(cands)))
                    random.shuffle(perm)
                    cands = [cands[p] for p in perm]
                    target_label = perm.index(target_label) if target_label < len(perm) else 0
                all_targets.append(target_label)

            ids = [cls_id, sep_id]
            pos = []
            for cand in cands:
                ids.append(mask_id)
                pos.append(len(ids) - 1)
                ids.extend(self.tokenizer.encode(f" {cand}", add_special_tokens=False))

            ids.append(sep_id)
            state_tokens = self.tokenizer.encode(format_state(c.state), add_special_tokens=False)
            if len(state_tokens) > self.max_state_tokens:
                state_tokens = state_tokens[:self.max_state_tokens]
            ids.extend(state_tokens)
            ids.append(sep_id)

            all_ids.append(ids)
            all_candidate_indices.append(pos)

        max_len = max((len(ids) for ids in all_ids), default=0)
        padded_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
        masks = torch.zeros((len(all_ids), max_len), dtype=torch.float)

        for i, ids in enumerate(all_ids):
            sz = len(ids)
            padded_ids[i, :sz] = torch.as_tensor(ids, dtype=torch.long)
            masks[i, :sz] = 1.0

        targets = torch.as_tensor(all_targets, dtype=torch.long) if has_targets else None
        return Batch(input_ids=padded_ids, mask=masks, candidate_indices=all_candidate_indices, targets=targets)
