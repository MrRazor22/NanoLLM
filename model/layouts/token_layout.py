from dataclasses import dataclass
from pathlib import Path
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union
import torch

if TYPE_CHECKING:
    from data.case import DecisionCase

from inference.questions import Choice, DecisionState, Question, Score, format_state

@dataclass(frozen=True)
class TokenLayout:
    """Explicit domain DTO for tokenized input sequence and question slot positions."""
    input_ids: Sequence[int]
    mask: Sequence[float]
    slots: Sequence[Sequence[int]]

@dataclass(frozen=True)
class PredictionLayout:
    """Explicit domain DTO for model output scores and corresponding question slot positions."""
    scores: Sequence[float]
    slots: Sequence[Sequence[int]]

@dataclass(frozen=True)
class TargetSlot:
    """Explicit domain DTO for a question's masked slot positions and target value."""
    target_type: str
    indices: Sequence[int]
    value: Union[int, float]

@dataclass(frozen=True)
class SampleTargets:
    """Explicit domain DTO for all target slots in a single sample."""
    slots: Sequence[TargetSlot]

@dataclass(frozen=True)
class Batch:
    """Explicit domain DTO for a batch of encoded tokens and targets."""
    input_ids: torch.Tensor
    mask: torch.Tensor
    targets: Sequence[SampleTargets]

def encode_layout(
    tokenizer: object,
    state: DecisionState,
    questions: Sequence[Question],
    max_state_tokens: int = 384,
    tok_cache: Optional[Dict[str, List[int]]] = None,
) -> TokenLayout:
    """Colocated Transform: encodes decision state and questions into TokenLayout."""
    cls_id = getattr(tokenizer, "cls_token_id", 50281)
    sep_id = getattr(tokenizer, "sep_token_id", 50282)
    mask_id = getattr(tokenizer, "mask_token_id", 50284)

    def _encode(text: str) -> List[int]:
        if tok_cache is not None and text in tok_cache:
            return tok_cache[text]
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if tok_cache is not None:
            tok_cache[text] = encoded
        return encoded

    state_tokens = _encode(format_state(state))
    if len(state_tokens) > max_state_tokens:
        state_tokens = state_tokens[:max_state_tokens]

    ids = [cls_id] + list(state_tokens)
    slots = []
    for q in questions:
        ids.append(sep_id)
        q_text = f" {q.name}: {q.instruction}" if q.instruction else f" {q.name}:"
        ids.extend(_encode(q_text))
        if isinstance(q, Choice):
            pos = []
            opt_items = q.options.items() if isinstance(q.options, dict) else [(o, None) for o in q.options]
            for k, v in opt_items:
                ids.append(mask_id)
                pos.append(len(ids) - 1)
                opt_text = f" {k}: {v}" if v else f" {k}"
                ids.extend(_encode(opt_text))
            slots.append(pos)
        else:
            ids.append(mask_id)
            slots.append([len(ids) - 1])
    ids.append(sep_id)
    return TokenLayout(input_ids=ids, mask=[1.0] * len(ids), slots=slots)

def collate_batch(
    tokenizer: object,
    cases: Sequence["DecisionCase"],
    max_state_tokens: int = 384,
    shuffle_options: bool = False,
    tok_cache: Optional[Dict[str, List[int]]] = None,
) -> Batch:
    """Colocated Transform: batches domain DecisionCase instances into padded Batch."""
    all_ids, targets = [], []
    for c in cases:
        q_list = []
        for q in c.questions:
            if shuffle_options and isinstance(q, Choice):
                keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                raw_idx = q.target if isinstance(q.target, int) else (keys.index(q.target) if q.target in keys else (int(q.target) if str(q.target).isdigit() else 0))
                indices = list(range(len(keys)))
                random.shuffle(indices)
                new_idx = indices.index(raw_idx) if 0 <= raw_idx < len(indices) else 0
                shuffled_opts = {keys[i]: q.options[keys[i]] for i in indices} if isinstance(q.options, dict) else [keys[i] for i in indices]
                q_list.append(Choice(name=q.name, options=shuffled_opts, instruction=q.instruction, target=new_idx))
            else:
                q_list.append(q)

        layout = encode_layout(tokenizer, c.state, q_list, max_state_tokens=max_state_tokens, tok_cache=tok_cache)
        sample_targets = []
        for q, pos in zip(q_list, layout.slots):
            if isinstance(q, Choice):
                keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                idx = q.target if isinstance(q.target, int) else (keys.index(q.target) if q.target in keys else (int(q.target) if str(q.target).isdigit() else 0))
                sample_targets.append(TargetSlot("choice", pos, idx))
            else:
                sample_targets.append(TargetSlot("score" if isinstance(q, Score) else "noul", pos, float(q.target) if q.target is not None else 0.0))
        all_ids.append(list(layout.input_ids))
        targets.append(SampleTargets(sample_targets))

    max_len = max((len(ids) for ids in all_ids), default=0)
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    padded_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    masks = torch.zeros((len(all_ids), max_len), dtype=torch.float)
    for i, ids in enumerate(all_ids):
        sz = len(ids)
        padded_ids[i, :sz] = torch.as_tensor(ids, dtype=torch.long)
        masks[i, :sz] = 1.0
    return Batch(input_ids=padded_ids, mask=masks, targets=targets)

__all__ = [
    "TokenLayout",
    "PredictionLayout",
    "TargetSlot",
    "SampleTargets",
    "Batch",
    "encode_layout",
    "collate_batch",
]
