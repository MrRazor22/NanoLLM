from typing import Optional, Protocol, Sequence, runtime_checkable
import torch

from data.case import (
    ChoiceOption, DecisionCase, IOption, NoulOption, ScoreOption,
    choice_target_index, format_state, score_to_norm, shuffle_choice,
)
from model.token_layout import Batch, SampleTargets, TargetSlot, TokenLayout, TokenSlot

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
    """Opinion Sub-Primitive: Formats and tokenizes decision states and cases into model layouts."""
    def encode(self, state: str, option: IOption) -> TokenLayout: ...
    def collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> Batch: ...

class MaskedLayout(ILayout):
    """Standard masked token layout policy encoding decision option and candidate slots."""

    def __init__(self, tokenizer: Optional[ITokenizer] = None, max_state_tokens: int = 8192):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens

    def encode(self, state: str, option: IOption) -> TokenLayout:
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id

        ids = [cls_id, sep_id]
        slots = []

        q_type = "choice" if isinstance(option, ChoiceOption) else ("score" if isinstance(option, ScoreOption) else "noul")
        prompt = f" {q_type} decision:"
        ids.extend(self.tokenizer.encode(prompt, add_special_tokens=False))

        if isinstance(option, ChoiceOption):
            pos = []
            opt_items = option.options.items() if isinstance(option.options, dict) else [(o, None) for o in option.options]
            for k, v in opt_items:
                ids.append(mask_id)
                pos.append(len(ids) - 1)
                opt_text = f" {k}: {v}" if v else f" {k}"
                ids.extend(self.tokenizer.encode(opt_text, add_special_tokens=False))
            slots.append(TokenSlot(pos))
        else:
            ids.append(mask_id)
            slots.append(TokenSlot([len(ids) - 1]))

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
            opt = shuffle_choice(c.option) if shuffle and isinstance(c.option, ChoiceOption) else c.option
            layout = self.encode(c.state, opt)
            padded_layouts.append(layout)

            slots = []
            pos = layout.slots[0]
            if isinstance(opt, ChoiceOption):
                slots.append(TargetSlot("choice", pos, choice_target_index(opt)))
            elif isinstance(opt, ScoreOption):
                target_val = score_to_norm(opt, opt.answer.score) if opt.answer is not None else 0.0
                slots.append(TargetSlot("score", pos, target_val))
            elif isinstance(opt, NoulOption):
                is_pos = opt.answer.decision if opt.answer is not None else False
                slots.append(TargetSlot("noul", pos, 1.0 if is_pos else 0.0))
            all_targets.append(SampleTargets(slots))

        max_len = max((len(l.input_ids) for l in padded_layouts), default=0)
        padded_ids = torch.full((len(padded_layouts), max_len), pad_id, dtype=torch.long)
        masks = torch.zeros((len(padded_layouts), max_len), dtype=torch.float)

        for i, layout in enumerate(padded_layouts):
            sz = len(layout.input_ids)
            padded_ids[i, :sz] = torch.as_tensor(layout.input_ids, dtype=torch.long)
            masks[i, :sz] = 1.0

        return Batch(input_ids=padded_ids, mask=masks, targets=all_targets)
