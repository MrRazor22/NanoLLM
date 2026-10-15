from typing import Optional, Protocol, Sequence, runtime_checkable
import torch

from data.case import DecisionCase
from inference.questions import Choice, DecisionState, Question, Score, format_state
from model.token_layout import Batch, SampleTargets, TargetSlot, TokenLayout

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
    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout: ...
    def collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> Batch: ...

class MaskedLayout(ILayout):
    """Standard masked token layout policy encoding questions and candidate slots."""

    def __init__(self, tokenizer: Optional[ITokenizer] = None, max_state_tokens: int = 384):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens

    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout:
        cls_id = getattr(self.tokenizer, "cls_token_id", 50281)
        sep_id = getattr(self.tokenizer, "sep_token_id", 50282)
        mask_id = getattr(self.tokenizer, "mask_token_id", 50284)

        ids = [cls_id]
        slots = []
        for q in questions:
            ids.append(sep_id)
            q_type = "choice" if isinstance(q, Choice) else ("score" if isinstance(q, Score) else "noul")
            prompt = f" {q_type} question: {q.instruction}" if q.instruction else f" {q_type} question for {q.name}:"
            ids.extend(self.tokenizer.encode(prompt, add_special_tokens=False))

            if isinstance(q, Choice):
                pos = []
                opt_items = q.options.items() if isinstance(q.options, dict) else [(o, None) for o in q.options]
                for k, v in opt_items:
                    ids.append(mask_id)
                    pos.append(len(ids) - 1)
                    opt_text = f" {k}: {v}" if v else f" {k}"
                    ids.extend(self.tokenizer.encode(opt_text, add_special_tokens=False))
                slots.append(pos)
            else:
                ids.append(mask_id)
                slots.append([len(ids) - 1])

        ids.append(sep_id)
        state_tokens = self.tokenizer.encode(format_state(state), add_special_tokens=False)
        if len(state_tokens) > self.max_state_tokens:
            state_tokens = state_tokens[:self.max_state_tokens]
        ids.extend(state_tokens)
        ids.append(sep_id)

        return TokenLayout(input_ids=ids, mask=[1.0] * len(ids), slots=slots)

    def collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> Batch:
        pad_id = getattr(self.tokenizer, "pad_token_id", 50283)
        padded_layouts: list[TokenLayout] = []
        all_targets: list[SampleTargets] = []

        for c in cases:
            questions = [q.shuffle() if shuffle and isinstance(q, Choice) else q for q in c.questions]
            layout = self.encode(c.state, questions)
            padded_layouts.append(layout)

            slots = []
            for q, pos in zip(questions, layout.slots):
                if isinstance(q, Choice):
                    slots.append(TargetSlot("choice", pos, q.target_index))
                else:
                    target_type = "score" if isinstance(q, Score) else "noul"
                    target_val = float(q.target) if q.target is not None else 0.0
                    slots.append(TargetSlot(target_type, pos, target_val))
            all_targets.append(SampleTargets(slots))

        max_len = max((len(l.input_ids) for l in padded_layouts), default=0)
        padded_ids = torch.full((len(padded_layouts), max_len), pad_id, dtype=torch.long)
        masks = torch.zeros((len(padded_layouts), max_len), dtype=torch.float)

        for i, layout in enumerate(padded_layouts):
            sz = len(layout.input_ids)
            padded_ids[i, :sz] = torch.as_tensor(layout.input_ids, dtype=torch.long)
            masks[i, :sz] = 1.0

        return Batch(input_ids=padded_ids, mask=masks, targets=all_targets)
