from dataclasses import dataclass
from typing import Protocol, Self, Sequence, runtime_checkable
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from data.case import DecisionCase

@dataclass(frozen=True, slots=True)
class BatchTokens:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    candidate_idx: Sequence[torch.Tensor]

    def to(self, device: torch.device) -> Self:
        return BatchTokens(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            candidate_idx=[idx.to(device) for idx in self.candidate_idx],
        )

@runtime_checkable
class ILayout(Protocol):
    def encode(self, cases: Sequence[DecisionCase]) -> BatchTokens: ...

class MaskedLayout(ILayout):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_sequence_tokens: int):
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id.")
        self.tokenizer = tokenizer
        self.max_sequence_tokens = max_sequence_tokens

    def encode(self, cases: Sequence[DecisionCase]) -> BatchTokens:
        if not cases:
            empty = torch.empty((0, 0), dtype=torch.long)
            return BatchTokens(input_ids=empty, attention_mask=empty, candidate_idx=[])

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id

        batch_token_ids = []
        batch_candidate_idx = []

        for case in cases:
            ids = [cls_id, sep_id]
            candidate_idx = []
            for cand in case.candidates:
                ids.append(mask_id)
                candidate_idx.append(len(ids) - 1)
                ids.extend(self.tokenizer.encode(f" {cand}", add_special_tokens=False))

            ids.append(sep_id)
            if case.instruction:
                ids.extend(self.tokenizer.encode(case.instruction, add_special_tokens=False))
                ids.append(sep_id)

            available_budget = max(0, self.max_sequence_tokens - (len(ids) + 1))
            state_tokens = self.tokenizer.encode(case.state, add_special_tokens=False)
            if len(state_tokens) > available_budget:
                state_tokens = state_tokens[:available_budget]
            ids.extend(state_tokens)
            ids.append(sep_id)

            batch_token_ids.append(torch.as_tensor(ids, dtype=torch.long))
            batch_candidate_idx.append(torch.as_tensor(candidate_idx, dtype=torch.long))

        input_ids = pad_sequence(batch_token_ids, batch_first=True, padding_value=pad_id)
        attention_mask = (input_ids != pad_id).long()

        return BatchTokens(input_ids=input_ids, attention_mask=attention_mask, candidate_idx=batch_candidate_idx)
