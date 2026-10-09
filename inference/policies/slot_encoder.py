import torch
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
from inference.questions import DecisionState, Question, TokenLayout, format_state
from training.batches import TargetSlot

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

class SlotEncoder(ILayoutEncoder):
    """Encodes state and questions into token layout with masked slots, optionally extracting target annotations."""
    def __init__(self, tokenizer: ITokenizer, max_state_tokens: int = 384):
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens
        self.mask_id = getattr(tokenizer, "mask_token_id", 50281)
        self.sep_id = getattr(tokenizer, "sep_token_id", getattr(tokenizer, "eos_token_id", 50282))
        self.pad_id = getattr(tokenizer, "pad_token_id", 50283)

    def encode(self, state: DecisionState, questions: Sequence[Question]) -> TokenLayout:
        ids, slots, _ = self.encode_with_targets(state, questions)
        return TokenLayout(input_ids=ids, mask=[1.0] * len(ids), slots=slots)

    def encode_with_targets(self, state: DecisionState, questions: Sequence[Question]) -> Tuple[List[int], List[List[int]], List[TargetSlot]]:
        state_ids = self.tokenizer.encode(format_state(state))
        ids = state_ids[:self.max_state_tokens] if len(state_ids) > self.max_state_tokens else list(state_ids)
        slots, targets = [], []
        for q in questions:
            ids.append(self.sep_id)
            ins = getattr(q, "instruction", None)
            ids.extend(self.tokenizer.encode(f" {q.name}: {ins}" if ins else f" {q.name}:"))
            opts, tgt = getattr(q, "options", None), getattr(q, "target", None)
            q_type = getattr(q, "q_type", "choice" if opts is not None else "noul")
            if opts:
                pos = []
                opt_items = opts.items() if isinstance(opts, dict) else [(o, None) for o in opts]
                for k, v in opt_items:
                    ids.append(self.mask_id)
                    pos.append(len(ids) - 1)
                    ids.extend(self.tokenizer.encode(f" {k}: {v}" if v else f" {k}"))
                slots.append(pos)
                targets.append(TargetSlot(q_type, pos, int(tgt) if tgt is not None else 0))
            else:
                ids.append(self.mask_id)
                pos = [len(ids) - 1]
                slots.append(pos)
                targets.append(TargetSlot(q_type, pos, float(tgt) if tgt is not None else 0.0))
        return ids, slots, targets

    def collate_batch(self, items: Sequence[Tuple[List[int], List[TargetSlot]]], device: Optional[torch.device] = None) -> Dict[str, Union[torch.Tensor, Sequence[Sequence[int]], Sequence[Sequence[TargetSlot]]]]:
        dev = device or torch.device("cpu")
        all_ids, batch_meta = [it[0] for it in items], [it[1] for it in items]
        max_len, bsz = max(len(ids) for ids in all_ids), len(all_ids)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((bsz, max_len), dtype=torch.float, device=dev)
        for i, ids in enumerate(all_ids):
            sz = len(ids)
            input_ids[i, :sz] = torch.as_tensor(ids, dtype=torch.long)
            mask[i, :sz] = 1.0
        return {"input_ids": input_ids, "mask": mask, "slots": [[m.indices for m in row] for row in batch_meta], "targets": batch_meta, "meta": batch_meta}

__all__ = ["ITokenizer", "ILayoutEncoder", "SlotEncoder"]
