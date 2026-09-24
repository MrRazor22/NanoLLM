from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
import torch
from nanollm.policies.tokenizer import ITokenizer

@dataclass(frozen=True)
class CompiledLayout:
    input_ids: torch.Tensor
    mask: torch.Tensor
    slots: List[List[int]]

class ISlotAssembler(Protocol):
    def assemble_single(self, state: str, questions: Sequence[Any], device: Optional[torch.device] = None) -> CompiledLayout: ...
    def assemble_batch(self, batch: Sequence[Any]) -> Dict[str, Any]: ...

class SlotAssembler(ISlotAssembler):
    def __init__(self, tokenizer: ITokenizer):
        self.tokenizer = tokenizer

    def _render_sample(self, state: str, questions: Sequence[Any]) -> Tuple[List[int], List[Tuple[str, List[int], Any]]]:
        ids = self.tokenizer.encode(state)
        meta = []
        for q in questions:
            ids.append(self.tokenizer.sep_id)
            ids.extend(self.tokenizer.encode(f" {q.name}:"))
            opts = getattr(q, "options", None)
            target = getattr(q, "target", None)
            q_type = getattr(q, "q_type", "choice" if opts is not None else "noul")
            if opts:
                pos = []
                opt_items = opts.items() if isinstance(opts, dict) else [(o, None) for o in opts]
                for k, v in opt_items:
                    ids.append(self.tokenizer.mask_id)
                    pos.append(len(ids) - 1)
                    text = f" {k}: {v}" if v else f" {k}"
                    ids.extend(self.tokenizer.encode(text))
                meta.append((q_type, pos, int(target) if target is not None else 0))
            else:
                ids.append(self.tokenizer.mask_id)
                meta.append((q_type, [len(ids) - 1], float(target) if target is not None else 0.0))
        return ids, meta

    def assemble_single(self, state: str, questions: Sequence[Any], device: Optional[torch.device] = None) -> CompiledLayout:
        ids, meta = self._render_sample(state, questions)
        dev = device if device is not None else torch.device("cpu")
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        mask = torch.ones((1, len(ids)), dtype=torch.float, device=dev)
        slots = [item[1] for item in meta]
        return CompiledLayout(input_ids=input_ids, mask=mask, slots=slots)

    def assemble_batch(self, batch: Sequence[Any]) -> Dict[str, Any]:
        all_ids: List[List[int]] = []
        batch_meta: List[List[Any]] = []
        for sample in batch:
            ids, meta = self._render_sample(sample.state, sample.questions)
            all_ids.append(ids)
            batch_meta.append(meta)

        max_len = max(len(ids) for ids in all_ids)
        padded = [ids + [self.tokenizer.pad_id] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "mask": torch.tensor(masks, dtype=torch.float),
            "meta": batch_meta,
        }
