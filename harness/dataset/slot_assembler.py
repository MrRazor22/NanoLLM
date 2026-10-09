from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch
from transformers import AutoTokenizer
from nanollm.engine.policies.slot_encoder import ITokenizer
from nanollm.engine.questions import DecisionState, Question, format_state
from nanollm.trainer.batches import TargetSlot

class SlotAssembler:
    """Renders domain state and question slots into token indices and builds collated batch tensors."""

    def __init__(self, tokenizer_or_name: Union[str, ITokenizer]):
        if isinstance(tokenizer_or_name, str):
            tok = AutoTokenizer.from_pretrained(tokenizer_or_name)
        elif hasattr(tokenizer_or_name, "encode"):
            tok = tokenizer_or_name
        else:
            raise ValueError(f"Invalid tokenizer: {tokenizer_or_name}")
        self.tokenizer = tok
        self.mask_id = getattr(tok, "mask_token_id", 50281)
        self.sep_id = getattr(tok, "sep_token_id", getattr(tok, "eos_token_id", 50282))
        self.pad_id = getattr(tok, "pad_token_id", 50283)

    def render_sample(self, state: DecisionState, questions: Sequence[Question]) -> Tuple[List[int], List[TargetSlot]]:
        text = format_state(state)
        state_ids = self.tokenizer.encode(text)
        ids = state_ids[:384] if len(state_ids) > 384 else list(state_ids)
        meta = []
        for q in questions:
            ids.append(self.sep_id)
            ins = getattr(q, "instruction", None)
            header = f" {q.name}: {ins}" if ins else f" {q.name}:"
            ids.extend(self.tokenizer.encode(header))
            opts = getattr(q, "options", None)
            target = getattr(q, "target", None)
            q_type = getattr(q, "q_type", "choice" if opts is not None else "noul")
            if opts:
                pos = []
                opt_items = opts.items() if isinstance(opts, dict) else [(o, None) for o in opts]
                for k, v in opt_items:
                    ids.append(self.mask_id)
                    pos.append(len(ids) - 1)
                    ids.extend(self.tokenizer.encode(f" {k}: {v}" if v else f" {k}"))
                meta.append(TargetSlot(q_type, pos, int(target) if target is not None else 0))
            else:
                ids.append(self.mask_id)
                meta.append(TargetSlot(q_type, [len(ids) - 1], float(target) if target is not None else 0.0))
        return ids, meta

    def assemble_batch(
        self,
        items: Sequence[Tuple[List[int], List[TargetSlot]]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        dev = device or torch.device("cpu")
        all_ids = [it[0] for it in items]
        batch_meta = [it[1] for it in items]
        max_len = max(len(ids) for ids in all_ids)
        bsz = len(all_ids)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((bsz, max_len), dtype=torch.float, device=dev)
        for i, ids in enumerate(all_ids):
            sz = len(ids)
            input_ids[i, :sz] = torch.as_tensor(ids, dtype=torch.long)
            mask[i, :sz] = 1.0
        return {
            "input_ids": input_ids,
            "mask": mask,
            "slots": [[item[1] for item in m] for m in batch_meta],
            "meta": batch_meta,
        }

@runtime_checkable
class IBatchCollator(Protocol):
    """Collation policy contract: collates a batch of dataset samples into tensors."""
    def __call__(self, batch: Sequence[Any]) -> Dict[str, Any]: ...

class MultiQuestionCollator(IBatchCollator):
    """Pure collation policy that renders and packs token slots into a batch tensor dict."""

    def __init__(self, tokenizer_or_assembler: Any):
        if hasattr(tokenizer_or_assembler, "assemble_batch"):
            self.assembler = tokenizer_or_assembler
        else:
            self.assembler = SlotAssembler(tokenizer_or_assembler)
        self._cache: Dict[int, Tuple[List[int], List[Any]]] = {}

    def __call__(self, batch: Sequence[Any]) -> Dict[str, Any]:
        items = []
        for s in batch:
            k = id(s)
            cached = self._cache.get(k)
            if cached is None:
                cached = self._cache[k] = (
                    s if isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], list)
                    else self.assembler.render_sample(s.state, s.questions)
                )
            items.append(cached)
        return self.assembler.assemble_batch(items)

__all__ = ["SlotAssembler", "IBatchCollator", "MultiQuestionCollator"]
