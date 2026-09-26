from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
import torch
from transformers import AutoTokenizer

class ISlotAssembler(Protocol):
    def assemble(self, samples: Union[Any, Sequence[Any]], device: Optional[torch.device] = None) -> Dict[str, Any]: ...

class SlotAssembler(ISlotAssembler):
    def __init__(self, tokenizer_or_name: Any):
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
        self.vocab_size = getattr(tok, "vocab_size", 50280)

    @staticmethod
    def _format_state(state: Any) -> str:
        if isinstance(state, dict): return " | ".join(f"{k}: {v}" for k, v in state.items())
        if isinstance(state, str) and state.startswith("{") and state.endswith("}"):
            try:
                import json
                d = json.loads(state)
                if isinstance(d, dict): return " | ".join(f"{k}: {v}" for k, v in d.items())
            except Exception: pass
        return str(state)

    def render_sample(self, state: str, questions: Sequence[Any]) -> Tuple[List[int], List[Tuple[str, List[int], Any]]]:
        state_ids = self.tokenizer.encode(self._format_state(state))
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
                    text = f" {k}: {v}" if v else f" {k}"
                    ids.extend(self.tokenizer.encode(text))
                meta.append((q_type, pos, int(target) if target is not None else 0))
            else:
                ids.append(self.mask_id)
                meta.append((q_type, [len(ids) - 1], float(target) if target is not None else 0.0))
        return ids, meta

    _render_sample = render_sample

    def assemble(self, samples: Union[Any, Sequence[Any]], device: Optional[torch.device] = None) -> Dict[str, Any]:
        dev = device if device is not None else torch.device("cpu")
        is_single = not isinstance(samples, (list, tuple)) or (isinstance(samples, tuple) and len(samples) == 2 and isinstance(samples[0], list))
        batch = [samples] if is_single else list(samples)

        all_ids: List[List[int]] = []
        batch_meta: List[List[Any]] = []
        for item in batch:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], list):
                ids, meta = item
            elif hasattr(item, "state") and hasattr(item, "questions"):
                ids, meta = self.render_sample(item.state, item.questions)
            elif isinstance(item, dict) and "state" in item and "questions" in item:
                ids, meta = self.render_sample(item["state"], item["questions"])
            else:
                raise ValueError(f"Unsupported sample format: {type(item)}")
            all_ids.append(ids)
            batch_meta.append(meta)

        max_len = max(len(ids) for ids in all_ids)
        padded = [ids + [self.pad_id] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long, device=dev),
            "mask": torch.tensor(masks, dtype=torch.float, device=dev),
            "slots": [item[1] for item in batch_meta[0]] if is_single else [[item[1] for item in m] for m in batch_meta],
            "meta": batch_meta,
        }

    def assemble_single(self, state: str, questions: Sequence[Any], device: Optional[torch.device] = None) -> Any:
        return self.assemble(type("Sample", (), {"state": state, "questions": questions})(), device=device)

    def assemble_batch(self, batch: Sequence[Any]) -> Dict[str, Any]:
        return self.assemble(batch)
