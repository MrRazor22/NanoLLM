from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import torch
from nanollm.tokenizer import ByteTokenizer

@dataclass(frozen=True)
class QuestionSpec:
    name: str
    q_type: str
    target: Any
    options: Optional[List[str]] = None

@dataclass(frozen=True)
class DecisionSample:
    state: str
    questions: List[QuestionSpec]

class MultiQuestionCollator:
    def __init__(self, tokenizer: ByteTokenizer):
        self.tokenizer = tokenizer

    def _collate_options(self, batch: List[DecisionSample]) -> Dict[str, Any]:
        opt_ids_list: List[List[int]] = []
        sample_opt_slices: List[List[int]] = []
        for sample in batch:
            slices = []
            for q in sample.questions:
                if q.options:
                    start = len(opt_ids_list)
                    for opt in q.options:
                        opt_ids_list.append(self.tokenizer.encode(opt))
                    slices.append(start)
                    slices.append(len(opt_ids_list))
            sample_opt_slices.append(slices)

        if not opt_ids_list:
            return {"opt_ids": None, "opt_mask": None, "opt_slices": sample_opt_slices}

        max_len = max(len(ids) for ids in opt_ids_list)
        padded = [ids + [self.tokenizer.pad_id] * (max_len - len(ids)) for ids in opt_ids_list]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in opt_ids_list]
        return {
            "opt_ids": torch.tensor(padded, dtype=torch.long),
            "opt_mask": torch.tensor(masks, dtype=torch.float),
            "opt_slices": sample_opt_slices
        }

    def __call__(self, batch: List[DecisionSample]) -> Dict[str, Any]:
        all_ids: List[List[int]] = []
        all_positions: List[List[int]] = []
        all_targets: List[List[Any]] = []
        types = [q.q_type for q in batch[0].questions]

        for sample in batch:
            ids = self.tokenizer.encode(sample.state)
            positions: List[int] = []
            targets: List[Any] = []
            for q in sample.questions:
                positions.append(len(ids))
                ids.append(self.tokenizer.q_marker_id)
                ids.extend(self.tokenizer.encode(f" {q.name}"))
                targets.append(q.target)
            all_ids.append(ids)
            all_positions.append(positions)
            all_targets.append(targets)

        max_len = max(len(ids) for ids in all_ids)
        padded_ids = [ids + [self.tokenizer.pad_id] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]
        opt_data = self._collate_options(batch)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "mask": torch.tensor(masks, dtype=torch.float),
            "question_positions": torch.tensor(all_positions, dtype=torch.long),
            "targets": torch.tensor(all_targets, dtype=torch.float),
            "types": types,
            **opt_data
        }

def load_jsonl(path: str) -> List[DecisionSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                specs = []
                for q in item["questions"]:
                    opts = q[3] if len(q) > 3 else None
                    specs.append(QuestionSpec(name=q[0], q_type=q[1], target=q[2], options=opts))
                samples.append(DecisionSample(state=item["state"], questions=specs))
    return samples
