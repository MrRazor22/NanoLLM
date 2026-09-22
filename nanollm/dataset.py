from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch
from nanollm.tokenizer import ByteTokenizer

@dataclass(frozen=True)
class DecisionSample:
    state: str
    questions: List[Tuple[str, str, Any]]

class MultiQuestionCollator:
    def __init__(self, tokenizer: ByteTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[DecisionSample]) -> Dict[str, Any]:
        all_ids: List[List[int]] = []
        all_positions: List[List[int]] = []
        all_targets: List[List[Any]] = []
        types = [q[1] for q in batch[0].questions]

        for sample in batch:
            ids = self.tokenizer.encode(sample.state)
            positions: List[int] = []
            targets: List[Any] = []
            for name, q_type, target in sample.questions:
                positions.append(len(ids))
                ids.append(self.tokenizer.q_marker_id)
                ids.extend(self.tokenizer.encode(f" {name}"))
                targets.append(target)
            all_ids.append(ids)
            all_positions.append(positions)
            all_targets.append(targets)

        max_len = max(len(ids) for ids in all_ids)
        padded_ids = [ids + [self.tokenizer.pad_id] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "mask": torch.tensor(masks, dtype=torch.float),
            "question_positions": torch.tensor(all_positions, dtype=torch.long),
            "targets": torch.tensor(all_targets, dtype=torch.float),
            "types": types
        }

def load_jsonl(path: str) -> List[DecisionSample]:
    import json
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                samples.append(DecisionSample(state=item["state"], questions=[tuple(q) for q in item["questions"]]))
    return samples
