from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
from nanollm.inference.policies.assembler import SlotAssembler

@dataclass(frozen=True)
class QuestionSpec:
    name: str
    q_type: str
    target: Any
    options: Optional[Any] = None
    instruction: Optional[str] = None

@dataclass(frozen=True)
class DecisionSample:
    state: str
    questions: List[QuestionSpec]

class MultiQuestionCollator:
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

def to_decision_sample(item: Dict[str, Any]) -> DecisionSample:
    specs = [
        QuestionSpec(
            name=q[0],
            q_type=q[1],
            target=q[2],
            options=q[3] if len(q) > 3 else None,
            instruction=q[4] if len(q) > 4 else None,
        )
        for q in item["questions"]
    ]
    return DecisionSample(state=item["state"], questions=specs)

def load_jsonl(path: str) -> List[DecisionSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(to_decision_sample(json.loads(line)))
    return samples

