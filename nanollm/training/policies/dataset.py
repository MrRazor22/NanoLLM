from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import json
from nanollm.engine.policies.assembler import SlotAssembler

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

    def __call__(self, batch: Sequence[DecisionSample]) -> Dict[str, Any]:
        return self.assembler.assemble_batch(batch)

def load_jsonl(path: str) -> List[DecisionSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
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
                samples.append(DecisionSample(state=item["state"], questions=specs))
    return samples
