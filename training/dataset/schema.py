from dataclasses import dataclass
from typing import Any, List, Optional

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
