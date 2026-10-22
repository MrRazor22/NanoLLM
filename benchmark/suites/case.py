from dataclasses import dataclass
from inference.options import IAnswer, IOption

@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    state: str
    option: IOption
    expected: IAnswer
    category: str
