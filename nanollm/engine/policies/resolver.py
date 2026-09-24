from typing import Any, Dict, List, Protocol, Sequence
import torch
from nanollm.engine.schema import Answer, Choice, ChoiceResult, Noul, NoulResult, Score, ScoreResult

class IResolver(Protocol):
    def resolve(self, questions: Sequence[Any], slots: List[List[int]], sample_scores: torch.Tensor) -> Dict[str, Answer]: ...

class DecisionResolver(IResolver):
    def resolve(self, questions: Sequence[Any], slots: List[List[int]], sample_scores: torch.Tensor) -> Dict[str, Answer]:
        answers: Dict[str, Answer] = {}
        for q, pos in zip(questions, slots):
            logits = sample_scores[pos]
            if isinstance(q, Choice):
                probs = logits.softmax(dim=-1)
                best_i = int(probs.argmax().item())
                opt_keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                answers[q.name] = ChoiceResult(
                    choice=opt_keys[best_i],
                    confidence=float(probs[best_i].item()),
                    probabilities={k: float(probs[j].item()) for j, k in enumerate(opt_keys)},
                )
            elif isinstance(q, Noul):
                p = float(torch.sigmoid(logits[0]).item())
                answers[q.name] = NoulResult(value=p >= 0.5, probability=p)
            elif isinstance(q, Score):
                norm = float(torch.sigmoid(logits[0]).item())
                answers[q.name] = ScoreResult(
                    score=q.min_value + norm * (q.max_value - q.min_value),
                    normalized=norm,
                )
        return answers
