from typing import Dict, List, Protocol, Sequence, runtime_checkable
import torch
import torch.nn.functional as F
from nanollm.inference.schema import Answer, Choice, ChoiceResult, Noul, NoulResult, Question, Score, ScoreResult

@runtime_checkable
class IDecisionResolver(Protocol):
    """ATA Policy Contract: Decodes raw slot logits into typed domain Answers."""
    def resolve(
        self,
        questions: Sequence[Question],
        slots: Sequence[List[int]],
        raw_scores: torch.Tensor,
    ) -> Dict[str, Answer]: ...

class DecisionResolver(IDecisionResolver):
    """ATA Injected Policy (π): Resolves raw logits into Choice, Noul, and Score results."""

    def resolve(
        self,
        questions: Sequence[Question],
        slots: Sequence[List[int]],
        raw_scores: torch.Tensor,
    ) -> Dict[str, Answer]:
        answers: Dict[str, Answer] = {}
        for q, slot_indices in zip(questions, slots):
            if isinstance(q, Choice):
                opt_keys = list(q.options.keys()) if isinstance(q.options, dict) else list(q.options)
                logits = raw_scores[slot_indices].squeeze(-1)
                probs = F.softmax(logits, dim=-1)
                idx = int(torch.argmax(probs).item())
                prob_dict = {k: float(p.item()) for k, p in zip(opt_keys, probs)}
                answers[q.name] = ChoiceResult(choice=opt_keys[idx], confidence=float(probs[idx].item()), probabilities=prob_dict)
            elif isinstance(q, Noul):
                prob = float(torch.sigmoid(raw_scores[slot_indices[0]]).item())
                answers[q.name] = NoulResult(value=prob >= 0.5, probability=prob)
            elif isinstance(q, Score):
                val = float(raw_scores[slot_indices[0]].item())
                norm = max(0.0, min(1.0, (val - q.min_value) / max(1e-5, (q.max_value - q.min_value))))
                answers[q.name] = ScoreResult(score=val, normalized=norm)
        return answers

__all__ = ["IDecisionResolver", "DecisionResolver"]
