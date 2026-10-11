from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence, Union, runtime_checkable
import torch
from transformers import AutoTokenizer

from inference.questions import DecisionResult, DecisionState, Question
from inference.policies.decision_resolver import DecisionResolver, IResolver
from inference.policies.slot_encoder import ILayoutEncoder, SlotEncoder
from inference.policies.torch_scorer import IScorer, TorchScorer

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Sovereign operational boundary for neural decision inference."""
    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates layout encoding, neural scoring, and logit resolution."""

    def __init__(
        self,
        encoder: Optional[ILayoutEncoder] = None,
        scorer: Optional[IScorer] = None,
        resolver: Optional[IResolver] = None,
    ):
        self.encoder = encoder or SlotEncoder()
        self.scorer = scorer or TorchScorer.from_backbone()
        self.resolver = resolver or DecisionResolver()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        backbone_name: str = "answerdotai/ModernBERT-base",
        device: Optional[str] = None,
        max_state_tokens: int = 384,
    ) -> "DecisionEngine":
        from training.checkpoints import resolve_default_checkpoint
        ckpt_path = Path(checkpoint_path) if checkpoint_path else resolve_default_checkpoint()
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        encoder = SlotEncoder(tokenizer, max_state_tokens=max_state_tokens)
        dev = torch.device(device) if device else None
        scorer = TorchScorer.from_backbone(
            backbone_name=backbone_name,
            checkpoint_path=ckpt_path,
            device=dev,
        )
        return cls(encoder=encoder, scorer=scorer, resolver=DecisionResolver())

    @property
    def model(self):
        return getattr(self.scorer, "model", None)

    @property
    def device(self):
        return getattr(self.scorer, "device", None)

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        layout = self.encoder.encode(state, questions)
        scores = self.scorer.score(layout).scores
        answers = self.resolver.resolve(questions, layout.slot_indices, scores)
        return DecisionResult(answers=answers)

    def __or__(self, layer: Union[type, Callable[["IDecisionEngine"], "IDecisionEngine"]]) -> "IDecisionEngine":
        return layer(self) if isinstance(layer, type) else (getattr(layer, "attach")(self) if hasattr(layer, "attach") else layer(self))

__all__ = ["IDecisionEngine", "DecisionEngine"]
