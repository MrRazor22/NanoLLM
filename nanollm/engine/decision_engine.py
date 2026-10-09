from pathlib import Path
from typing import Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch
from transformers import AutoModel, AutoTokenizer

from nanollm.trainer.policies.registry import DEFAULT_CHECKPOINT, resolve_default_checkpoint
from nanollm.engine.questions import DecisionResult, DecisionState, Question
from nanollm.engine.policies.slot_encoder import ILayoutEncoder, SlotEncoder
from nanollm.engine.layers.profiling_layer import ProfilingLayer
from nanollm.engine.policies.decision_resolver import DecisionResolver, IResolver
from nanollm.engine.policies.torch_scorer import IScorer, TorchScorer, _ScoringHead

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Operational boundary for neural decision inference (Generic & Multimodal)."""

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        """Executes one-shot typed decisions over the provided state context."""
        ...

class DecisionEngine(IDecisionEngine):
    """Core Primitive Implementation: Coordinates layout encoding, neural scoring, and logit resolution."""

    def __init__(
        self,
        encoder: ILayoutEncoder,
        scorer: IScorer,
        resolver: Optional[IResolver] = None,
    ):
        self.encoder = encoder
        self.scorer = scorer
        self.resolver = resolver or DecisionResolver()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        backbone_name: str = "answerdotai/ModernBERT-base",
        device: Optional[str] = None,
        max_state_tokens: int = 384,
    ) -> "DecisionEngine":
        ckpt_path = Path(checkpoint_path) if checkpoint_path else resolve_default_checkpoint()
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        scorer = TorchScorer.from_backbone(backbone_name, checkpoint_path=str(ckpt_path), device=dev)
        scorer.model.eval()

        encoder = SlotEncoder(tokenizer, max_state_tokens=max_state_tokens)
        resolver = DecisionResolver()
        return cls(encoder, scorer, resolver)

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        layout = self.encoder.encode(state, questions)
        score_layout = self.scorer.score(layout)
        answers = self.resolver.resolve(questions, layout.slot_indices, score_layout.scores)
        return DecisionResult(answers=answers)

    def decide_batch(
        self, batch: Sequence[Tuple[DecisionState, Sequence[Question]]]
    ) -> Sequence[DecisionResult]:
        return [self.decide(s, q) for s, q in batch]

    def add(self, layer: Union[type, ProfilingLayer]) -> IDecisionEngine:
        if isinstance(layer, type):
            return layer(self)
        if hasattr(layer, "attach"):
            return layer.attach(self)
        return layer

    def __or__(self, layer: Union[type, ProfilingLayer]) -> IDecisionEngine:
        return self.add(layer)

__all__ = ["IDecisionEngine", "DecisionEngine", "DEFAULT_CHECKPOINT"]
