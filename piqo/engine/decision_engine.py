from pathlib import Path
from typing import Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch
from transformers import AutoModel, AutoTokenizer

from piqo.trainer.policies.registry import DEFAULT_CHECKPOINT, resolve_default_checkpoint
from piqo.engine.questions import Composable, DecisionResult, DecisionState, Question
from piqo.engine.policies.slot_encoder import ILayoutEncoder, SlotEncoder
from piqo.engine.policies.decision_resolver import DecisionResolver, IResolver
from piqo.engine.policies.torch_scorer import IScorer, TorchScorer

@runtime_checkable
class IDecisionEngine(Protocol):
    """Core Primitive: Operational boundary for neural decision inference (Generic & Multimodal)."""
    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(Composable, IDecisionEngine):
    """Core Primitive Implementation: Coordinates layout encoding, neural scoring, and logit resolution."""
    def __init__(self, encoder: ILayoutEncoder, scorer: IScorer, resolver: Optional[IResolver] = None):
        self.encoder, self.scorer, self.resolver = encoder, scorer, resolver or DecisionResolver()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Optional[Union[str, Path]] = None, backbone_name: str = "answerdotai/ModernBERT-base", device: Optional[str] = None, max_state_tokens: int = 384) -> "DecisionEngine":
        ckpt_path = Path(checkpoint_path) if checkpoint_path else resolve_default_checkpoint()
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        scorer = TorchScorer.from_backbone(backbone_name, checkpoint_path=str(ckpt_path), device=dev)
        scorer.model.eval()
        return cls(SlotEncoder(tokenizer, max_state_tokens=max_state_tokens), scorer, DecisionResolver())

    def decide(self, state: DecisionState, questions: Sequence[Question]) -> DecisionResult:
        layout = self.encoder.encode(state, questions)
        score_layout = self.scorer.score(layout)
        answers = self.resolver.resolve(questions, layout.slot_indices, score_layout.scores)
        return DecisionResult(answers=answers)

__all__ = ["IDecisionEngine", "DecisionEngine", "DEFAULT_CHECKPOINT"]
