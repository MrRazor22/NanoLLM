from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Union
import torch
from transformers import AutoModel
from pipeline import PipelineComposable
from nanollm.inference.assembler import ISlotAssembler, SlotAssembler
from nanollm.inference.resolver import DecisionResolver, IDecisionResolver
from nanollm.inference.schema import DecisionResult, Question
from nanollm.model import ModelConfig, NanoModel

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "checkpoint_champion_v4.pt"

class IDecisionEngine(Protocol):
    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(PipelineComposable, IDecisionEngine):
    """ATA Root Primitive (P): Executes neural model forward pass and coordinates slot extraction."""

    def __init__(
        self,
        model: NanoModel,
        assembler: ISlotAssembler,
        device: torch.device,
        resolver: Optional[IDecisionResolver] = None,
    ):
        self.model = model
        self.assembler = assembler
        self.device = device
        self.resolver = resolver or DecisionResolver()
        self._warmup()

    def _warmup(self) -> None:
        if self.device.type == "cuda":
            dummy = torch.zeros((1, 8), dtype=torch.long, device=self.device)
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    self.model(dummy)
            torch.cuda.synchronize()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        backbone_name: str = "answerdotai/ModernBERT-base",
        device: Optional[str] = None,
        resolver: Optional[IDecisionResolver] = None,
    ) -> "DecisionEngine":
        ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        assembler = SlotAssembler(backbone_name)
        backbone = AutoModel.from_pretrained(backbone_name)
        config = ModelConfig(vocab_size=getattr(assembler, "vocab_size", 50280), hidden_dim=768, num_layers=22, num_heads=12)
        model = NanoModel(config, backbone=backbone).to(dev)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))
        model.eval()
        return cls(model, assembler, dev, resolver=resolver)

    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult:
        layout = self.assembler.assemble(type("Sample", (), {"state": state, "questions": questions})(), device=self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                scores = self.model(layout["input_ids"], layout["mask"])
        raw_scores = scores[0]
        answers = self.resolver.resolve(questions, layout["slots"], raw_scores)
        return DecisionResult(answers=answers)
