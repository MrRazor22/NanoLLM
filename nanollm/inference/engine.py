from pathlib import Path
from typing import Optional, Protocol, Sequence, Union
import torch
from transformers import AutoModel
from nanollm.inference.policies.assembler import ISlotAssembler, SlotAssembler
from nanollm.inference.policies.resolver import DecisionResolver, IResolver
from nanollm.inference.policies.tokenizer import SubwordTokenizer
from nanollm.inference.schema import DecisionResult, Question
from nanollm.model import ModelConfig, NanoModel

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "checkpoint_champion_v2.pt"

class IDecisionEngine(Protocol):
    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(IDecisionEngine):
    def __init__(
        self,
        model: NanoModel,
        assembler: ISlotAssembler,
        resolver: IResolver,
        device: torch.device,
    ):
        self.model = model
        self.assembler = assembler
        self.resolver = resolver
        self.device = device
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
    ) -> "DecisionEngine":
        ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = SubwordTokenizer(backbone_name)
        assembler = SlotAssembler(tokenizer)
        resolver = DecisionResolver()
        backbone = AutoModel.from_pretrained(backbone_name)
        config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_dim=768, num_layers=22, num_heads=12)
        model = NanoModel(config, backbone=backbone).to(dev)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))
        model.eval()
        return cls(model, assembler, resolver, dev)

    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult:
        layout = self.assembler.assemble_single(state, questions, device=self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                scores = self.model(layout.input_ids, layout.mask)
        answers = self.resolver.resolve(questions, layout.slots, scores[0])
        return DecisionResult(answers=answers)