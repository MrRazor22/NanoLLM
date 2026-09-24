from typing import Optional, Protocol, Sequence
import torch
from transformers import AutoModel
from nanollm.engine.layers.profiling import ProfilingLayer
from nanollm.engine.policies.assembler import ISlotAssembler, SlotAssembler
from nanollm.engine.policies.resolver import DecisionResolver, IResolver
from nanollm.engine.policies.tokenizer import SubwordTokenizer
from nanollm.engine.substrate import DecisionSubstrate, ISubstrate, ModelConfig
from nanollm.schema import DecisionResult, Question

class IDecisionEngine(Protocol):
    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(IDecisionEngine):
    def __init__(
        self,
        substrate: ISubstrate,
        assembler: ISlotAssembler,
        resolver: IResolver,
        device: torch.device,
    ):
        self.substrate = substrate
        self.assembler = assembler
        self.resolver = resolver
        self.device = device
        self._warmup()

    def _warmup(self) -> None:
        if self.device.type == "cuda":
            dummy = torch.zeros((1, 8), dtype=torch.long, device=self.device)
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    self.substrate(dummy)
            torch.cuda.synchronize()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        backbone_name: str = "answerdotai/ModernBERT-base",
        device: Optional[str] = None,
    ) -> IDecisionEngine:
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = SubwordTokenizer(backbone_name)
        assembler = SlotAssembler(tokenizer)
        resolver = DecisionResolver()
        backbone = AutoModel.from_pretrained(backbone_name)
        config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_dim=768, num_layers=22, num_heads=12)
        substrate = DecisionSubstrate(config, backbone=backbone).to(dev)
        substrate.load_state_dict(torch.load(checkpoint_path, map_location=dev))
        substrate.eval()
        core_engine = cls(substrate, assembler, resolver, dev)
        return ProfilingLayer(core_engine)

    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult:
        layout = self.assembler.assemble_single(state, questions, device=self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                scores = self.substrate(layout.input_ids, layout.mask)
        answers = self.resolver.resolve(questions, layout.slots, scores[0])
        return DecisionResult(answers=answers)
