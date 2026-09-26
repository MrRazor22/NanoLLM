from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Union
import torch
import torch.nn.functional as F
from transformers import AutoModel
from nanollm.inference.policies.assembler import ISlotAssembler, SlotAssembler
from nanollm.inference.schema import Answer, Choice, ChoiceResult, DecisionResult, Noul, NoulResult, Question, Score, ScoreResult
from nanollm.model import ModelConfig, NanoModel

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "checkpoint_champion_v4.pt"

class IDecisionEngine(Protocol):
    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult: ...

class DecisionEngine(IDecisionEngine):
    def __init__(self, model: NanoModel, assembler: ISlotAssembler, device: torch.device):
        self.model = model
        self.assembler = assembler
        self.device = device
        self._warmup()

    def add(self, layer: Any, **kwargs: Any) -> "IDecisionEngine":
        if isinstance(layer, type): return layer(self, **kwargs)
        if hasattr(layer, "attach"): return layer.attach(self)
        return layer(self, **kwargs)

    def __or__(self, layer: Any) -> "IDecisionEngine":
        return self.add(layer)

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
        assembler = SlotAssembler(backbone_name)
        backbone = AutoModel.from_pretrained(backbone_name)
        config = ModelConfig(vocab_size=getattr(assembler, "vocab_size", 50280), hidden_dim=768, num_layers=22, num_heads=12)
        model = NanoModel(config, backbone=backbone).to(dev)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))
        model.eval()
        return cls(model, assembler, dev)

    def decide(self, state: str, questions: Sequence[Question]) -> DecisionResult:
        layout = self.assembler.assemble(type("Sample", (), {"state": state, "questions": questions})(), device=self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                scores = self.model(layout["input_ids"], layout["mask"])
        raw_scores = scores[0]

        answers: Dict[str, Answer] = {}
        for q, slot_indices in zip(questions, layout["slots"]):
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
        return DecisionResult(answers=answers)