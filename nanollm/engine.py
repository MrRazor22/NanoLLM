from typing import Dict, List, Optional, Protocol
import time
import torch
from transformers import AutoModel
from nanollm.dataset import DecisionSample, MultiQuestionCollator
from nanollm.model import ModelConfig, NanoModel
from nanollm.schema import (
    Answer,
    Choice,
    ChoiceResult,
    DecisionResult,
    Noul,
    NoulResult,
    Question,
    Score,
    ScoreResult,
)
from nanollm.tokenizer import SubwordTokenizer

class IDecisionEngine(Protocol):
    def decide(self, state: str, questions: List[Question]) -> DecisionResult: ...

class DecisionEngine(IDecisionEngine):
    def __init__(
        self,
        model: NanoModel,
        collator: MultiQuestionCollator,
        device: torch.device,
    ):
        self.model = model
        self.collator = collator
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        backbone_name: str = "bert-base-uncased",
        device: Optional[str] = None,
    ) -> IDecisionEngine:
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = SubwordTokenizer(backbone_name)
        collator = MultiQuestionCollator(tokenizer)
        backbone = AutoModel.from_pretrained(backbone_name)
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            max_choices=16,
        )
        model = NanoModel(config, backbone=backbone).to(dev)
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
        model.eval()
        return cls(model, collator, dev)

    def decide(self, state: str, questions: List[Question]) -> DecisionResult:
        spec = []
        for q in questions:
            if isinstance(q, Choice):
                spec.append((q.name, "choice", 0))
            elif isinstance(q, Noul):
                spec.append((q.name, "noul", 0.0))
            elif isinstance(q, Score):
                spec.append((q.name, "score", 0.0))

        sample = DecisionSample(state=state, questions=spec)
        batch = self.collator([sample])
        input_ids = batch["input_ids"].to(self.device)
        positions = batch["question_positions"].to(self.device)
        mask = batch["mask"].to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(input_ids, positions, mask)
        latency_ms = (time.perf_counter() - start) * 1000.0

        answers: Dict[str, Answer] = {}
        for i, q in enumerate(questions):
            if isinstance(q, Choice):
                probs = logits[0, i, :len(q.options)].softmax(dim=-1)
                idx = int(probs.argmax().item())
                answers[q.name] = ChoiceResult(
                    choice=q.options[idx],
                    confidence=float(probs[idx].item()),
                    probabilities={opt: float(probs[j].item()) for j, opt in enumerate(q.options)},
                )
            elif isinstance(q, Noul):
                p = float(torch.sigmoid(logits[0, i, 0]).item())
                answers[q.name] = NoulResult(value=p >= 0.5, probability=p)
            elif isinstance(q, Score):
                norm = float(torch.sigmoid(logits[0, i, 0]).item())
                answers[q.name] = ScoreResult(
                    score=q.min_value + norm * (q.max_value - q.min_value),
                    normalized=norm,
                )

        return DecisionResult(answers=answers, latency_ms=latency_ms)
