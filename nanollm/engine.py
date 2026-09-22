from typing import Dict, List, Optional, Protocol
import time
import torch
from transformers import AutoModel
from nanollm.dataset import DecisionSample, MultiQuestionCollator, QuestionSpec
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
        tokenizer: SubwordTokenizer,
        collator: MultiQuestionCollator,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.collator = collator
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        backbone_name: str = "answerdotai/ModernBERT-base",
        device: Optional[str] = None,
    ) -> IDecisionEngine:
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = SubwordTokenizer(backbone_name)
        collator = MultiQuestionCollator(tokenizer)
        backbone = AutoModel.from_pretrained(backbone_name)
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=768,
            num_layers=22,
            num_heads=12,
            proj_dim=256,
        )
        model = NanoModel(config, backbone=backbone).to(dev)
        if torch.cuda.is_available() and dev.type == "cuda":
            model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return cls(model, tokenizer, collator, dev)

    def decide(self, state: str, questions: List[Question]) -> DecisionResult:
        specs = []
        for q in questions:
            if isinstance(q, Choice):
                specs.append(QuestionSpec(name=q.name, q_type="choice", target=0, options=q.options))
            elif isinstance(q, Noul):
                specs.append(QuestionSpec(name=q.name, q_type="noul", target=0.0))
            elif isinstance(q, Score):
                specs.append(QuestionSpec(name=q.name, q_type="score", target=0.0))

        sample = DecisionSample(state=state, questions=specs)
        batch = self.collator([sample])
        input_ids = batch["input_ids"].to(self.device)
        positions = batch["question_positions"].to(self.device)
        mask = batch["mask"].to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_ids, positions, mask)
            opt_logits: Dict[str, torch.Tensor] = {}
            for i, q in enumerate(questions):
                if isinstance(q, Choice):
                    opt_tokens = [self.tokenizer.encode(opt) for opt in q.options]
                    max_len = max(len(t) for t in opt_tokens)
                    padded = [t + [self.tokenizer.pad_id] * (max_len - len(t)) for t in opt_tokens]
                    masks = [[1] * len(t) + [0] * (max_len - len(t)) for t in opt_tokens]
                    opt_ids_t = torch.tensor(padded, dtype=torch.long, device=self.device)
                    opt_mask_t = torch.tensor(masks, dtype=torch.float, device=self.device)
                    opt_vecs = self.model.encode_options(opt_ids_t, opt_mask_t)
                    q_vec = outputs["q_choice"][0, i]
                    opt_logits[q.name] = outputs["scale"] * (q_vec @ opt_vecs.T)

        latency_ms = (time.perf_counter() - start) * 1000.0

        answers: Dict[str, Answer] = {}
        for i, q in enumerate(questions):
            if isinstance(q, Choice):
                probs = opt_logits[q.name].softmax(dim=-1)
                idx = int(probs.argmax().item())
                answers[q.name] = ChoiceResult(
                    choice=q.options[idx],
                    confidence=float(probs[idx].item()),
                    probabilities={opt: float(probs[j].item()) for j, opt in enumerate(q.options)},
                )
            elif isinstance(q, Noul):
                p = float(torch.sigmoid(outputs["noul"][0, i]).item())
                answers[q.name] = NoulResult(value=p >= 0.5, probability=p)
            elif isinstance(q, Score):
                norm = float(torch.sigmoid(outputs["score"][0, i]).item())
                answers[q.name] = ScoreResult(
                    score=q.min_value + norm * (q.max_value - q.min_value),
                    normalized=norm,
                )

        return DecisionResult(answers=answers, latency_ms=latency_ms)
