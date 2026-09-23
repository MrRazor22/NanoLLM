// ============================================================================
// FILE: benchmark.py (121 code lines, 135 total)
// ============================================================================

from typing import Dict, List, Tuple
import sys
import time
import torch
from nanollm import Choice, DecisionEngine, Noul, Score

BENCHMARK_SUITE: List[Dict] = [
    {
        "domain": "Medical / Clinical Triage",
        "query": "Patient presents with sudden onset unilateral facial droop, slurred speech, and right arm weakness.",
        "options": ["stroke_emergency", "dermatology_rash", "orthopedic_sprain", "routine_dental"],
        "expected": "stroke_emergency",
        "is_urgent": True
    },
    {
        "domain": "Medical / Clinical Triage",
        "query": "Severe itchy red maculopapular rash developing across torso after taking amoxicillin dose.",
        "options": ["drug_allergy_reaction", "bone_fracture", "concussion", "hypertension"],
        "expected": "drug_allergy_reaction",
        "is_urgent": True
    },
    {
        "domain": "Medical / Clinical Triage",
        "query": "Twisted right ankle during basketball, severe localized swelling and inability to bear weight.",
        "options": ["orthopedic_injury", "stroke_emergency", "cardiac_arrest", "food_poisoning"],
        "expected": "orthopedic_injury",
        "is_urgent": False
    },
    {
        "domain": "Legal & Contracts",
        "query": "Neither party shall disclose confidential trade secrets, client lists, or algorithms to third parties.",
        "options": ["confidentiality_nda", "governing_law", "limitation_of_liability", "severability"],
        "expected": "confidentiality_nda",
        "is_urgent": False
    },
    {
        "domain": "Legal & Contracts",
        "query": "This agreement shall be governed by and construed under the laws of the State of Delaware.",
        "options": ["governing_law_jurisdiction", "force_majeure", "indemnification", "confidentiality_nda"],
        "expected": "governing_law_jurisdiction",
        "is_urgent": False
    },
    {
        "domain": "Legal & Contracts",
        "query": "Supplier shall indemnify and hold harmless the customer from any third party patent infringement claims.",
        "options": ["indemnification_defense", "payment_terms", "term_and_termination", "severability"],
        "expected": "indemnification_defense",
        "is_urgent": True
    },
    {
        "domain": "Developer & Git",
        "query": "Need to save my current uncommitted working directory edits temporarily so I can pull origin main.",
        "options": ["git_stash", "git_rebase", "git_cherry_pick", "git_reset_hard"],
        "expected": "git_stash",
        "is_urgent": False
    },
    {
        "domain": "Developer & Git",
        "query": "Apply commit 7a8b9c from develop branch directly onto release-1.2 branch.",
        "options": ["git_cherry_pick", "git_commit_amend", "git_clean", "git_stash"],
        "expected": "git_cherry_pick",
        "is_urgent": False
    },
    {
        "domain": "Developer & Git",
        "query": "Discard all unstaged and staged changes completely and revert back to commit HEAD.",
        "options": ["git_reset_hard", "git_merge", "git_stash_pop", "git_branch"],
        "expected": "git_reset_hard",
        "is_urgent": True
    }
]

def run_benchmark(engine: DecisionEngine):
    print("\n==================================================")
    print("      SCIENTIFIC ZERO-SHOT OOD BENCHMARK SUITE     ")
    print("==================================================")

    latencies = []
    correct_choices = 0
    correct_nouls = 0
    brier_sum = 0.0

    for item in BENCHMARK_SUITE:
        questions = [
            Choice("category", options=item["options"]),
            Noul("is_urgent"),
            Score("severity", min_value=0.0, max_value=100.0)
        ]

        result = engine.decide(state=item["query"], questions=questions)
        latencies.append(result.latency_ms)

        ans_choice = result.answers["category"]
        ans_noul = result.answers["is_urgent"]

        is_choice_ok = ans_choice.choice == item["expected"]
        is_noul_ok = ans_noul.value == item["is_urgent"]

        if is_choice_ok:
            correct_choices += 1
        if is_noul_ok:
            correct_nouls += 1

        brier_sum += (ans_noul.probability - (1.0 if item["is_urgent"] else 0.0)) ** 2

        status_str = "PASS" if is_choice_ok else "FAIL"
        print(f"\n[{status_str}] [{item['domain']}]")
        print(f"  Query:      {item['query'][:80]}...")
        print(f"  Prediction: {ans_choice.choice} ({ans_choice.confidence * 100:.1f}%) | Expected: {item['expected']}")
        print(f"  Urgent:     {ans_noul.value} (p={ans_noul.probability:.3f}) | Expected: {item['is_urgent']}")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    choice_acc = (correct_choices / len(BENCHMARK_SUITE)) * 100.0
    noul_acc = (correct_nouls / len(BENCHMARK_SUITE)) * 100.0
    brier_score = brier_sum / len(BENCHMARK_SUITE)

    print("\n==================================================")
    print("              FINAL BENCHMARK RESULTS             ")
    print("==================================================")
    print(f"Zero-Shot OOD Accuracy: {choice_acc:.1f}% ({correct_choices}/{len(BENCHMARK_SUITE)})")
    print(f"Calibrated Noul Accuracy: {noul_acc:.1f}% ({correct_nouls}/{len(BENCHMARK_SUITE)})")
    print(f"Epistemic Brier Score:   {brier_score:.4f} (0.0 = perfect)")
    print(f"Inference Latency P50:   {p50:.2f} ms")
    print(f"Inference Latency P95:   {p95:.2f} ms")
    print("==================================================\n")

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint.pt"
    engine = DecisionEngine.from_checkpoint(checkpoint_path)
    run_benchmark(engine)

if __name__ == "__main__":
    main()


// ============================================================================
// FILE: evaluate.py (80 code lines, 94 total)
// ============================================================================

from typing import List
import os
import sys
import time
from nanollm import Choice, DecisionEngine, Noul, Score
from nanollm.dataset import load_jsonl

DEFAULT_DEPARTMENTS = ["Billing", "Infrastructure / Tech", "Enterprise / Sales", "Security"]

def evaluate_dataset(engine: DecisionEngine, path: str):
    if not os.path.exists(path):
        print(f"Dataset path does not exist: {path}")
        return

    samples = load_jsonl(path)
    total_samples = len(samples)
    choice_correct = 0
    noul_correct = 0
    noul_brier_sum = 0.0
    score_mae_sum = 0.0

    start_time = time.perf_counter()
    for sample in samples:
        choice_q = next(q for q in sample.questions if q.q_type == "choice")
        urgent_q = next(q for q in sample.questions if q.q_type == "noul")
        score_q = next(q for q in sample.questions if q.q_type == "score")

        options = choice_q.options if choice_q.options else DEFAULT_DEPARTMENTS
        questions = [
            Choice("department", options=options),
            Noul("is_urgent"),
            Score("severity", min_value=0.0, max_value=100.0)
        ]

        result = engine.decide(state=sample.state, questions=questions)
        dept_res = result.answers["department"]
        target_label = options[choice_q.target]
        if dept_res.choice == target_label:
            choice_correct += 1

        urgent_res = result.answers["is_urgent"]
        target_urgent = urgent_q.target >= 0.5
        if urgent_res.value == target_urgent:
            noul_correct += 1
        noul_brier_sum += (urgent_res.probability - float(target_urgent)) ** 2

        score_res = result.answers["severity"]
        score_mae_sum += abs(score_res.score - (score_q.target * 100.0))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency = elapsed_ms / max(1, total_samples)

    print("==================================================")
    print(f"          STANDARDIZED EVALUATION REPORT          ")
    print("==================================================")
    print(f"Target Dataset:       {path} ({total_samples} samples)")
    print(f"Inference Latency:    {avg_latency:.2f} ms / sample")
    print("--------------------------------------------------")
    print(f"Choice Accuracy:      {(choice_correct / total_samples) * 100:.2f}%")
    print(f"Noul Accuracy:        {(noul_correct / total_samples) * 100:.2f}%")
    print(f"Noul Brier Score:     {noul_brier_sum / total_samples:.4f} (0.0 = perfect)")
    print(f"Score Error (MAE):    {score_mae_sum / total_samples:.2f} points")
    print("==================================================")

def predict_single(engine: DecisionEngine, query: str, options: List[str]):
    questions = [
        Choice("department", options=options),
        Noul("urgent"),
        Score("severity", min_value=0.0, max_value=100.0)
    ]
    result = engine.decide(query, questions)
    dept = result.answers["department"]
    urgent = result.answers["urgent"]
    sev = result.answers["severity"]

    print(f"\nQuery: {query}")
    print(f"  Latency:    {result.latency_ms:.2f} ms")
    print(f"  Selection:  {dept.choice} ({dept.confidence * 100:.1f}%)")
    for opt, prob in dept.probabilities.items():
        print(f"    - {opt}: {prob * 100:.1f}%")
    print(f"  Urgent:     {urgent.value} (p={urgent.probability:.3f})")
    print(f"  Severity:   {sev.score:.1f}/100")

def main():
    engine = DecisionEngine.from_checkpoint("checkpoint.pt")
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        query = " ".join(sys.argv[1:])
        predict_single(engine, query, DEFAULT_DEPARTMENTS)
    else:
        dataset_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--data" else "data/test.jsonl"
        evaluate_dataset(engine, dataset_path)

if __name__ == "__main__":
    main()


// ============================================================================
// FILE: example.py (20 code lines, 26 total)
// ============================================================================

from nanollm import Choice, DecisionEngine, Noul, Score

def main():
    engine = DecisionEngine.from_checkpoint("checkpoint.pt")
    
    state = "Our primary Postgres database CPU reached 99% and connection pool is exhausted."
    questions = [
        Choice("department", options=["Billing", "Infrastructure / Tech", "Enterprise / Sales", "Security"]),
        Noul("urgent"),
        Score("severity", min_value=0.0, max_value=100.0),
    ]

    result = engine.decide(state=state, questions=questions)

    dept = result.answers["department"]
    urgent = result.answers["urgent"]
    sev = result.answers["severity"]

    print(f"State: {state}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    print(f"Department: {dept.choice} ({dept.confidence * 100:.1f}%)")
    print(f"Urgent:     {urgent.value} (p={urgent.probability:.3f})")
    print(f"Severity:   {sev.score:.1f}/100")

if __name__ == "__main__":
    main()


// ============================================================================
// FILE: nanollm/__init__.py (37 code lines, 38 total)
// ============================================================================

from nanollm.dataset import DecisionSample, MultiQuestionCollator, QuestionSpec
from nanollm.engine import DecisionEngine, IDecisionEngine
from nanollm.loss import CalibratedLoss
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
from nanollm.tokenizer import ByteTokenizer, SubwordTokenizer

__all__ = [
    "Answer",
    "ByteTokenizer",
    "CalibratedLoss",
    "Choice",
    "ChoiceResult",
    "DecisionEngine",
    "DecisionResult",
    "DecisionSample",
    "IDecisionEngine",
    "ModelConfig",
    "MultiQuestionCollator",
    "NanoModel",
    "Noul",
    "NoulResult",
    "Question",
    "QuestionSpec",
    "Score",
    "ScoreResult",
    "SubwordTokenizer",
]


// ============================================================================
// FILE: nanollm/dataset.py (82 code lines, 93 total)
// ============================================================================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import torch
from nanollm.tokenizer import ByteTokenizer

@dataclass(frozen=True)
class QuestionSpec:
    name: str
    q_type: str
    target: Any
    options: Optional[List[str]] = None

@dataclass(frozen=True)
class DecisionSample:
    state: str
    questions: List[QuestionSpec]

class MultiQuestionCollator:
    def __init__(self, tokenizer: ByteTokenizer):
        self.tokenizer = tokenizer

    def _collate_options(self, batch: List[DecisionSample]) -> Dict[str, Any]:
        opt_ids_list: List[List[int]] = []
        sample_opt_slices: List[List[int]] = []
        for sample in batch:
            slices = []
            for q in sample.questions:
                if q.options:
                    start = len(opt_ids_list)
                    for opt in q.options:
                        opt_ids_list.append(self.tokenizer.encode(opt))
                    slices.append(start)
                    slices.append(len(opt_ids_list))
            sample_opt_slices.append(slices)

        if not opt_ids_list:
            return {"opt_ids": None, "opt_mask": None, "opt_slices": sample_opt_slices}

        max_len = max(len(ids) for ids in opt_ids_list)
        padded = [ids + [self.tokenizer.pad_id] * (max_len - len(ids)) for ids in opt_ids_list]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in opt_ids_list]
        return {
            "opt_ids": torch.tensor(padded, dtype=torch.long),
            "opt_mask": torch.tensor(masks, dtype=torch.float),
            "opt_slices": sample_opt_slices
        }

    def __call__(self, batch: List[DecisionSample]) -> Dict[str, Any]:
        all_ids: List[List[int]] = []
        all_positions: List[List[int]] = []
        all_targets: List[List[Any]] = []
        types = [q.q_type for q in batch[0].questions]

        for sample in batch:
            ids = self.tokenizer.encode(sample.state)
            positions: List[int] = []
            targets: List[Any] = []
            for q in sample.questions:
                positions.append(len(ids))
                ids.append(self.tokenizer.q_marker_id)
                ids.extend(self.tokenizer.encode(f" {q.name}"))
                targets.append(q.target)
            all_ids.append(ids)
            all_positions.append(positions)
            all_targets.append(targets)

        max_len = max(len(ids) for ids in all_ids)
        padded_ids = [ids + [self.tokenizer.pad_id] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]
        opt_data = self._collate_options(batch)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "mask": torch.tensor(masks, dtype=torch.float),
            "question_positions": torch.tensor(all_positions, dtype=torch.long),
            "targets": torch.tensor(all_targets, dtype=torch.float),
            "types": types,
            **opt_data
        }

def load_jsonl(path: str) -> List[DecisionSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                specs = []
                for q in item["questions"]:
                    opts = q[3] if len(q) > 3 else None
                    specs.append(QuestionSpec(name=q[0], q_type=q[1], target=q[2], options=opts))
                samples.append(DecisionSample(state=item["state"], questions=specs))
    return samples


// ============================================================================
// FILE: nanollm/engine.py (107 code lines, 116 total)
// ============================================================================

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


// ============================================================================
// FILE: nanollm/loss.py (46 code lines, 50 total)
// ============================================================================

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibratedLoss(nn.Module):
    def __init__(self, brier_weight: float = 0.5):
        super().__init__()
        self.brier_weight = brier_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        types: List[str],
        opt_vectors: Optional[torch.Tensor] = None,
        opt_slices: Optional[List[List[int]]] = None
    ) -> torch.Tensor:
        b = targets.shape[0]
        total_loss = torch.tensor(0.0, device=targets.device)
        scale = outputs.get("scale", torch.tensor(1.0, device=targets.device))

        for idx, q_type in enumerate(types):
            q_target = targets[:, idx]
            if q_type == "choice":
                if opt_vectors is not None and opt_slices is not None:
                    choice_losses = []
                    for s_idx in range(b):
                        slices = opt_slices[s_idx]
                        if slices:
                            start, end = slices[0], slices[1]
                            opts = opt_vectors[start:end]
                            q_vec = outputs["q_choice"][s_idx, idx]
                            logits = scale * (q_vec @ opts.T)
                            t = q_target[s_idx].long().unsqueeze(0)
                            choice_losses.append(F.cross_entropy(logits.unsqueeze(0), t))
                    if choice_losses:
                        total_loss = total_loss + torch.stack(choice_losses).mean()
            elif q_type == "noul":
                noul_logits = outputs["noul"][:, idx]
                prob = torch.sigmoid(noul_logits)
                bce = F.binary_cross_entropy_with_logits(noul_logits, q_target)
                brier = F.mse_loss(prob, q_target)
                total_loss = total_loss + (1.0 - self.brier_weight) * bce + self.brier_weight * brier
            elif q_type == "score":
                score_logits = outputs["score"][:, idx]
                prob = torch.sigmoid(score_logits)
                total_loss = total_loss + F.mse_loss(prob, q_target)

        return total_loss / max(1, len(types))


// ============================================================================
// FILE: nanollm/model.py (90 code lines, 101 total)
// ============================================================================

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50280
    hidden_dim: int = 768
    num_layers: int = 22
    num_heads: int = 12
    max_seq_len: int = 8192
    proj_dim: int = 256

class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
        return self.proj(attn.permute(0, 2, 1, 3).reshape(b, s, -1))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))

class NanoModel(nn.Module):
    def __init__(self, config: ModelConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.backbone = backbone
        hidden_dim = backbone.config.hidden_size if backbone is not None else config.hidden_dim

        if backbone is None:
            self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
            self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
            self.ln_f = nn.LayerNorm(config.hidden_dim)

        self.proj_choice = nn.Linear(hidden_dim, config.proj_dim)
        self.proj_option = nn.Linear(hidden_dim, config.proj_dim)
        self.noul_head = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def _encode(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.backbone is not None:
            return self.backbone(input_ids=input_ids, attention_mask=mask).last_hidden_state
        b, s = input_ids.shape
        pos = torch.arange(0, s, device=input_ids.device)
        attn_mask = mask[:, None, None, :].bool() if mask is not None and mask.dim() == 2 else mask
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.ln_f(x)

    def encode_options(self, opt_ids: torch.Tensor, opt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._encode(opt_ids, opt_mask)
        if opt_mask is not None:
            mask_exp = opt_mask.unsqueeze(-1)
            pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        return F.normalize(self.proj_option(pooled), dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        question_positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        b, s = input_ids.shape
        x = self._encode(input_ids, mask)
        b_idx = torch.arange(b, device=input_ids.device).unsqueeze(1)
        q_features = x[b_idx, question_positions]
        q_choice = F.normalize(self.proj_choice(q_features), dim=-1)
        return {
            "q_choice": q_choice,
            "noul": self.noul_head(q_features).squeeze(-1),
            "score": self.score_head(q_features).squeeze(-1),
            "scale": self.logit_scale.exp().clamp(max=100.0)
        }


// ============================================================================
// FILE: nanollm/schema.py (33 code lines, 42 total)
// ============================================================================

from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass(frozen=True)
class Choice:
    name: str
    options: List[str]

@dataclass(frozen=True)
class Noul:
    name: str

@dataclass(frozen=True)
class Score:
    name: str
    min_value: float = 0.0
    max_value: float = 100.0

Question = Union[Choice, Noul, Score]

@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    confidence: float
    probabilities: Dict[str, float]

@dataclass(frozen=True)
class NoulResult:
    value: bool
    probability: float

@dataclass(frozen=True)
class ScoreResult:
    score: float
    normalized: float

Answer = Union[ChoiceResult, NoulResult, ScoreResult]

@dataclass(frozen=True)
class DecisionResult:
    answers: Dict[str, Answer]
    latency_ms: float


// ============================================================================
// FILE: nanollm/tokenizer.py (26 code lines, 34 total)
// ============================================================================

from typing import List

class ByteTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.q_marker_id = 1

    @property
    def vocab_size(self) -> int:
        return 258

    def encode(self, text: str) -> List[int]:
        return [b + 2 for b in text.encode("utf-8")]

    def decode(self, token_ids: List[int]) -> str:
        valid_bytes = [b - 2 for b in token_ids if b >= 2]
        return bytes(valid_bytes).decode("utf-8", errors="ignore")

class SubwordTokenizer:
    def __init__(self, name: str = "answerdotai/ModernBERT-base"):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(name)
        self.pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else 50283
        self.q_marker_id = self._tok.sep_token_id if self._tok.sep_token_id is not None else 50282

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)


// ============================================================================
// FILE: prepare_data.py (128 code lines, 140 total)
// ============================================================================

from typing import Dict, List, Tuple
import json
import os
import random
from datasets import load_dataset

SUBJECTS = {
    0: ["credit card", "monthly invoice", "Stripe payment", "wire transfer", "annual subscription"],
    1: ["PostgreSQL replica", "Redis cache", "Kubernetes pod", "Docker container", "Kafka consumer"],
    2: ["enterprise contract", "volume discount", "annual SLA agreement", "procurement review"],
    3: ["phishing email", "brute force login", "compromised API key", "unauthorized access", "ransomware alert"]
}
PREDICATES = {
    0: [("was charged twice unexpectedly", True, (0.8, 0.95)), ("needs to be updated", False, (0.2, 0.45))],
    1: [("crashed with out of memory error", True, (0.85, 1.0)), ("scheduled maintenance window", False, (0.1, 0.3))],
    2: [("ready for legal signature", False, (0.25, 0.5)), ("immediate escalation for renewal", True, (0.7, 0.9))],
    3: [("detected from unknown IP address", True, (0.9, 1.0)), ("blocked automatically by firewall", False, (0.4, 0.65))]
}
DEPT_NAMES = ["Billing", "Infrastructure", "Enterprise Sales", "Security"]

def build_tech_samples(rng: random.Random, count: int) -> List[Dict]:
    records = []
    for _ in range(count):
        dept = rng.randint(0, 3)
        subject = rng.choice(SUBJECTS[dept])
        pred, urgent, (s_min, s_max) = rng.choice(PREDICATES[dept])
        options = list(DEPT_NAMES)
        rng.shuffle(options)
        records.append({
            "state": f"Alert regarding {subject}: {pred}.",
            "questions": [
                ["domain", "choice", options.index(DEPT_NAMES[dept]), options],
                ["is_urgent", "noul", 1.0 if urgent else 0.0],
                ["severity", "score", round(rng.uniform(s_min, s_max), 3)]
            ]
        })
    return records

def build_banking_samples(rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    ds = load_dataset("mteb/banking77")
    labels = sorted(list(set(row["label_text"] for row in ds["train"])))
    splits = {}
    for split_name in ["train", "test"]:
        records = []
        for row in ds[split_name]:
            text, label = row["text"].strip(), row["label_text"]
            if not text:
                continue
            distractors = rng.sample([l for l in labels if l != label], 4)
            options = [label] + distractors
            rng.shuffle(options)
            urgent = 1.0 if any(k in text.lower() for k in ["stolen", "lost", "fraud", "decline", "fail"]) else 0.0
            records.append({
                "state": text,
                "questions": [
                    ["intent", "choice", options.index(label), options],
                    ["is_urgent", "noul", urgent],
                    ["severity", "score", round(rng.uniform(0.7, 0.95) if urgent else rng.uniform(0.1, 0.4), 3)]
                ]
            })
        splits[split_name] = records
    return splits["train"], splits["test"]

def build_massive_samples(rng: random.Random) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    ds = load_dataset("mteb/amazon_massive_intent", "en")
    labels = sorted(list(set(row["label_text"] for row in ds["train"])))
    splits = {}
    for split_name in ["train", "validation", "test"]:
        records = []
        for row in ds[split_name]:
            text, label = row["text"].strip(), row["label_text"]
            if not text:
                continue
            distractors = rng.sample([l for l in labels if l != label], 4)
            options = [label] + distractors
            rng.shuffle(options)
            is_cmd = 1.0 if any(text.lower().startswith(w) for w in ["set", "play", "call", "turn", "open", "mute", "send", "book", "order", "wake"]) else 0.0
            records.append({
                "state": text,
                "questions": [
                    ["action", "choice", options.index(label), options],
                    ["is_command", "noul", is_cmd],
                    ["priority", "score", round(rng.uniform(0.6, 0.9) if is_cmd else rng.uniform(0.2, 0.5), 3)]
                ]
            })
        splits[split_name] = records
    return splits["train"], splits["validation"], splits["test"]

def build_sentiment_samples(rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    ds = load_dataset("mteb/tweet_sentiment_extraction")
    splits = {}
    for split_name in ["train", "test"]:
        records = []
        for row in ds[split_name]:
            text, label = row["text"].strip(), row["label_text"]
            if not text:
                continue
            options = ["positive", "neutral", "negative"]
            rng.shuffle(options)
            is_neg = 1.0 if label == "negative" else 0.0
            score = rng.uniform(0.7, 1.0) if label == "positive" else (rng.uniform(0.4, 0.6) if label == "neutral" else rng.uniform(0.0, 0.3))
            records.append({
                "state": text,
                "questions": [
                    ["sentiment", "choice", options.index(label), options],
                    ["is_negative", "noul", is_neg],
                    ["polarity", "score", round(score, 3)]
                ]
            })
        splits[split_name] = records
    return splits["train"], splits["test"]

def main():
    os.makedirs("data", exist_ok=True)
    rng = random.Random(42)

    b_tr, b_te = build_banking_samples(rng)
    m_tr, m_val, m_te = build_massive_samples(rng)
    s_tr, s_te = build_sentiment_samples(rng)
    t_tr = build_tech_samples(rng, 20000)
    t_val = build_tech_samples(rng, 2000)
    t_te = build_tech_samples(rng, 1000)

    train_all = b_tr + m_tr + s_tr + t_tr
    rng.shuffle(train_all)

    val_all = m_val + s_te[:2000] + t_val
    rng.shuffle(val_all)

    test_all = b_te + m_te + s_te[2000:4000] + t_te
    rng.shuffle(test_all)

    for path, d in [("data/train.jsonl", train_all), ("data/val.jsonl", val_all), ("data/test.jsonl", test_all)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in d:
                f.write(json.dumps(s) + "\n")
        print(f"Wrote {len(d)} records to {path}")

if __name__ == "__main__":
    main()


// ============================================================================
// FILE: train.py (88 code lines, 105 total)
// ============================================================================

from typing import Any, Dict
import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from nanollm import (
    CalibratedLoss,
    ModelConfig,
    MultiQuestionCollator,
    NanoModel,
    SubwordTokenizer,
)
from nanollm.dataset import load_jsonl

def evaluate(model: NanoModel, dataloader: DataLoader, loss_fn: CalibratedLoss, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            positions = batch["question_positions"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            types = batch["types"]

            opt_ids = batch["opt_ids"].to(device) if batch["opt_ids"] is not None else None
            opt_mask = batch["opt_mask"].to(device) if batch["opt_mask"] is not None else None

            outputs = model(input_ids, positions, mask)
            opt_vectors = model.encode_options(opt_ids, opt_mask) if opt_ids is not None else None
            loss = loss_fn(outputs, targets, types, opt_vectors=opt_vectors, opt_slices=batch["opt_slices"])
            total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Initializing on: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    backbone_name = "answerdotai/ModernBERT-base"
    tokenizer = SubwordTokenizer(backbone_name)
    collator = MultiQuestionCollator(tokenizer)

    train_samples = load_jsonl("data/train.jsonl")
    val_samples = load_jsonl("data/val.jsonl")

    train_loader = DataLoader(train_samples, batch_size=32, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_samples, batch_size=32, shuffle=False, collate_fn=collator)

    backbone = AutoModel.from_pretrained(backbone_name)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_dim=768, num_layers=22, num_heads=12, proj_dim=256)
    model = NanoModel(config, backbone=backbone).to(device)
    loss_fn = CalibratedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    epochs = 3
    best_val_loss = float("inf")
    total_steps = len(train_loader)
    print(f"[DATA] Train: {len(train_samples)} samples | Val: {len(val_samples)} samples | Batch: 32 | Steps/Epoch: {total_steps}\n", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.perf_counter()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            positions = batch["question_positions"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            types = batch["types"]

            opt_ids = batch["opt_ids"].to(device) if batch["opt_ids"] is not None else None
            opt_mask = batch["opt_mask"].to(device) if batch["opt_mask"] is not None else None

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(input_ids, positions, mask)
                opt_vectors = model.encode_options(opt_ids, opt_mask) if opt_ids is not None else None
                loss = loss_fn(outputs, targets, types, opt_vectors=opt_vectors, opt_slices=batch["opt_slices"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            if (step + 1) % 100 == 0 or (step + 1) == total_steps:
                elapsed = time.perf_counter() - start_time
                speed = (step + 1) / max(0.01, elapsed)
                eta = (total_steps - (step + 1)) / max(0.01, speed)
                pct = ((step + 1) / total_steps) * 100
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Step {step+1:04d}/{total_steps:04d} ({pct:3.0f}%) | Loss: {loss.item():.4f} | {speed:.1f} batch/s | ETA: {int(eta)}s", flush=True)

        avg_train = train_loss / total_steps
        avg_val = evaluate(model, val_loader, loss_fn, device)
        saved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "checkpoint.pt")
            saved = " --> [SAVED BEST CHECKPOINT]"
        print(f"\n>>> Epoch {epoch:02d} Complete | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}{saved}\n", flush=True)

if __name__ == "__main__":
    main()


