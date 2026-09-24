import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nanollm import (
    ByteTokenizer,
    CalibratedLoss,
    Choice,
    DecisionResolver,
    DecisionSample,
    DecisionSubstrate,
    ModelConfig,
    MultiQuestionCollator,
    Noul,
    ProfilingEvaluator,
    QuestionSpec,
    Score,
    SlotAssembler,
    split_train_val,
)

def test_tokenizer():
    tokenizer = ByteTokenizer()
    text = "Refund my card"
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text
    assert len(tokens) == len(text.encode("utf-8"))

def test_assembler_single_and_batch():
    tokenizer = ByteTokenizer()
    assembler = SlotAssembler(tokenizer)
    qs = [
        Choice("dept", ["billing", "tech"]),
        Noul("urgent"),
        Score("sev", 0.0, 10.0),
    ]
    layout = assembler.assemble_single("Server is down", qs)
    assert layout.input_ids.shape[0] == 1
    assert layout.mask.shape == layout.input_ids.shape
    assert len(layout.slots) == 3

    collator = MultiQuestionCollator(assembler)
    samples = [
        DecisionSample("Short", [QuestionSpec("q1", "noul", 1.0)]),
        DecisionSample("Much longer sentence", [QuestionSpec("q1", "noul", 0.0)]),
    ]
    batch = collator(samples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["mask"].shape
    assert len(batch["meta"]) == 2

def test_substrate_and_loss():
    config = ModelConfig(vocab_size=260, hidden_dim=64, num_layers=2, num_heads=2)
    model = DecisionSubstrate(config)
    input_ids = torch.randint(0, 260, (2, 16))
    mask = torch.ones((2, 16))
    scores = model(input_ids, mask)
    assert scores.shape == (2, 16)

    loss_fn = CalibratedLoss()
    meta = [
        [("choice", [2, 5], 0), ("noul", [8], 1.0)],
        [("choice", [3, 7], 1), ("noul", [10], 0.0)],
    ]
    loss = loss_fn(scores, meta, torch.device("cpu"))
    assert loss.item() > 0.0
    loss.backward()
    assert model.tok_emb.weight.grad is not None

def test_resolver():
    resolver = DecisionResolver()
    qs = [Choice("dept", ["billing", "tech"]), Noul("urgent")]
    slots = [[0, 1], [2]]
    scores = torch.tensor([2.0, 0.5, 1.2])
    answers = resolver.resolve(qs, slots, scores)
    assert answers["dept"].choice == "billing"
    assert answers["urgent"].value is True

def test_evaluator_and_builder():
    train, val = split_train_val([{"i": i} for i in range(20)], val_ratio=0.1)
    assert len(train) + len(val) == 20
    assert len(val) == 10  # min val is 10

    from nanollm import ModelEvaluator
    dummy_items = [
        {"category": "test", "state": "hello", "questions": {"q1": {}}, "gold": {"q1": {"label": "yes"}}},
        {"category": "test", "state": "world", "questions": {"q1": {}}, "gold": {"q1": {"label": "no"}}},
    ]
    evaluator = ProfilingEvaluator(ModelEvaluator("dummy", lambda s, q: {"q1": "yes"}))
    rep = evaluator.evaluate(dummy_items)
    assert rep["total_questions"] == 2
    assert rep["total_correct"] == 1
    assert rep["overall_acc"] == 0.5
    assert "p50_ms" in rep

if __name__ == "__main__":
    test_tokenizer()
    test_assembler_single_and_batch()
    test_substrate_and_loss()
    test_resolver()
    test_evaluator_and_builder()
    print("ALL TESTS PASSED!")

