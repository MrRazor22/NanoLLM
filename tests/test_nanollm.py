import torch
from nanollm import (
    ByteTokenizer,
    CalibratedLoss,
    DecisionSample,
    ModelConfig,
    MultiQuestionCollator,
    NanoModel,
)

def test_tokenizer():
    tokenizer = ByteTokenizer()
    text = "Refund my card"
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text
    assert len(tokens) == len(text.encode("utf-8"))

def test_collator():
    tokenizer = ByteTokenizer()
    collator = MultiQuestionCollator(tokenizer)
    samples = [
        DecisionSample(state="Short", questions=[("q1", "noul", 1.0)]),
        DecisionSample(state="Much longer sentence", questions=[("q1", "noul", 0.0)])
    ]
    batch = collator(samples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["mask"].shape
    assert batch["question_positions"].shape == (2, 1)

def test_model():
    config = ModelConfig(vocab_size=260, hidden_dim=64, num_layers=2, num_heads=2, max_choices=16)
    model = NanoModel(config)
    input_ids = torch.randint(0, 260, (2, 16))
    q_pos = torch.tensor([[4], [8]])
    logits = model(input_ids, q_pos)
    assert logits.shape == (2, 1, 16)
    loss = logits.sum()
    loss.backward()
    assert model.tok_emb.weight.grad is not None

def test_loss():
    loss_fn = CalibratedLoss()
    logits = torch.randn(2, 3, 16, requires_grad=True)
    targets = torch.tensor([[1.0, 1.0, 0.8], [0.0, 0.0, 0.2]])
    types = ["choice", "noul", "score"]
    loss = loss_fn(logits, targets, types)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
