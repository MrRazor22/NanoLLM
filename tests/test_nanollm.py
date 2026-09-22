import torch
from nanollm import (
    ByteTokenizer,
    CalibratedLoss,
    DecisionSample,
    ModelConfig,
    MultiQuestionCollator,
    NanoModel,
    QuestionSpec,
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
        DecisionSample(state="Short", questions=[QuestionSpec("q1", "noul", 1.0)]),
        DecisionSample(state="Much longer sentence", questions=[QuestionSpec("q1", "noul", 0.0)])
    ]
    batch = collator(samples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["mask"].shape
    assert batch["question_positions"].shape == (2, 1)

def test_model_and_loss():
    config = ModelConfig(vocab_size=260, hidden_dim=64, num_layers=2, num_heads=2, proj_dim=32)
    model = NanoModel(config)
    input_ids = torch.randint(0, 260, (2, 16))
    q_pos = torch.tensor([[4], [8]])
    outputs = model(input_ids, q_pos)

    assert outputs["q_choice"].shape == (2, 1, 32)
    assert outputs["noul"].shape == (2, 1)
    assert outputs["score"].shape == (2, 1)

    loss_fn = CalibratedLoss()
    targets = torch.tensor([[0.0], [0.0]])
    opt_vectors = torch.randn(4, 32)
    opt_slices = [[0, 2], [2, 4]]
    loss = loss_fn(outputs, targets, ["choice"], opt_vectors=opt_vectors, opt_slices=opt_slices)
    assert loss.item() > 0.0
    loss.backward()
    assert model.tok_emb.weight.grad is not None
