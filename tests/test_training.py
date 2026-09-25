import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nanollm.model import ModelConfig, NanoModel
from nanollm.inference import ByteTokenizer, SlotAssembler
from nanollm.training import (
    CalibratedLoss,
    DecisionSample,
    MultiQuestionCollator,
    QuestionSpec,
    split_train_val,
)

def test_collator_and_dataset():
    tokenizer = ByteTokenizer()
    assembler = SlotAssembler(tokenizer)
    collator = MultiQuestionCollator(assembler)
    samples = [
        DecisionSample("Short", [QuestionSpec("q1", "noul", 1.0)]),
        DecisionSample("Much longer sentence", [QuestionSpec("q1", "noul", 0.0)]),
    ]
    batch = collator(samples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["mask"].shape
    assert len(batch["meta"]) == 2

def test_loss():
    config = ModelConfig(vocab_size=260, hidden_dim=64, num_layers=2, num_heads=2)
    model = NanoModel(config)
    input_ids = torch.randint(0, 260, (2, 16))
    mask = torch.ones((2, 16))
    scores = model(input_ids, mask)

    loss_fn = CalibratedLoss()
    meta = [
        [("choice", [2, 5], 0), ("noul", [8], 1.0)],
        [("choice", [3, 7], 1), ("noul", [10], 0.0)],
    ]
    loss = loss_fn(scores, meta, torch.device("cpu"))
    assert loss.item() > 0.0
    loss.backward()
    assert model.tok_emb.weight.grad is not None

def test_builder_split():
    train, val = split_train_val([{"i": i} for i in range(20)], val_ratio=0.1)
    assert len(train) + len(val) == 20
    assert len(val) == 10

def test_checkpointing_pipe():
    from nanollm.training import CheckpointingLayer

    class MockTrainer:
        def add(self, layer, **kwargs):
            if isinstance(layer, type): return layer(self, **kwargs)
            if hasattr(layer, "attach"): return layer.attach(self)
            return layer(self, **kwargs)
        def __or__(self, layer): return self.add(layer)
        def train_epoch(self, loader): return 0.5
        def evaluate(self, loader): return 0.4

    trainer = MockTrainer() | CheckpointingLayer(output_path="dummy.pt")
    assert isinstance(trainer, CheckpointingLayer)
    assert trainer.inner is not None

if __name__ == "__main__":
    test_collator_and_dataset()
    test_loss()
    test_builder_split()
    test_checkpointing_pipe()
    print("Training tests passed!")
