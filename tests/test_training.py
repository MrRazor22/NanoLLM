import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from piqo import CalibratedLoss, CheckpointingLayer, DecisionSample, EpochTrainer, StepStats
from harness import ILeakagePolicy, NGramLeakagePolicy, TrainingDataset, split_train_val

def test_loss():
    loss_fn = CalibratedLoss()
    scores = torch.randn(2, 16)
    meta = [
        [("choice", [2, 5], 0), ("noul", [8], 1.0)],
        [("choice", [3, 7], 1), ("noul", [10], 0.0)],
    ]
    loss = loss_fn(scores, meta, torch.device("cpu"))
    assert loss.item() > 0.0

def test_builder_split():
    train, val = split_train_val([{"i": i} for i in range(20)], val_ratio=0.1)
    assert len(train) + len(val) == 20
    assert len(val) == 10

def test_checkpointing_pipe():
    class MockTrainer:
        def __init__(self):
            self.model = nn.Linear(10, 2)
        def add(self, layer, **kwargs):
            if isinstance(layer, type): return layer(self, **kwargs)
            if hasattr(layer, "attach"): return layer.attach(self)
            return layer(self, **kwargs)
        def __or__(self, layer): return self.add(layer)
        def fit(self, train_batches, val_batches=None, epochs=1):
            yield StepStats(1, 1, 1, 1, 0.5, 10.0, val_loss=0.3)
        def evaluate(self, loader): return 0.4

    trainer = MockTrainer() | CheckpointingLayer(output_path="dummy.pt")
    assert isinstance(trainer, CheckpointingLayer)
    assert trainer.inner is not None
    stats = list(trainer.fit([]))
    assert len(stats) == 1
    if os.path.exists("dummy.pt"):
        os.remove("dummy.pt")

def test_ngram_leakage_policy():
    policy: ILeakagePolicy = NGramLeakagePolicy(n_gram=5)
    train_samples = [
        DecisionSample(state="User wants to transfer money urgently to another checking account", questions=[]),
        DecisionSample(state="Cancel my premium cloud subscription immediately", questions=[]),
    ]
    holdouts = [
        DecisionSample(state="Please transfer money urgently to another checking account right now", questions=[]),
    ]
    leaks = policy.find_leaks(train_samples, holdouts)
    assert len(leaks) == 1
    assert leaks[0].state == train_samples[0].state

    clean_holdouts = [
        DecisionSample(state="What is the weather forecast for tomorrow morning?", questions=[]),
    ]
    assert len(policy.find_leaks(train_samples, clean_holdouts)) == 0

if __name__ == "__main__":
    test_loss()
    test_builder_split()
    test_checkpointing_pipe()
    test_ngram_leakage_policy()
    print("Training tests passed!")
