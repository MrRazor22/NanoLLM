import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from piqo import (
    Choice,
    DecisionEngine,
    DecisionResult,
    Noul,
    ProfilingLayer,
    Question,
    Score,
)

class DummyTokenizer:
    mask_token_id = 256
    sep_token_id = 257
    pad_token_id = 258
    def encode(self, text: str):
        return list(text.encode("utf-8"))
    def decode(self, ids):
        return bytes([i for i in ids if i < 256]).decode("utf-8", errors="replace")

class DummyModel(nn.Module):
    def forward(self, input_ids, mask=None):
        return torch.zeros_like(input_ids, dtype=torch.float)

def test_decision_engine():
    tokenizer = DummyTokenizer()
    model = DummyModel()
    engine = DecisionEngine(model, tokenizer, device=torch.device("cpu"))
    qs = [
        Choice("dept", ["billing", "tech"]),
        Noul("urgent", instruction="Is urgent?"),
        Score("severity", min_value=0.0, max_value=100.0),
    ]
    res = engine.decide("Database is down", qs)
    assert "dept" in res.answers
    assert "urgent" in res.answers
    assert "severity" in res.answers
    assert res.answers["dept"].choice in ["billing", "tech"]
    assert isinstance(res.answers["urgent"].value, bool)
    assert 0.0 <= res.answers["severity"].score <= 100.0

def test_layer_pipe():
    class MockEngine:
        def decide(self, state, questions):
            return DecisionResult(answers={})
        def add(self, layer, **kwargs):
            if isinstance(layer, type): return layer(self, **kwargs)
            if hasattr(layer, "attach"): return layer.attach(self)
            return layer(self, **kwargs)
        def __or__(self, layer): return self.add(layer)

    engine = MockEngine() | ProfilingLayer
    assert isinstance(engine, ProfilingLayer)
    res = engine.decide("test", [])
    assert res.latency_ms is not None

if __name__ == "__main__":
    test_decision_engine()
    test_layer_pipe()
    print("Inference tests passed!")
