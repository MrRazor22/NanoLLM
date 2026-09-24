import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nanollm.model import ModelConfig, NanoModel
from nanollm.inference import (
    ByteTokenizer,
    Choice,
    DecisionResolver,
    Noul,
    Score,
    SlotAssembler,
)

def test_tokenizer():
    tokenizer = ByteTokenizer()
    text = "Refund my card"
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text
    assert len(tokens) == len(text.encode("utf-8"))

def test_assembler():
    tokenizer = ByteTokenizer()
    assembler = SlotAssembler(tokenizer)
    qs = [Choice("dept", ["billing", "tech"]), Noul("urgent"), Score("sev", 0.0, 10.0)]
    layout = assembler.assemble_single("Server is down", qs)
    assert layout.input_ids.shape[0] == 1
    assert layout.mask.shape == layout.input_ids.shape
    assert len(layout.slots) == 3

def test_model():
    config = ModelConfig(vocab_size=260, hidden_dim=64, num_layers=2, num_heads=2)
    model = NanoModel(config)
    input_ids = torch.randint(0, 260, (2, 16))
    mask = torch.ones((2, 16))
    scores = model(input_ids, mask)
    assert scores.shape == (2, 16)

def test_resolver():
    resolver = DecisionResolver()
    qs = [Choice("dept", ["billing", "tech"]), Noul("urgent")]
    slots = [[0, 1], [2]]
    scores = torch.tensor([2.0, 0.5, 1.2])
    answers = resolver.resolve(qs, slots, scores)
    assert answers["dept"].choice == "billing"
    assert answers["urgent"].value is True

if __name__ == "__main__":
    test_tokenizer()
    test_assembler()
    test_model()
    test_resolver()
    print("Engine tests passed!")
