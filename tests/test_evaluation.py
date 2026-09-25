import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import ModelEvaluator, ProfilingEvaluatorLayer

def test_evaluator():
    dummy_items = [
        {"category": "test", "state": "hello", "questions": {"q1": {}}, "gold": {"q1": {"label": "yes"}}},
        {"category": "test", "state": "world", "questions": {"q1": {}}, "gold": {"q1": {"label": "no"}}},
    ]
    evaluator = ModelEvaluator("dummy", lambda s, q: {"q1": "yes"}) | ProfilingEvaluatorLayer()
    rep = evaluator.evaluate(dummy_items)
    assert rep["total_questions"] == 2
    assert rep["total_correct"] == 1
    assert rep["overall_acc"] == 0.5
    assert "p50_ms" in rep

if __name__ == "__main__":
    test_evaluator()
    print("Evaluator unit test passed!")

