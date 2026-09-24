import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanollm.evaluation import ModelEvaluator, ProfilingEvaluatorLayer, print_benchmark_table

def test_evaluator():
    dummy_items = [
        {"category": "test", "state": "hello", "questions": {"q1": {}}, "gold": {"q1": {"label": "yes"}}},
        {"category": "test", "state": "world", "questions": {"q1": {}}, "gold": {"q1": {"label": "no"}}},
    ]
    evaluator = ProfilingEvaluatorLayer(ModelEvaluator("dummy", lambda s, q: {"q1": "yes"}))
    rep = evaluator.evaluate(dummy_items)
    assert rep["total_questions"] == 2
    assert rep["total_correct"] == 1
    assert rep["overall_acc"] == 0.5
    assert "p50_ms" in rep

def test_reporter():
    rep = {
        "name": "TestModel",
        "overall_acc": 0.85,
        "total_correct": 17,
        "total_questions": 20,
        "by_cat": {"agent_tool_routing": 0.9},
        "p50_ms": 12.3,
        "p90_ms": 18.5,
    }
    # Should not raise exception
    print_benchmark_table([rep])

if __name__ == "__main__":
    test_evaluator()
    test_reporter()
    print("Evaluation tests passed!")
