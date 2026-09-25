from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanollm import Choice, DecisionEngine, Noul, ProfilingLayer, Score

def main():
    engine = DecisionEngine.from_checkpoint() | ProfilingLayer
    
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
