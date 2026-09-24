from typing import Any, Dict, Sequence

def print_benchmark_table(reports: Sequence[Dict[str, Any]]) -> None:
    cats = {
        "agent_tool_routing": "Agent Tool Routing (30)",
        "safety_guardrails": "Safety & Guardrails (30)",
        "triage_incident": "Incident Triage (90 Qs)",
        "negative_constraints": "Negative Constraints (30)",
    }
    col_w = 16
    header = f"{'Category / Metric':30s} | " + " | ".join(f"{r.get('name', 'Model')[:col_w]:{col_w}s}" for r in reports)
    sep = "=" * len(header)
    dash = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(dash)
    for c, label in cats.items():
        row = f"{label:30s} | " + " | ".join(f"{r.get('by_cat', {}).get(c, 0)*100:{col_w-1}.1f}%" for r in reports)
        print(row)
    print(dash)
    row_acc = f"{'OVERALL ACCURACY':30s} | " + " | ".join(f"{r.get('overall_acc', 0)*100:{col_w-1}.1f}%" for r in reports)
    print(row_acc)
    row_score = f"{'Total Score':30s} | " + " | ".join(f"{r.get('total_correct', 0):5d}/{r.get('total_questions', 0):3d}" + " "*(col_w-9) for r in reports)
    print(row_score)
    row_p50 = f"{'P50 Latency (CUDA)':30s} | " + " | ".join(f"{r.get('p50_ms', 0):{col_w-4}.1f} ms" for r in reports)
    print(row_p50)
    row_p90 = f"{'P90 Latency (CUDA)':30s} | " + " | ".join(f"{r.get('p90_ms', 0):{col_w-4}.1f} ms" for r in reports)
    print(row_p90)
    print(sep + "\n")
