from typing import Any, Dict, Sequence

LAYA_SUITES = {
    "jev.ag_news": {"label": "ag_news (400)", "jev": 0.910, "laya": 0.950},
    "jev.emotion": {"label": "emotion (400)", "jev": 0.480, "laya": 0.595},
    "jev.banking77_full": {"label": "banking77 (400, 77 cls)", "jev": 0.870, "laya": 0.425},
    "app.support_triage": {"label": "support_triage (400)", "jev": None, "laya": 0.775},
    "app.email_spam": {"label": "email_spam (400)", "jev": None, "laya": 0.933},
    "app.phishing": {"label": "phishing (400)", "jev": None, "laya": 0.958},
}

def print_benchmark_table(reports: Sequence[Dict[str, Any]]) -> None:
    if not reports:
        return
    rep = reports[0]
    by_cat = rep.get("by_cat", {})

    if any(k in LAYA_SUITES for k in by_cat):
        _print_laya_table(reports)
    else:
        _print_agentic_table(reports)

def _print_laya_table(reports: Sequence[Dict[str, Any]]) -> None:
    sep = "=" * 88
    dash = "-" * 88
    print("\n" + sep)
    print(f"{'Task / Suite (400 each)':28s} | {'Jev (Pub)':11s} | {'Laya SOTA':11s} | {'NanoLLM Champ':14s} | {'Delta vs Best':13s}")
    print(dash)
    rep = reports[0]
    by_cat = rep.get("by_cat", {})

    total_best = []
    total_nano = []

    for cat_key, meta in LAYA_SUITES.items():
        if cat_key not in by_cat:
            continue
        label = meta["label"]
        jev_str = f"{meta['jev']*100:9.1f}%" if meta["jev"] is not None else "     -     "
        laya_str = f"{meta['laya']*100:9.1f}%" if meta["laya"] is not None else "     -     "
        nano_acc = by_cat[cat_key]
        nano_str = f"{nano_acc*100:12.1f}%"

        valid = [v for v in (meta["jev"], meta["laya"]) if v is not None]
        best_val = max(valid) if valid else 0.0
        delta = (nano_acc - best_val) * 100.0
        delta_str = f"{delta:+11.1f}%"

        total_best.append(best_val)
        total_nano.append(nano_acc)

        print(f"{label:28s} | {jev_str} | {laya_str} | {nano_str} | {delta_str}")

    print(dash)
    mean_nano = sum(total_nano) / max(1, len(total_nano)) * 100.0
    mean_best = sum(total_best) / max(1, len(total_best)) * 100.0
    delta_mean = mean_nano - mean_best
    print(f"{'OVERALL AVERAGE':28s} |             | {mean_best:9.1f}% | {mean_nano:12.1f}% | {delta_mean:+11.1f}%")
    tot_c = rep.get("total_correct", 0)
    tot_q = rep.get("total_questions", 0)
    print(f"{'Total Score':28s} |             |             | {tot_c:5d}/{tot_q:<6d} |")
    p50 = rep.get("p50_ms", 0.0)
    p90 = rep.get("p90_ms", 0.0)
    p50_delta = p50 - 33.0
    print(f"{'P50 Latency (CUDA)':28s} |    246.0 ms |     33.0 ms | {p50:9.1f} ms   | {p50_delta:+10.1f} ms")
    print(f"{'P90 Latency (CUDA)':28s} |             |     49.0 ms | {p90:9.1f} ms   |")
    print(sep + "\n")

def _print_agentic_table(reports: Sequence[Dict[str, Any]]) -> None:
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
    print("\n" + sep + "\n" + header + "\n" + dash)
    for c, label in cats.items():
        print(f"{label:30s} | " + " | ".join(f"{r.get('by_cat', {}).get(c, 0)*100:{col_w-1}.1f}%" for r in reports))
    print(dash)
    print(f"{'OVERALL ACCURACY':30s} | " + " | ".join(f"{r.get('overall_acc', 0)*100:{col_w-1}.1f}%" for r in reports))
    print(f"{'Total Score':30s} | " + " | ".join(f"{r.get('total_correct', 0):5d}/{r.get('total_questions', 0):3d}" + " "*(col_w-9) for r in reports))
    print(f"{'P50 Latency (CUDA)':30s} | " + " | ".join(f"{r.get('p50_ms', 0):{col_w-4}.1f} ms" for r in reports))
    print(f"{'P90 Latency (CUDA)':30s} | " + " | ".join(f"{r.get('p90_ms', 0):{col_w-4}.1f} ms" for r in reports))
    print(sep + "\n")

