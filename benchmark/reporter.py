from typing import Any, Dict, Optional
import numpy as np

COMPETITOR_REFS: Dict[str, Dict[str, Any]] = {
    # Laya 6-Suite
    "jev.ag_news": {"label": "AG News", "ref_name": "Laya", "ref_val": 0.950},
    "jev.emotion": {"label": "DAIR Emotion", "ref_name": "Laya", "ref_val": 0.595},
    "jev.banking77_full": {"label": "Banking77 (77 classes)", "ref_name": "Laya", "ref_val": 0.425},
    "app.support_triage": {"label": "Support Triage", "ref_name": "Laya", "ref_val": 0.775},
    "app.email_spam": {"label": "Email Spam", "ref_name": "Laya", "ref_val": 0.933},
    "app.phishing": {"label": "Phishing", "ref_name": "Laya", "ref_val": 0.958},
    # Typed Decisions Workflows (vs Verdict 2.0 / Laya)
    "agent_trace_observability": {"label": "Agent Observability", "ref_name": "Verdict2", "ref_val": 0.732},
    "customer_service": {"label": "Customer Service", "ref_name": "Verdict2", "ref_val": 0.784},
    "invoice_processing": {"label": "Invoice Processing", "ref_name": "Verdict2", "ref_val": 0.812},
    "security_incidents": {"label": "Security Incidents", "ref_name": "Verdict2", "ref_val": 0.756},
    # Abstention Challenge Slices
    "slice_missing_option": {"label": "Missing Option Abstention", "ref_name": "Verdict2", "ref_val": 0.755},
    "slice_distant_oos": {"label": "Distant Out-of-Scope", "ref_name": "Verdict2", "ref_val": 0.980},
    # Agentic Suite (100 cases each)
    "agent_tool_routing": {"label": "Agent Tool Routing (100)", "ref_name": "Baseline", "ref_val": 0.850},
    "safety_guardrails": {"label": "Safety & Guardrails (100)", "ref_name": "Laya", "ref_val": 0.708},
    "negative_constraints": {"label": "Negative Constraints (100)", "ref_name": "Baseline", "ref_val": 0.850},
    "triage_incident": {"label": "Incident Triage (100)", "ref_name": "Laya", "ref_val": 0.502},
}

def print_scorecard(report: Dict[str, Any], track: str = "benchmark") -> None:
    sep = "=" * 84
    dash = "-" * 84
    print("\n" + sep)
    title = f"NANOLLM HONEST BENCHMARK: {track.upper().replace('_', ' ')}"
    print(f"{title:^84}")
    print(sep)
    print(f"{'Task / Workflow Slice':34s} | {'Count':6s} | {'NanoLLM':9s} | {'Competitor':13s} | {'Delta':7s}")
    print(dash)

    by_cat = report.get("by_cat", {})
    counts = report.get("cat_counts", {})
    total_accs, total_refs = [], []

    for cat, acc in sorted(by_cat.items()):
        meta = COMPETITOR_REFS.get(cat, {"label": cat, "ref_name": "Baseline", "ref_val": None})
        label = meta["label"]
        n_samples = str(counts.get(cat, "-"))
        nano_str = f"{acc*100:5.1f}%"
        ref_val = meta["ref_val"]
        if ref_val is not None:
            ref_str = f"{ref_val*100:5.1f}% ({meta['ref_name']})"
            delta = (acc - ref_val) * 100.0
            delta_str = f"{delta:+6.1f}%"
            total_refs.append(ref_val)
        else:
            ref_str = "    -        "
            delta_str = "  -   "
        total_accs.append(acc)
        print(f"{label:34s} | {n_samples:>6s} | {nano_str:>9s} | {ref_str:>13s} | {delta_str:>7s}")

    print(dash)
    overall_acc = report.get("overall_acc", 0.0) * 100.0
    tot_c = report.get("total_correct", 0)
    tot_q = report.get("total_questions", 0)
    ref_mean = float(np.mean(total_refs)) if total_refs else None
    ref_mean_str = f"{ref_mean*100:5.1f}%" if ref_mean is not None else "     -     "
    delta_mean = (overall_acc - ref_mean * 100.0) if ref_mean is not None else 0.0
    d_str = f"{delta_mean:+6.1f}%" if ref_mean is not None else "  -  "
    score_str = f"{tot_c}/{tot_q}"
    print(f"{'OVERALL AVERAGE':34s} | {score_str:>6s} | {overall_acc:8.1f}% | {ref_mean_str:>13s} | {d_str:>7s}")

    extras = []
    if "p50_ms" in report:
        extras.append(f"P50 Latency: {report['p50_ms']:.1f} ms")
    if "brier_score" in report:
        extras.append(f"Brier Loss: {report['brier_score']:.4f}")
    if "ece" in report:
        extras.append(f"ECE Calibration: {report['ece']*100:.2f}%")
    if extras:
        print(dash)
        print("  " + "  |  ".join(extras))

    if "selective_classification" in report:
        print(dash)
        print("Selective Classification (Automation vs Retained Accuracy):")
        for row in report["selective_classification"]:
            cov = row["coverage"] * 100.0
            ret = row["retained_accuracy"] * 100.0
            risk = row["selective_risk"] * 100.0
            thresh = row.get("threshold", 0.0)
            print(f"  Coverage: {cov:5.1f}% -> Retained Acc: {ret:5.2f}% | Selective Risk: {risk:5.2f}% (min conf: {thresh:.3f})")
    print(sep + "\n")

print_benchmark_table = print_scorecard
