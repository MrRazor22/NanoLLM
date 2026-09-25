import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

LABELS: Dict[str, str] = {
    "agent_tool_routing": "Agent Tool Routing (100)",
    "negative_constraints": "Negative Constraints (100)",
    "safety_guardrails": "Safety & Guardrails (100)",
    "triage_incident": "Incident Triage (100)",
    "slice_missing_option": "Missing Option Abstention",
    "slice_distant_oos": "Distant Out-of-Scope",
    "agent_trace_observability": "Agent Observability",
    "customer_service": "Customer Service",
    "invoice_processing": "Invoice Processing",
    "security_incidents": "Security Incidents",
    "jev.ag_news": "AG News",
    "jev.emotion": "DAIR Emotion",
    "jev.banking77_full": "Banking77 (77 classes)",
    "app.support_triage": "Support Triage",
    "app.email_spam": "Email Spam",
    "app.phishing": "Phishing",
}

def load_competitor_cache() -> Dict[str, Any]:
    p = Path(__file__).resolve().parent / "data" / "competitor_cache.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def print_scorecard(report: Dict[str, Any], track: str = "benchmark") -> None:
    cache = load_competitor_cache().get(track, {})
    v_info = cache.get("Verdict 2.0 (151M)", {})
    l_info = cache.get("Laya (421M)", {})
    v_slices = v_info.get("slices", {}) if v_info else {}
    l_slices = l_info.get("slices", {}) if l_info else {}

    sep = "=" * 95
    dash = "-" * 95
    print("\n" + sep)
    title = f"NANOLLM HONEST BENCHMARK: {track.upper().replace('_', ' ')}"
    print(f"{title:^95}")
    print(sep)
    print(f"{'Task / Workflow Slice':30s} | {'Count':6s} | {'NanoLLM':9s} | {'Verdict 2.0':11s} | {'Laya 0.2.1':10s} | {'Delta vs Best':14s}")
    print(dash)

    by_cat = report.get("by_cat", {})
    counts = report.get("cat_counts", {})
    for cat, acc in sorted(by_cat.items()):
        label = LABELS.get(cat, cat)
        n_samples = str(counts.get(cat, "-"))
        nano_str = f"{acc * 100:5.1f}%"
        v_val = v_slices.get(cat)
        l_val = l_slices.get(cat)
        v_str = f"{v_val * 100:5.1f}%" if v_val is not None else "     -     "
        l_str = f"{l_val * 100:5.1f}%" if l_val is not None else "    -     "
        
        comps = [c for c in (v_val, l_val) if c is not None]
        if comps:
            delta = (acc - max(comps)) * 100.0
            d_str = f"{delta:+5.1f}% (WIN)" if delta > 0 else f"{delta:+5.1f}%"
        else:
            d_str = "Baseline"
        print(f"{label:30s} | {n_samples:>6s} | {nano_str:>9s} | {v_str:>11s} | {l_str:>10s} | {d_str:>14s}")

    print(dash)
    overall_acc = report.get("overall_acc", 0.0) * 100.0
    tot_c = report.get("total_correct", 0)
    tot_q = report.get("total_questions", 0)
    v_tot = v_info.get("overall") if v_info else None
    l_tot = l_info.get("overall") if l_info else None
    v_tot_str = f"{v_tot * 100:5.1f}%" if v_tot is not None else "     -     "
    l_tot_str = f"{l_tot * 100:5.1f}%" if l_tot is not None else "    -     "
    comps_tot = [c for c in (v_tot, l_tot) if c is not None]
    if comps_tot:
        d_tot = overall_acc - max(comps_tot) * 100.0
        d_tot_str = f"{d_tot:+5.1f}% (WIN)" if d_tot > 0 else f"{d_tot:+5.1f}%"
    else:
        d_tot_str = "Baseline"
    score_str = f"{tot_c}/{tot_q}"
    print(f"{'OVERALL AVERAGE':30s} | {score_str:>6s} | {overall_acc:8.1f}% | {v_tot_str:>11s} | {l_tot_str:>10s} | {d_tot_str:>14s}")

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
