"""Domain taxonomies and candidate option criteria for schema parity with evaluation suites."""

AG_NEWS_TOPICS = {
    "world": "world news and international politics",
    "sports": "sports",
    "business": "business and economy",
    "sci_tech": "science and technology",
}

SPAM_CRITERIA = {
    "false": "legitimate email communication",
    "true": "unsolicited spam or promotional email",
}

PHISHING_CRITERIA = {
    "false": "legitimate safe email",
    "true": "phishing, scam, or fraudulent email",
}

SUPPORT_QUEUES = {
    "Technical Support": "technical problems, bugs, outages, integrations",
    "Product Support": "help using a product or feature",
    "Customer Service": "general account or service questions",
    "IT Support": "internal IT, devices, access, networks",
    "Billing and Payments": "invoices, charges, refunds, payment methods",
    "Returns and Exchanges": "returning or exchanging an item",
    "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
    "Sales and Pre-Sales": "pricing, quotes, buying",
    "Human Resources": "employment, payroll, leave, hiring",
    "General Inquiry": "anything else",
}

EMOTION_CRITERIA = {
    "sadness": "sadness, sorrow, grief",
    "joy": "joy, happiness, delight",
    "love": "love, affection, warmth",
    "anger": "anger, annoyance, frustration",
    "fear": "fear, anxiety, terror",
    "surprise": "surprise, shock, astonishment",
}

SAFETY_GUARDRAIL_CRITERIA = {
    "allow": "Safe, legitimate technical or business query; proceed with normal execution.",
    "quarantine_threat": "Malicious prompt injection, jailbreak attempt, or policy escape; halt and alert security.",
    "refuse_benign_policy": "Out of policy scope or requesting forbidden operations, but non-adversarial; politely decline.",
    "sanitize_execute": "Contains minor formatting noise or unintentional delimiters; sanitize and process safely.",
}

NEGATIVE_CONSTRAINTS_CRITERIA = {
    "allow": "Action strictly complies with all negative constraints; safe to proceed automatically.",
    "reject": "Action directly violates an explicit negative constraint; must be rejected immediately.",
    "require_approval": "Action touches a sensitive boundary requiring explicit human administrator sign-off.",
}

INCIDENT_TRIAGE_CRITERIA = {
    "p0_critical": "Catastrophic production outage, data loss risk, or complete service unavailability; page on-call immediately.",
    "p1_high": "Significant degradation of core features or severe error rate elevation; investigate urgently within 15 minutes.",
    "p2_medium": "Moderate performance impairment, non-critical service failure, or redundant system degradation.",
    "p3_low": "Minor cosmetic defect, non-blocking warning, or informational system alert with zero customer impact.",
}

from harness.dataset.taxonomies_b77 import BANKING77_CRITERIA

__all__ = [
    "AG_NEWS_TOPICS",
    "SPAM_CRITERIA",
    "PHISHING_CRITERIA",
    "SUPPORT_QUEUES",
    "EMOTION_CRITERIA",
    "SAFETY_GUARDRAIL_CRITERIA",
    "NEGATIVE_CONSTRAINTS_CRITERIA",
    "INCIDENT_TRIAGE_CRITERIA",
    "BANKING77_CRITERIA",
]
