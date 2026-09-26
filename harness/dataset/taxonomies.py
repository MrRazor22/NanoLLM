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
}

__all__ = [
    "AG_NEWS_TOPICS",
    "SPAM_CRITERIA",
    "PHISHING_CRITERIA",
    "SUPPORT_QUEUES",
    "EMOTION_CRITERIA",
    "SAFETY_GUARDRAIL_CRITERIA",
]
