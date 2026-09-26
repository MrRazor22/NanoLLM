import random
from typing import List
from training.dataset.dataset import IDataSourcePolicy
from training.dataset.sources.generic import GenericChoiceSource
from training.dataset.sources.glaive import GlaiveToolSource
from training.dataset.sources.typed import TypedDecisionsSource

def get_default_training_sources(seed: int = 42) -> List[IDataSourcePolicy]:
    """Default training data composition recipe."""
    rng = random.Random(seed)
    return [
        TypedDecisionsSource(repeat=3),
        GlaiveToolSource(limit=5000, rng=rng),
        GenericChoiceSource("mteb/amazon_massive_intent", "en", "train", "text", "label_text", "action", "What is the user's intent in this utterance?", 6000),
        GenericChoiceSource("fancyzhx/ag_news", None, "train", "text", "label", "label", "What is the topic of the article?", 6000, per_class_limit=1500),
        GenericChoiceSource("SetFit/enron_spam", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nMessage: {(r.get('message') or '')[:2000]}", lambda r: "spam" if int(r.get("label", 0)) == 1 else "legitimate", "is_spam", "Is this email unsolicited spam or bulk marketing?", 3000),
        GenericChoiceSource("zefang-liu/phishing-email-dataset", None, "train", lambda r: f"Email:\n{(r.get('Email Text') or '')[:2000]}", lambda r: "phishing" if r.get("Email Type") == "Phishing Email" else "safe", "is_phishing", "Is this email a phishing or scam attempt?", 3000),
        GenericChoiceSource("Tobi-Bueck/customer-support-tickets", None, "train", lambda r: f"Subject: {r.get('subject', '')}\n\nBody: {(r.get('body') or '')[:2000]}", "queue", "queue", "Which support queue should handle this ticket?", 5000, filter_fn=lambda r: r.get("language") == "en" and r.get("body"), per_class_limit=500),
        GenericChoiceSource("dair-ai/emotion", "split", "train", "text", "label", "emotion", "Which emotion is most strongly expressed in text?", 3000),
        GenericChoiceSource("mteb/banking77", None, "train", "text", lambda r: str(r.get("label_text", "")).replace("_", " "), "intent", "What is the primary customer inquiry or banking request?", 6000),
        GenericChoiceSource("deepset/prompt-injections", None, "train", "text", lambda r: "quarantine_threat" if int(r.get("label", 0)) == 1 else "allow", "guardrail_action", "Determine the safety policy action for this user input.", 1000),
        GenericChoiceSource("jackhhao/jailbreak-classification", None, "train", "prompt", lambda r: "quarantine_threat" if str(r.get("type", "")).lower() == "jailbreak" else "allow", "guardrail_action", "Determine the safety policy action for this user input.", 1500),
    ]

__all__ = ["get_default_training_sources"]
