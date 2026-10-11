from typing import List
from data.sources.data_source import IDataSource
from data.sources.glaive_source import GlaiveSource
from data.sources.decisions_source import DecisionsSource
from data.sources.generic_source import GenericSource

def standard_sources(seed: int = 42) -> List[IDataSource]:
    """Default collection of diverse multi-task, multimodal-ready decision sources."""
    return [
        GlaiveSource(limit=5000, seed=seed),
        DecisionsSource(repeat=5),
        GenericSource("mteb/banking77", "intent", "What is the primary customer request?", limit=5000, split="train", tag="unsplit"),
        GenericSource("mteb/banking77", "intent", "What is the primary customer request?", limit=400, split="test", tag="test"),
        GenericSource("fancyzhx/ag_news", "topic", "What is the topic of article?", limit=5000, split="train", text_field="text", label_field="label", tag="unsplit"),
        GenericSource("fancyzhx/ag_news", "topic", "What is the topic of article?", limit=400, split="test", text_field="text", label_field="label", tag="test"),
        GenericSource("dair-ai/emotion", "emotion", "Which emotion is most strongly expressed?", limit=5000, split="train", text_field="text", label_field="label", tag="unsplit"),
        GenericSource("dair-ai/emotion", "emotion", "Which emotion is most strongly expressed?", limit=400, split="test", text_field="text", label_field="label", tag="test"),
        GenericSource("Tobi-Bueck/customer-support-tickets", "queue", "Which support queue should handle this ticket?", limit=4000, split="train", text_field="body", label_field="queue", tag="unsplit"),
        GenericSource("zefang-liu/phishing-email-dataset", "is_phishing", "Is this email a phishing or scam attempt?", limit=4000, split="train", text_field="Email Text", label_field="Email Type", tag="unsplit"),
        GenericSource("SetFit/enron_spam", "is_spam", "Is this email unsolicited spam or bulk marketing?", limit=4000, split="train", text_field="text", label_field="label", tag="unsplit"),
        GenericSource("SetFit/enron_spam", "is_spam", "Is this email unsolicited spam or bulk marketing?", limit=400, split="test", text_field="text", label_field="label", tag="test"),
        GenericSource("lmsys/toxic-chat", "jailbreak", "Does prompt try to bypass system safety instructions?", limit=3000, split="train", text_field="user_input", label_field="jailbreaking", tag="unsplit"),
        GenericSource("lmsys/toxic-chat", "jailbreak", "Does prompt try to bypass system safety instructions?", limit=400, split="test", text_field="user_input", label_field="jailbreaking", tag="test"),
        GenericSource("SetFit/amazon_massive_intent_en-US", "intent", "What is the user intent in this utterance?", limit=5000, split="train", text_field="text", label_field="label_text", tag="unsplit"),
        GenericSource("SetFit/amazon_massive_intent_en-US", "intent", "What is the user intent in this utterance?", limit=400, split="test", text_field="text", label_field="label_text", tag="test"),
    ]

__all__ = ["standard_sources"]
