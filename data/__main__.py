import argparse
from pathlib import Path
from data import (
    CORPUS_DIR,
    Dataset,
    BudgetSplitPolicy,
    NGramLeakagePolicy,
    DecisionsSource,
    GenericSource,
    GlaiveSource,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Piqo Dataset Corpus Builder")
    parser.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-quota", type=int, default=40000)
    parser.add_argument("--val-quota", type=int, default=2500)
    parser.add_argument("--test-quota", type=int, default=2500)
    parser.add_argument("--n-gram", type=int, default=13)
    args = parser.parse_args()

    split_policy = BudgetSplitPolicy(
        train_quota=args.train_quota,
        val_quota=args.val_quota,
        test_quota=args.test_quota,
        seed=args.seed,
    )
    leakage_policy = NGramLeakagePolicy(n_gram=args.n_gram)

    dataset = Dataset(
        split_policy=split_policy,
        leakage_policy=leakage_policy,
        corpus_dir=Path(args.corpus_dir),
        seed=args.seed,
    )

    # 1. Glaive function-calling tools
    dataset.register(GlaiveSource(limit=5000, seed=args.seed))

    # 2. Typed multi-question decisions
    dataset.register(DecisionsSource(repeat=5))

    # 3. Banking77 intent classification
    dataset.register(GenericSource("mteb/banking77", "intent", "What is the primary customer request?", limit=5000, split="train", tag="unsplit"))
    dataset.register(GenericSource("mteb/banking77", "intent", "What is the primary customer request?", limit=400, split="test", tag="test"))

    # 4. AG News topic categorization
    dataset.register(GenericSource("fancyzhx/ag_news", "topic", "What is the topic of article?", limit=5000, split="train", text_field="text", label_field="label", tag="unsplit"))
    dataset.register(GenericSource("fancyzhx/ag_news", "topic", "What is the topic of article?", limit=400, split="test", text_field="text", label_field="label", tag="test"))

    # 5. Dair Emotion detection
    dataset.register(GenericSource("dair-ai/emotion", "emotion", "Which emotion is most strongly expressed?", limit=5000, split="train", text_field="text", label_field="label", tag="unsplit"))
    dataset.register(GenericSource("dair-ai/emotion", "emotion", "Which emotion is most strongly expressed?", limit=400, split="test", text_field="text", label_field="label", tag="test"))

    # 6. Customer Support ticket queue triage
    dataset.register(GenericSource("Tobi-Bueck/customer-support-tickets", "queue", "Which support queue should handle this ticket?", limit=4000, split="train", text_field="body", label_field="queue", tag="unsplit"))

    # 7. Phishing Email detection
    dataset.register(GenericSource("zefang-liu/phishing-email-dataset", "is_phishing", "Is this email a phishing or scam attempt?", limit=4000, split="train", text_field="Email Text", label_field="Email Type", tag="unsplit"))

    # 8. Enron Spam filter
    dataset.register(GenericSource("SetFit/enron_spam", "is_spam", "Is this email unsolicited spam or bulk marketing?", limit=4000, split="train", text_field="text", label_field="label", tag="unsplit"))
    dataset.register(GenericSource("SetFit/enron_spam", "is_spam", "Is this email unsolicited spam or bulk marketing?", limit=400, split="test", text_field="text", label_field="label", tag="test"))

    # 9. Toxic-chat jailbreak & guardrails
    dataset.register(GenericSource("lmsys/toxic-chat", "jailbreak", "Does prompt try to bypass system safety instructions?", limit=3000, split="train", text_field="user_input", label_field="jailbreaking", tag="unsplit"))
    dataset.register(GenericSource("lmsys/toxic-chat", "jailbreak", "Does prompt try to bypass system safety instructions?", limit=400, split="test", text_field="user_input", label_field="jailbreaking", tag="test"))

    # 10. Massive Intent categorization
    dataset.register(GenericSource("SetFit/amazon_massive_intent_en-US", "intent", "What is the user intent in this utterance?", limit=5000, split="train", text_field="text", label_field="label_text", tag="unsplit"))
    dataset.register(GenericSource("SetFit/amazon_massive_intent_en-US", "intent", "What is the user intent in this utterance?", limit=400, split="test", text_field="text", label_field="label_text", tag="test"))

    print(f"Building dataset corpus into {args.corpus_dir}...")
    splits = dataset.prepare()
    print(f"Corpus ready: {len(splits.train)} train, {len(splits.val)} val, {len(splits.test)} test records (zero leakage verified).")

if __name__ == "__main__":
    main()
