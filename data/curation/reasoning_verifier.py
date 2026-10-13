import re
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable
from data.case import DecisionCase
from inference.questions import Choice

@runtime_checkable
class IVerifierPolicy(Protocol):
    """Sovereign policy contract: audits synthesized reasoning cases for structural integrity and logical validity."""
    @property
    def signature(self) -> str: ...
    def verify_case(self, case: DecisionCase) -> Tuple[bool, str]: ...
    def filter_cases(self, cases: Sequence[DecisionCase]) -> Tuple[Sequence[DecisionCase], dict[str, int]]: ...

CONDITION_MARKERS = {
    "not", "no", "never", "none", "neither", "nor", "low", "high",
    "flat", "zero", "empty", "full", "unlocked", "locked", "failed",
    "without", "except", "but", "although", "while", "despite",
    "instead", "stable", "normal", "abnormal", "spike", "drop",
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "and", "or", "as", "it",
}

def extract_content_words(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}

class ReasoningVerifier(IVerifierPolicy):
    """Rigorous structural and semantic audit policy for universal reasoning cases."""

    def __init__(
        self,
        max_length_ratio: float = 2.5,
        min_options: int = 3,
        require_false_friend: bool = True,
        require_condition_marker: bool = True,
    ):
        self.max_length_ratio = max_length_ratio
        self.min_options = min_options
        self.require_false_friend = require_false_friend
        self.require_condition_marker = require_condition_marker

    @property
    def signature(self) -> str:
        import hashlib, json
        payload = {
            "type": "ReasoningVerifier",
            "max_length_ratio": self.max_length_ratio,
            "min_options": self.min_options,
            "require_false_friend": self.require_false_friend,
            "require_condition_marker": self.require_condition_marker,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]

    def verify_case(self, case: DecisionCase) -> Tuple[bool, str]:
        state_str = str(case.state).strip()
        if not state_str:
            return False, "empty_state"

        choice_q = next((q for q in case.questions if isinstance(q, Choice)), None)
        if choice_q is None:
            return False, "no_choice_question"

        opts = list(choice_q.options.keys()) if isinstance(choice_q.options, dict) else list(choice_q.options)
        if len(opts) < self.min_options:
            return False, f"too_few_options_{len(opts)}"

        gold = choice_q.target
        gold_str = opts[gold] if isinstance(gold, int) and gold < len(opts) else (str(gold) if gold in opts else None)
        if gold_str is None:
            return False, "invalid_or_missing_gold"

        # 1. Conditionality Invariant: Context must express explicit condition, negation, or modifier
        state_words = extract_content_words(state_str)
        if self.require_condition_marker:
            has_marker = any(m in state_str.lower().split() or m in state_words for m in CONDITION_MARKERS)
            if not has_marker:
                return False, "missing_condition_marker"

        # 2. Length Bias Invariant: Gold length must not be an outlier relative to distractors
        gold_len = len(gold_str)
        distractor_lens = [len(o) for o in opts if o != gold_str]
        avg_dist_len = sum(distractor_lens) / max(1, len(distractor_lens))
        ratio = max(gold_len / max(1, avg_dist_len), avg_dist_len / max(1, gold_len))
        if ratio > self.max_length_ratio:
            return False, f"length_bias_ratio_{ratio:.2f}"

        # 3. Lexical False-Friend Invariant: At least one distractor MUST share words with the state
        # (guarantees that this case forces the model to evaluate predicates, not do trivial word exclusion)
        if self.require_false_friend:
            distractors = [o for o in opts if o != gold_str]
            has_false_friend = False
            for d in distractors:
                d_words = extract_content_words(d)
                overlap = state_words.intersection(d_words)
                if overlap:
                    has_false_friend = True
                    break
            if not has_false_friend:
                return False, "no_lexical_false_friend"

        return True, "valid"

    def filter_cases(self, cases: Sequence[DecisionCase]) -> Tuple[Sequence[DecisionCase], dict[str, int]]:
        valid = []
        stats = {}
        for c in cases:
            ok, reason = self.verify_case(c)
            stats[reason] = stats.get(reason, 0) + 1
            if ok:
                valid.append(c)
        return valid, stats
