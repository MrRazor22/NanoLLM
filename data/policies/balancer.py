from collections import Counter, defaultdict
import logging
import random
from typing import Optional, Protocol, Sequence, runtime_checkable
import xxhash

from data.case import DecisionCase

logger = logging.getLogger(__name__)

@runtime_checkable
class ICaseBalancer(Protocol):
    def prepare(self, cases: Sequence[DecisionCase]) -> None: ...
    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]: ...

class RoundRobinCaseBalancer(ICaseBalancer):
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.known_slots: list[tuple] = []
        self.source_choice_counts: Counter[tuple[str, str]] = Counter()

    def prepare(self, cases: Sequence[DecisionCase]) -> None:
        self.rng = random.Random(self.seed)
        self.source_choice_counts = Counter((c.meta.source, c.choice()) for c in cases if c.choice())
        slots_seen: set[tuple] = set()
        for c in cases:
            slots_seen.add(self._slot(c))
        self.known_slots = sorted(slots_seen, key=lambda s: str(s))
        self.rng.shuffle(self.known_slots)
        logger.info(f"Prepared balancer with {len(self.known_slots)} unique buckets across {len(cases)} cases.")

    def _slot(self, c: DecisionCase) -> tuple:
        choice = c.choice()
        source = c.meta.source
        candidate_type = c.meta.candidate_type
        choice_signature = choice if self.source_choice_counts.get((source, choice), 0) >= 2 else ""
        instruction_signature = xxhash.xxh64_intdigest(c.instruction.encode()) if c.instruction else 0
        return (
            source,
            candidate_type,
            instruction_signature,
            len(c.candidates),
            c.target_index(),
            choice_signature,
        )

    def balance(self, cases: Sequence[DecisionCase]) -> Sequence[DecisionCase]:
        if not cases:
            return ()

        buckets = defaultdict(list)
        for c in cases:
            buckets[self._slot(c)].append(c)

        for slot_items in buckets.values():
            self.rng.shuffle(slot_items)

        active_slots = [s for s in self.known_slots if s in buckets and buckets[s]]
        if not active_slots:
            active_slots = list(buckets.keys())

        selected: list[DecisionCase] = []
        exhausted_count = 0
        num_slots = len(active_slots)

        while exhausted_count < num_slots:
            exhausted_count = 0
            for slot in active_slots:
                items = buckets[slot]
                if items:
                    selected.append(items.pop())
                else:
                    exhausted_count += 1

        return tuple(selected)
