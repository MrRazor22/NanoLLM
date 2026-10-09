from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable
from torch.utils.data import DataLoader
from piqo import Composable, DecisionSample
from harness.training_dataset import IDataset

@dataclass(frozen=True)
class CollisionRecord:
    sample_index: int
    holdout_index: int
    category: str
    normalized_snippet: str

@dataclass(frozen=True)
class IntegrityReport:
    total_checked: int
    total_holdouts: int
    collision_count: int
    collisions: Sequence[CollisionRecord]

    @property
    def is_clean(self) -> bool:
        return self.collision_count == 0

class ContaminationError(Exception):
    """Raised when training data collides with evaluation or holdout sets."""
    def __init__(self, report: IntegrityReport):
        super().__init__(
            f"Data contamination detected: {report.collision_count} / {report.total_checked} samples "
            f"collided with holdout evaluation records. Zero-contamination invariant violated."
        )
        self.report = report

@runtime_checkable
class IIntegrityGuard(Protocol):
    """Sovereign primitive contract for dataset holdout integrity verification."""
    def verify(
        self,
        samples: Sequence[DecisionSample],
        holdouts: Sequence[DecisionSample],
        strict: bool = True,
    ) -> IntegrityReport: ...

class IntegrityGuard(IIntegrityGuard):
    """Sovereign primitive: audits candidate samples against holdouts with zero-leakage enforcement."""

    def verify(
        self,
        samples: Sequence[DecisionSample],
        holdouts: Sequence[DecisionSample],
        strict: bool = True,
    ) -> IntegrityReport:
        norm_map: Dict[str, int] = {}
        for h_idx, h in enumerate(holdouts):
            key = self._normalize(h.state)
            if key and key not in norm_map:
                norm_map[key] = h_idx

        collisions: List[CollisionRecord] = []
        for s_idx, s in enumerate(samples):
            key = self._normalize(s.state)
            if key in norm_map:
                collisions.append(
                    CollisionRecord(
                        sample_index=s_idx,
                        holdout_index=norm_map[key],
                        category=s.category,
                        normalized_snippet=key[:80],
                    )
                )

        report = IntegrityReport(
            total_checked=len(samples),
            total_holdouts=len(holdouts),
            collision_count=len(collisions),
            collisions=collisions,
        )
        if strict and not report.is_clean:
            raise ContaminationError(report)
        return report

    @staticmethod
    def _normalize(state: object) -> str:
        if isinstance(state, dict):
            text = " ".join(f"{k}:{v}" for k, v in sorted(state.items()))
        else:
            text = str(state or "")
        return " ".join(text.lower().strip().split())

class IntegrityLayer(Composable, IDataset):
    """Composable decorator layer (lambda: IDataset -> IDataset) enforcing holdout non-contamination."""

    def __init__(
        self,
        holdouts: Sequence[DecisionSample],
        inner: Optional[IDataset] = None,
        guard: Optional[IIntegrityGuard] = None,
        strict: bool = True,
    ):
        self.holdouts = list(holdouts)
        self.guard = guard or IntegrityGuard()
        self.strict = strict
        self.inner: Optional[IDataset] = None
        self.report: Optional[IntegrityReport] = None
        if inner is not None:
            self.attach(inner)

    def attach(self, inner: IDataset) -> "IntegrityLayer":
        self.inner = inner
        self.report = self.guard.verify(self.inner.samples, self.holdouts, strict=self.strict)
        return self

    @property
    def samples(self) -> Sequence[DecisionSample]:
        if self.inner is None: raise RuntimeError("IntegrityLayer is not attached to an inner dataset.")
        return self.inner.samples

    def __len__(self) -> int:
        if self.inner is None: raise RuntimeError("IntegrityLayer is not attached to an inner dataset.")
        return len(self.inner)

    def __getitem__(self, idx: int) -> DecisionSample:
        if self.inner is None: raise RuntimeError("IntegrityLayer is not attached to an inner dataset.")
        return self.inner[idx]

    def get_loader(self, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
        if self.inner is None: raise RuntimeError("IntegrityLayer is not attached to an inner dataset.")
        return self.inner.get_loader(batch_size=batch_size, shuffle=shuffle)

__all__ = [
    "CollisionRecord",
    "IntegrityReport",
    "ContaminationError",
    "IIntegrityGuard",
    "IntegrityGuard",
    "IntegrityLayer",
]
