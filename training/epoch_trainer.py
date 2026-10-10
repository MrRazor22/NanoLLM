from pathlib import Path
import random
import time
from typing import Iterator, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch

from inference.case import DecisionCase
from inference.questions import Composable, IDecisionEngine
from training.batches import StepStats, TargetSlot, TrainingBatch
from training.policies.calibrated_loss import CalibratedLoss, ILoss

def _unpack_batch(b: TrainingBatch, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Sequence[Sequence[TargetSlot]]]:
    raw_ids, raw_mask, targets = b.input_ids, b.mask, [t.slots for t in b.targets]
    return torch.as_tensor(raw_ids, dtype=torch.long, device=dev), torch.as_tensor(raw_mask, dtype=torch.float, device=dev), targets

@runtime_checkable
class ITrainer(Protocol):
    """Core Primitive: Operational boundary for model optimization and evaluation."""
    def fit(self, cases: Sequence[DecisionCase], epochs: int = 1) -> Iterator[StepStats]: ...
    def evaluate(self, cases: Sequence[DecisionCase]) -> float: ...

class EpochTrainer(Composable, ITrainer):
    """Core Primitive Implementation: Supervised training loop over domain decision cases."""

    def __init__(
        self,
        engine: Optional[IDecisionEngine] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[ILoss] = None,
        lr: float = 1.5e-5,
        accum_steps: int = 2,
        batch_size: int = 16,
    ):
        if engine is None:
            from inference.decision_engine import DecisionEngine
            engine = DecisionEngine()
        self.engine = engine
        self.model = engine.model
        self.encoder = engine.encoder
        self.device = engine.device
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=lr, fused=self.device.type == "cuda")
        self.loss_fn = loss_fn or CalibratedLoss()
        self.accum_steps = accum_steps
        self.batch_size = batch_size
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

    @classmethod
    def from_backbone(
        cls,
        backbone_name: str = "answerdotai/ModernBERT-base",
        lr: float = 1.5e-5,
        engine: Optional[IDecisionEngine] = None,
        loss_fn: Optional[ILoss] = None,
        accum_steps: int = 2,
        batch_size: int = 16,
        init_checkpoint: Optional[Union[str, Path]] = None,
        compile: bool = False,
    ) -> "EpochTrainer":
        if engine is None:
            from inference.decision_engine import DecisionEngine
            engine = DecisionEngine.from_checkpoint(checkpoint_path=init_checkpoint, backbone_name=backbone_name)
        return cls(engine=engine, loss_fn=loss_fn, lr=lr, accum_steps=accum_steps, batch_size=batch_size)

    def _collate(self, cases: Sequence[DecisionCase], shuffle: bool = False) -> List[TrainingBatch]:
        items = list(cases)
        if shuffle:
            random.shuffle(items)
        return [self.encoder.encode_batch(items[i : i + self.batch_size]) for i in range(0, len(items), self.batch_size)]

    def evaluate(self, cases: Sequence[DecisionCase]) -> float:
        self.model.eval()
        batches = self._collate(cases, shuffle=False)
        val_total, m = torch.tensor(0.0, device=self.device), 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            for m, b in enumerate(batches):
                ids, mask, meta = _unpack_batch(b, self.device)
                val_total += self.loss_fn(self.model(ids, mask), meta)
        return (val_total / max(1, m + 1)).item()

    def fit(self, cases: Sequence[DecisionCase], epochs: int = 1) -> Iterator[StepStats]:
        batches = self._collate(cases, shuffle=True)
        total_batches = len(batches)
        total_steps = total_batches // self.accum_steps if total_batches else 0

        for ep in range(1, epochs + 1):
            self.model.train()
            total_loss, last_t, opt_step = torch.tensor(0.0, device=self.device), time.perf_counter(), 0
            self.optimizer.zero_grad(set_to_none=True)
            for n, b in enumerate(batches):
                ids, mask, meta = _unpack_batch(b, self.device)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    loss = self.loss_fn(self.model(ids, mask), meta) / self.accum_steps
                self.scaler.scale(loss).backward()
                total_loss += loss.detach() * self.accum_steps
                if (n + 1) % self.accum_steps == 0 or (n + 1 == total_batches):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    opt_step += 1
                    now = time.perf_counter()
                    dt = (now - last_t) * 1000.0 / self.accum_steps if (n + 1) % self.accum_steps == 0 else 0.0
                    last_t = now
                    lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                    yield StepStats(ep, epochs, opt_step, total_steps or opt_step, (total_loss / (n + 1)).item(), dt, lr, None, batch=n + 1, total_batches=total_batches)

__all__ = ["ITrainer", "EpochTrainer"]
