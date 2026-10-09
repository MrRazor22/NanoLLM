import time
from typing import Iterable, Iterator, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
import torch
import torch.nn as nn
from piqo.engine.questions import Composable
from piqo.engine.policies.torch_scorer import TorchScorer
from piqo.trainer.batches import StepStats, TargetSlot, TrainingBatch
from piqo.trainer.policies.calibrated_loss import CalibratedLoss, ILoss

def _unpack_batch(b: Union[TrainingBatch, dict], dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Sequence[Sequence[TargetSlot]]]:
    raw_ids = b["input_ids"] if isinstance(b, dict) else b.input_ids
    raw_mask = b["mask"] if isinstance(b, dict) else b.mask
    meta = b.get("targets", b.get("meta")) if isinstance(b, dict) else getattr(b, "targets", getattr(b, "meta", []))
    return torch.as_tensor(raw_ids, dtype=torch.long, device=dev), torch.as_tensor(raw_mask, dtype=torch.float, device=dev), meta

@runtime_checkable
class ITrainer(Protocol):
    """Core Primitive: Operational boundary for model optimization and evaluation."""
    def fit(self, train_batches: Iterable[TrainingBatch], val_batches: Optional[Iterable[TrainingBatch]] = None, epochs: int = 1) -> Iterator[StepStats]: ...
    def evaluate(self, batches: Iterable[TrainingBatch]) -> float: ...

class EpochTrainer(Composable, ITrainer):
    """Core Primitive Implementation: Supervised training loop over assembled batches."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: Optional[ILoss] = None, device: Optional[torch.device] = None, accum_steps: int = 2):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer, self.loss_fn, self.accum_steps = optimizer, loss_fn or CalibratedLoss(), accum_steps
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

    @classmethod
    def from_backbone(cls, backbone_name: str = "answerdotai/ModernBERT-base", lr: float = 1.5e-5, device: Optional[torch.device] = None, accum_steps: int = 2, init_checkpoint: Optional[str] = None, compile: bool = False) -> "EpochTrainer":
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scorer = TorchScorer.from_backbone(backbone_name, checkpoint_path=init_checkpoint, device=dev, compile=compile)
        return cls(scorer.model, torch.optim.AdamW(scorer.model.parameters(), lr=lr, fused=dev.type == "cuda"), device=dev, accum_steps=accum_steps)

    def evaluate(self, batches: Iterable[TrainingBatch]) -> float:
        self.model.eval()
        val_total, m = torch.tensor(0.0, device=self.device), 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            for m, b in enumerate(batches):
                ids, mask, meta = _unpack_batch(b, self.device)
                scores = self.model(ids, mask)
                val_total += self.loss_fn.loss_tensor(scores, meta) if hasattr(self.loss_fn, "loss_tensor") else self.loss_fn.forward(scores, meta)
        return (val_total / max(1, m + 1)).item()

    def fit(self, train_batches: Iterable[TrainingBatch], val_batches: Optional[Iterable[TrainingBatch]] = None, epochs: int = 1) -> Iterator[StepStats]:
        total_batches = len(train_batches) if hasattr(train_batches, "__len__") else 0
        total_steps = total_batches // self.accum_steps if total_batches else 0
        for ep in range(1, epochs + 1):
            self.model.train()
            total_loss, last_t, opt_step = torch.tensor(0.0, device=self.device), time.perf_counter(), 0
            self.optimizer.zero_grad(set_to_none=True)
            for n, b in enumerate(train_batches):
                ids, mask, meta = _unpack_batch(b, self.device)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    scores = self.model(ids, mask)
                    loss_node = self.loss_fn.loss_tensor(scores, meta) if hasattr(self.loss_fn, "loss_tensor") else self.loss_fn.forward(scores, meta)
                    loss = loss_node / self.accum_steps
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
                    is_last = (total_steps > 0 and opt_step >= total_steps) or (n + 1 == total_batches)
                    val_l = self.evaluate(val_batches) if (is_last and val_batches is not None) else None
                    lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                    yield StepStats(ep, epochs, opt_step, total_steps or opt_step, (total_loss / (n + 1)).item(), dt, lr, val_l, batch=n + 1, total_batches=total_batches)

__all__ = ["ITrainer", "EpochTrainer"]
