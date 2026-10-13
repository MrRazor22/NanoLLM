from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class StepStats:
    """Immutable domain DTO for step-level optimization metrics."""
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    loss: float
    elapsed_ms: float
    lr: float = 0.0
    val_loss: float | None = None
    batch: int = 0
    total_batches: int = 0
