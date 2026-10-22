from typing import Iterable, Optional, Protocol, runtime_checkable
import torch
from model.device import is_cuda, CUDA

@runtime_checkable
class ILearner(Protocol):
    def step(self, loss: torch.Tensor, lr: Optional[float] = None) -> None: ...
    def commit(self) -> None: ...

class AdamWLearner(ILearner):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-5,
        accum_steps: int = 1,
    ):
        param_list = list(params)
        cuda_enabled = is_cuda()
        self.accum_steps = max(1, accum_steps)
        self.accumulated = 0
        self.optimizer = torch.optim.AdamW(param_list, lr=lr, fused=cuda_enabled)
        self.scaler = torch.amp.GradScaler(CUDA, enabled=cuda_enabled)

    def step(self, loss: torch.Tensor, lr: Optional[float] = None) -> None:
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr

        scaled_loss = loss / self.accum_steps
        if self.scaler.is_enabled():
            scaled = self.scaler.scale(scaled_loss)
            scaled.backward()
        else:
            scaled_loss.backward()

        self.accumulated += 1
        if self.accumulated >= self.accum_steps:
            self.commit()

    def commit(self) -> None:
        if self.accumulated == 0:
            return
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulated = 0
