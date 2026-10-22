from typing import Iterable, Protocol, runtime_checkable
import torch
from model.device import is_cuda, CUDA

@runtime_checkable
class ILearner(Protocol):
    def learn(self, loss: torch.Tensor, lr: float) -> None: ...
    def update(self) -> None: ...

class AdamWLearner(ILearner):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        accum_steps: int = 1,
    ):
        param_list = list(params)
        cuda_enabled = is_cuda()
        self.accum_steps = max(1, accum_steps)
        self.accumulated = 0
        self.optimizer = torch.optim.AdamW(param_list, fused=cuda_enabled)
        self.scaler = torch.amp.GradScaler(CUDA, enabled=cuda_enabled)

    def learn(self, loss: torch.Tensor, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        normalized_loss = loss / self.accum_steps
        self.scaler.scale(normalized_loss).backward()

        self.accumulated += 1
        if self.accumulated >= self.accum_steps:
            self.update()

    def update(self) -> None:
        if self.accumulated == 0:
            return
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulated = 0
