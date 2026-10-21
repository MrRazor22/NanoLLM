from typing import Iterable, Optional, Protocol, runtime_checkable
import torch

@runtime_checkable
class ILearner(Protocol):
    def step(self, loss: torch.Tensor, lr: Optional[float] = None, commit: bool = True) -> None: ...

class AdamWLearner(ILearner):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-5,
        accum_steps: int = 1,
    ):
        param_list = list(params)
        is_cuda = any(p.device.type == "cuda" for p in param_list)
        self.accum_steps = max(1, accum_steps)
        self.optimizer = torch.optim.AdamW(param_list, lr=lr, fused=is_cuda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    def step(self, loss: torch.Tensor, lr: Optional[float] = None, commit: bool = True) -> None:
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr

        scaled_loss = loss / self.accum_steps
        if self.scaler.is_enabled():
            scaled = self.scaler.scale(scaled_loss)
            scaled.backward()
        else:
            scaled_loss.backward()

        if commit:
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
