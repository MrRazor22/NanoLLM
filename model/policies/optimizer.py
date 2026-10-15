from typing import Iterable, Optional, Protocol, Union, runtime_checkable
import torch

@runtime_checkable
class IGradientNode(Protocol):
    """Structural protocol for computational loss graph nodes supporting differentiation."""
    def backward(self) -> None: ...

@runtime_checkable
class IOptimizerPolicy(Protocol):
    """Sub-Primitive Policy: Coordinates parameter update strategy, gradient scaling, and accumulation."""
    def step(self, loss: IGradientNode, lr: Optional[float] = None) -> None: ...
    def zero_grad(self) -> None: ...

class AdamWOptimizer(IOptimizerPolicy):
    """Default optimizer policy: AdamW with AMP GradScaler and gradient accumulation."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-5,
        accum_steps: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.accum_steps = max(1, accum_steps)
        dev = torch.device(device) if isinstance(device, str) else device
        is_cuda = dev.type == "cuda" if dev is not None else torch.cuda.is_available()
        self.optimizer = torch.optim.AdamW(params, lr=lr, fused=is_cuda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
        self.opt_count = 0

    def step(self, loss: IGradientNode, lr: Optional[float] = None) -> None:
        if lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr

        loss_node = (loss / self.accum_steps) if isinstance(loss, torch.Tensor) else loss
        if self.scaler.is_enabled() and isinstance(loss_node, torch.Tensor):
            scaled = self.scaler.scale(loss_node)
            scaled.backward()
        else:
            loss_node.backward()

        self.opt_count += 1
        if self.opt_count % self.accum_steps == 0:
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
