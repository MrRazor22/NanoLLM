import random
import time
from typing import Callable, Iterator, Optional, Protocol, Sequence, runtime_checkable

from data.case import DecisionCase
from model.layouts.checkpoints import CheckpointMeta
from model.torch_model import IModel, TorchModel
from training.policies.schedule_policy import CosineSchedulePolicy, ISchedulePolicy
from training.stats import StepStats

@runtime_checkable
class ITrainer(Protocol):
    """Core Primitive: Operational boundary for model optimization and checkpoint persistence."""
    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]: ...
    def save(self, meta: Optional[CheckpointMeta] = None) -> None: ...

class EpochTrainer(ITrainer):
    """Core Primitive Implementation: Supervised training loop over domain decision cases."""

    def __init__(
        self,
        model: Optional[IModel] = None,
        batch_size: int = 16,
        schedule_policy: Optional[ISchedulePolicy] = None,
    ):
        self.model = model or TorchModel(accum_steps=1)
        self.batch_size = batch_size
        self.schedule_policy = schedule_policy or CosineSchedulePolicy()

    def save(self, meta: Optional[CheckpointMeta] = None) -> None:
        m = meta or CheckpointMeta()
        self.model.save(m.path, meta=m)

    @staticmethod
    def _estimate_length(case: DecisionCase) -> int:
        s_len = len(case.state) // 4 if isinstance(case.state, str) else 30
        q_len = sum(len(q.options) * 5 if hasattr(q, "options") and q.options else 10 for q in case.questions)
        return s_len + q_len

    def _bucketed_batches(self, items: list[DecisionCase]) -> list[list[DecisionCase]]:
        sorted_items = sorted(items, key=self._estimate_length)
        batches = [sorted_items[i : i + self.batch_size] for i in range(0, len(sorted_items), self.batch_size)]
        random.shuffle(batches)
        return batches

    def fit(
        self,
        cases: Sequence[DecisionCase],
        epochs: int = 1,
        val_cases: Optional[Sequence[DecisionCase]] = None,
    ) -> Iterator[StepStats]:
        items = list(cases)
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size if items else 0

        # Pre-encode validation cases once
        v_batches = []
        if val_cases and hasattr(self.model, "tokenizer"):
            from model.layouts.token_layout import collate_batch
            tok = getattr(self.model, "tokenizer")
            v_items = list(val_cases)
            for vi in range(0, len(v_items), self.batch_size):
                v_batches.append(collate_batch(tok, v_items[vi : vi + self.batch_size], shuffle_options=False))

        # Pre-cache tokenized layouts for training cases to avoid tokenization bottleneck in loop
        has_tok = hasattr(self.model, "tokenizer")
        tok = getattr(self.model, "tokenizer", None)
        max_state_tokens = getattr(self.model, "max_state_tokens", 384)
        encoded_cache = None
        if has_tok:
            from model.layouts.token_layout import encode_case
            encoded_cache = [encode_case(tok, c, max_state_tokens=max_state_tokens) for c in items]

        total_steps = epochs * total_batches
        base_lr = 1e-5
        if hasattr(self.model, "optimizer") and getattr(self.model, "optimizer") is not None:
            param_groups = getattr(self.model.optimizer, "param_groups", None)
            if param_groups and isinstance(param_groups, list) and len(param_groups) > 0:
                base_lr = float(param_groups[0].get("lr", 1e-5))

        for ep in range(1, epochs + 1):
            if encoded_cache is not None:
                from model.layouts.token_layout import pad_batch
                # Sort indices by length for minimal padding
                sorted_indices = sorted(range(len(items)), key=lambda idx: len(encoded_cache[idx][0].input_ids))
                batch_indices = [sorted_indices[i : i + self.batch_size] for i in range(0, len(sorted_indices), self.batch_size)]
                random.shuffle(batch_indices)
                batches = [[encoded_cache[i] for i in b_idx] for b_idx in batch_indices]
                use_pre_encoded = True
            else:
                batches = self._bucketed_batches(items)
                use_pre_encoded = False

            total_loss = 0.0

            for n, batch_data in enumerate(batches):
                global_step = (ep - 1) * total_batches + (n + 1)
                cur_lr = self.schedule_policy.get_lr(global_step, total_steps, base_lr)
                if hasattr(self.model, "optimizer") and getattr(self.model, "optimizer") is not None:
                    param_groups = getattr(self.model.optimizer, "param_groups", None)
                    if param_groups and isinstance(param_groups, list):
                        for g in param_groups:
                            g["lr"] = cur_lr

                step_start = time.perf_counter()
                if use_pre_encoded:
                    from model.layouts.token_layout import pad_batch
                    batch_obj = pad_batch(batch_data)
                    step_loss = self.model.train_step(batch_obj)
                else:
                    step_loss = self.model.train_step(batch_data)
                total_loss += step_loss
                dt = (time.perf_counter() - step_start) * 1000.0

                v_loss = None
                is_val_step = val_cases and (n + 1 == total_batches or n + 1 == (total_batches // 2))
                if is_val_step:
                    if v_batches:
                        v_total = sum(self.model.eval_step(vb) for vb in v_batches)
                        v_loss = v_total / max(1, len(v_batches))
                    else:
                        v_items = list(val_cases)
                        v_total, v_count = 0.0, 0
                        for vi in range(0, len(v_items), self.batch_size):
                            v_batch = v_items[vi : vi + self.batch_size]
                            v_total += self.model.eval_step(v_batch)
                            v_count += 1
                        v_loss = v_total / max(1, v_count)

                yield StepStats(
                    epoch=ep,
                    total_epochs=epochs,
                    step=n + 1,
                    total_steps=total_batches,
                    loss=total_loss / (n + 1),
                    elapsed_ms=dt,
                    lr=cur_lr,
                    val_loss=v_loss,
                    batch=n + 1,
                    total_batches=total_batches,
                )

    def __or__(self, layer: Callable[["ITrainer"], "ITrainer"]) -> "ITrainer":
        return layer(self)
