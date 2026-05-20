from __future__ import annotations

import math
from typing import List

from torch.optim import Optimizer


class NoamScheduler:
    """Noam learning-rate schedule from the Transformer paper.

    rate(step) = scale * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        scale: float = 1.0,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive.")
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        self.step_num = 0

    def rate(self, step: int) -> float:
        step = max(step, 1)
        return (
            self.scale
            * (self.d_model ** -0.5)
            * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        )

    def step(self) -> float:
        self.step_num += 1
        lr = self.rate(self.step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
