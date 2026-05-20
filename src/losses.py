from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing and PAD masking."""

    def __init__(self, smoothing: float, ignore_index: int) -> None:
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must satisfy 0 <= smoothing < 1.")
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError("logits must have shape [batch, seq, vocab].")
        if target.dim() != 2:
            raise ValueError("target must have shape [batch, seq].")

        vocab_size = logits.size(-1)
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        valid = target_flat.ne(self.ignore_index)
        if not bool(valid.any()):
            return logits_flat.sum() * 0.0

        logits_valid = logits_flat[valid]
        target_valid = target_flat[valid]

        log_probs = F.log_softmax(logits_valid, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target_valid.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()
