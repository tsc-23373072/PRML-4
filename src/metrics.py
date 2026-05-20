from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch


@torch.no_grad()
def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
) -> Tuple[int, int]:
    predictions = logits.argmax(dim=-1)
    valid = targets.ne(pad_id)
    correct = predictions.eq(targets) & valid
    return int(correct.sum().item()), int(valid.sum().item())


def _trim_sequence(seq: Sequence[int], eos_id: int, pad_id: int) -> List[int]:
    trimmed: List[int] = []
    for token in seq:
        token = int(token)
        if token == pad_id:
            continue
        trimmed.append(token)
        if token == eos_id:
            break
    return trimmed


@torch.no_grad()
def exact_match_count(
    generated: torch.Tensor,
    targets: torch.Tensor,
    eos_id: int,
    pad_id: int,
) -> Tuple[int, int]:
    if generated.dim() != 2 or targets.dim() != 2:
        raise ValueError("generated and targets must have shape [batch, seq].")

    matches = 0
    total = generated.size(0)
    for pred_row, target_row in zip(generated.cpu().tolist(), targets.cpu().tolist()):
        pred_trimmed = _trim_sequence(pred_row, eos_id=eos_id, pad_id=pad_id)
        target_trimmed = _trim_sequence(target_row, eos_id=eos_id, pad_id=pad_id)
        matches += int(pred_trimmed == target_trimmed)
    return matches, total
