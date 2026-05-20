from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return a boolean mask [seq_len, seq_len], True where attention is blocked."""
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


class MultiHeadAttention(nn.Module):
    """Multi-head attention with two modes.

    Modes
    -----
    standard:
        Separate q_proj, k_proj, v_proj.
    shared_kv:
        Separate q_proj and a single kv_proj reused as both K and V.
        This implements Attention(Q, KV, KV).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_mode: str = "standard",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if attention_mode not in {"standard", "shared_kv"}:
            raise ValueError("attention_mode must be 'standard' or 'shared_kv'.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_mode = attention_mode

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        if attention_mode == "standard":
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.kv_proj = None
        else:
            self.k_proj = None
            self.v_proj = None
            self.kv_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [B, H, S, Dk]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = x.shape
        if num_heads != self.num_heads or d_k != self.d_k:
            raise ValueError("Unexpected head tensor shape.")
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    @staticmethod
    def _apply_masks(
        scores: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # scores: [B, H, Q, K]
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask must be boolean: True means masked.")
            if attn_mask.dim() == 2:
                # [Q, K] -> [1, 1, Q, K]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [B, Q, K] -> [B, 1, Q, K]
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() != 4:
                raise ValueError("attn_mask must have 2, 3, or 4 dimensions.")
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must be boolean: True means PAD.")
            if key_padding_mask.dim() != 2:
                raise ValueError("key_padding_mask must have shape [batch, key_len].")
            # [B, K] -> [B, 1, 1, K]
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )
        return scores

    def forward(
        self,
        query_input: torch.Tensor,
        key_value_input: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query_input.dim() != 3 or key_value_input.dim() != 3:
            raise ValueError("Inputs must have shape [batch, seq, d_model].")

        q = self._split_heads(self.q_proj(query_input))

        if self.attention_mode == "standard":
            assert self.k_proj is not None and self.v_proj is not None
            k = self._split_heads(self.k_proj(key_value_input))
            v = self._split_heads(self.v_proj(key_value_input))
        else:
            assert self.kv_proj is not None
            kv = self._split_heads(self.kv_proj(key_value_input))
            k = kv
            v = kv

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self._apply_masks(scores, attn_mask, key_padding_mask)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)
        context = self._merge_heads(context)
        output = self.out_proj(context)

        if need_weights:
            # Return attention averaged over heads: [B, Q, K]
            return output, attn.mean(dim=1)
        return output, None
