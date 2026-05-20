from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from .attention import MultiHeadAttention, make_causal_mask
from .positional_encoding import SinusoidalPositionalEncoding


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        attention_mode: str,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_mode=attention_mode,
        )
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor],
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, weights = self.self_attn(
            query_input=x,
            key_value_input=x,
            key_padding_mask=src_padding_mask,
            need_weights=need_weights,
        )
        x = self.norm_attn(x + self.dropout_attn(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm_ffn(x + self.dropout_ffn(ffn_out))
        return x, weights


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        attention_mode: str,
    ) -> None:
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_mode=attention_mode,
        )
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_mode=attention_mode,
        )
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor],
        src_padding_mask: Optional[torch.Tensor],
        causal_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self_out, self_weights = self.masked_self_attn(
            query_input=x,
            key_value_input=x,
            attn_mask=causal_mask,
            key_padding_mask=tgt_padding_mask,
            need_weights=need_weights,
        )
        x = self.norm_self(x + self.dropout_self(self_out))

        cross_out, cross_weights = self.cross_attn(
            query_input=x,
            key_value_input=memory,
            key_padding_mask=src_padding_mask,
            need_weights=need_weights,
        )
        x = self.norm_cross(x + self.dropout_cross(cross_out))

        ffn_out = self.ffn(x)
        x = self.norm_ffn(x + self.dropout_ffn(ffn_out))
        return x, self_weights, cross_weights


class TransformerSeq2Seq(nn.Module):
    """Educational Transformer encoder-decoder reproduction.

    It uses batch-first tensors and supports:
    - standard attention: separate Q/K/V projections
    - shared_kv attention: K and V share the same projection
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
        attention_mode: str = "standard",
        share_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if attention_mode not in {"standard", "shared_kv"}:
            raise ValueError("attention_mode must be 'standard' or 'shared_kv'.")

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_model = d_model
        self.attention_mode = attention_mode
        self.share_embeddings = share_embeddings

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        if share_embeddings:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        self.src_positional = SinusoidalPositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
        )
        self.tgt_positional = SinusoidalPositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attention_mode=attention_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attention_mode=attention_mode,
                )
                for _ in range(num_layers)
            ]
        )

        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        if share_embeddings:
            self.generator.weight = self.tgt_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(
        self,
        src_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if src_padding_mask is None:
            src_padding_mask = src_ids.eq(self.pad_id)

        x = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        x = self.src_positional(x)

        encoder_weights: List[torch.Tensor] = []
        for layer in self.encoder_layers:
            x, weights = layer(x, src_padding_mask=src_padding_mask, need_weights=need_weights)
            if weights is not None:
                encoder_weights.append(weights)
        return x, encoder_weights

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        if tgt_padding_mask is None:
            tgt_padding_mask = decoder_input_ids.eq(self.pad_id)

        x = self.tgt_embedding(decoder_input_ids) * math.sqrt(self.d_model)
        x = self.tgt_positional(x)

        causal_mask = make_causal_mask(decoder_input_ids.size(1), device=decoder_input_ids.device)

        decoder_self_weights: List[torch.Tensor] = []
        decoder_cross_weights: List[torch.Tensor] = []

        for layer in self.decoder_layers:
            x, self_weights, cross_weights = layer(
                x=x,
                memory=memory,
                tgt_padding_mask=tgt_padding_mask,
                src_padding_mask=src_padding_mask,
                causal_mask=causal_mask,
                need_weights=need_weights,
            )
            if self_weights is not None:
                decoder_self_weights.append(self_weights)
            if cross_weights is not None:
                decoder_cross_weights.append(cross_weights)
        return x, decoder_self_weights, decoder_cross_weights

    def forward(
        self,
        src_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        memory, encoder_weights = self.encode(
            src_ids=src_ids,
            src_padding_mask=src_padding_mask,
            need_weights=need_weights,
        )
        decoder_output, decoder_self_weights, decoder_cross_weights = self.decode(
            decoder_input_ids=decoder_input_ids,
            memory=memory,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            need_weights=need_weights,
        )
        logits = self.generator(decoder_output)

        if need_weights:
            return logits, {
                "encoder_self": encoder_weights,
                "decoder_self": decoder_self_weights,
                "decoder_cross": decoder_cross_weights,
            }
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 8,
    ) -> torch.Tensor:
        self.eval()
        if src_padding_mask is None:
            src_padding_mask = src_ids.eq(self.pad_id)

        memory, _ = self.encode(src_ids=src_ids, src_padding_mask=src_padding_mask)
        batch_size = src_ids.size(0)
        generated = torch.full(
            (batch_size, 1),
            self.bos_id,
            dtype=torch.long,
            device=src_ids.device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=src_ids.device)

        for _ in range(max_new_tokens):
            decoder_output, _, _ = self.decode(
                decoder_input_ids=generated,
                memory=memory,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=generated.eq(self.pad_id),
            )
            logits = self.generator(decoder_output[:, -1:, :])
            next_token = logits.argmax(dim=-1)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | next_token.squeeze(1).eq(self.eos_id)
            if bool(finished.all()):
                break

        return generated[:, 1:]  # Remove BOS.
