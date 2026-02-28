#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CascadedESModel: Transformer with triple embedding (token + position + musical time).

Same factored output as FactorizedESModel (type head + per-type value heads)
but forward() takes (x, musical_times) and adds sinusoidal musical time encoding.
"""

import math
from typing import List

import torch
import torch.nn as nn


class MusicalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous musical time (in quarter notes).

    Uses a different base (default 100.0) than standard positional encoding
    to cover the typical 0–200 QN range in music pieces.
    """

    def __init__(self, d_model: int, base: float = 100.0, max_qn: float = 400.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        # Precompute div_term: shape (d_model//2,)
        half_d = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_d, dtype=torch.float32)
            * (-math.log(base) / half_d)
        )
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(self, musical_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            musical_times: (B, T) float tensor of cumulative QN times.

        Returns:
            (B, T, d_model) sinusoidal embeddings.
        """
        # musical_times: (B, T) → (B, T, 1)
        t = musical_times.unsqueeze(-1)
        # div_term: (half_d,) → (1, 1, half_d)
        div = self.div_term.unsqueeze(0).unsqueeze(0)
        angles = t * div  # (B, T, half_d)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, T, d_model)
        return emb


class PositionalEmbedding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D) with positional encoding added."""
        return x + self.pe[: x.size(1)].unsqueeze(0)


def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class CascadedESModel(nn.Module):
    """Factored Transformer with triple embedding for cascade generation.

    Embedding: token_emb(tok) + pos_emb(seq_pos) + musical_time_emb(cum_qn)

    Output: type_head → (B,T,num_types), value_heads → list of (B,T,head_size_i)
    """

    def __init__(
        self,
        pad_id: int,
        type_names: List[str],
        head_sizes: List[int],
        num_embeddings: int,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        ff_mult: int = 3,
        dropout: float = 0.12,
        musical_time_base: float = 100.0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.type_names = type_names
        self.head_sizes = head_sizes
        self.num_types = len(type_names)
        self.d_model = d_model

        self.tok_emb = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEmbedding(d_model)
        self.music_time_emb = MusicalTimeEmbedding(d_model, base=musical_time_base)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)

        self.type_head = nn.Linear(d_model, self.num_types, bias=True)
        self.value_heads = nn.ModuleList(
            [nn.Linear(d_model, s, bias=True) for s in head_sizes]
        )

        # Init
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for m in list(self.value_heads) + [self.type_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        musical_times: torch.Tensor,
    ):
        """
        Args:
            x: (B, T) global token IDs.
            musical_times: (B, T) float tensor of cumulative QN times.

        Returns:
            type_logits:  (B, T, num_types)
            value_logits: list of (B, T, head_size_i)
        """
        B, T = x.shape
        h = self.tok_emb(x)
        h = self.pos_emb(h)
        h = h + self.music_time_emb(musical_times)
        h = self.drop(h)

        attn_mask = make_causal_mask(T, x.device)
        pad_mask = x == self.pad_id
        h = self.tr(h, mask=attn_mask, src_key_padding_mask=pad_mask)

        type_logits = self.type_head(h)
        value_logits = [head(h) for head in self.value_heads]

        return type_logits, value_logits
