#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dense chorale Transformer model with continuation embedding.

Single softmax over 100 tokens (PAD/BOS/EOS + 46 pitches + REST + 50 chords).
No factored heads, no auxiliary loss.

Default configuration (~685K params):
    D_MODEL=128, N_HEADS=4, N_LAYERS=4, FF_MULT=3, DROPOUT=0.15
"""

import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding (same pattern as train.py)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D) with positional encoding added."""
        return x + self.pe[:x.size(1)].unsqueeze(0)


class ContinuationEmbedding(nn.Module):
    """Embedding for per-voice held-note counters (0-31).

    Added to the token+positional embedding at each position.
    """

    def __init__(self, d_model: int, max_count: int = 32):
        super().__init__()
        self.emb = nn.Embedding(max_count, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.01)

    def forward(self, cont_ids: torch.Tensor) -> torch.Tensor:
        """cont_ids: (B, T) integers in [0, 31] → (B, T, D)."""
        return self.emb(cont_ids)


def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular boolean mask for causal attention."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class ChoraleDenseModel(nn.Module):
    """Transformer encoder for dense chorale token prediction.

    Args:
        vocab_size: total vocabulary size (default 100)
        pad_id: padding token ID (default 0)
        d_model: model dimension
        n_heads: number of attention heads
        n_layers: number of transformer layers
        ff_mult: feedforward multiplier (ff_dim = d_model * ff_mult)
        dropout: dropout rate
    """

    def __init__(self,
                 vocab_size: int = 100,
                 pad_id: int = 0,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 ff_mult: int = 3,
                 dropout: float = 0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEmbedding(d_model)
        self.cont_emb = ContinuationEmbedding(d_model)

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

        self.head = nn.Linear(d_model, vocab_size, bias=True)

        # Init
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor,
                cont: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    (B, T) token IDs
        cont: (B, T) continuation counter IDs (0-31), or None

        Returns: (B, T, vocab_size) logits
        """
        B, T = x.shape
        h = self.tok_emb(x)
        h = self.pos_emb(h)
        if cont is not None:
            h = h + self.cont_emb(cont)
        h = self.drop(h)

        attn_mask = make_causal_mask(T, x.device)
        pad_mask = (x == self.pad_id)
        h = self.tr(h, mask=attn_mask, src_key_padding_mask=pad_mask)

        return self.head(h)  # (B, T, vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
