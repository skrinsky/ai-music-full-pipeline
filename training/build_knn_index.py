#!/usr/bin/env python3
"""
Build a FAISS nearest-neighbor index from training hidden states for KNN-LM guidance.

For each PITCH token position in the training data, we record:
  - the transformer hidden state just before the token was predicted (key)
  - the local value index of the PITCH token that followed (value)

At generation time, generate_v2.py queries this index with the current hidden state
and interpolates the neighbor distribution with the model's own prediction.

Usage:
    python training/build_knn_index.py \
        --ckpt      runs/checkpoints/es_model.pt \
        --vocab_json runs/events/event_vocab.json \
        --train_pkl  runs/events/events_train.pkl \
        --out        runs/knn/pitch_general \
        --pitch_type PITCH_GENERAL \
        --max_seqs   5000 \
        --device     auto

Outputs:
    runs/knn/pitch_general.faiss   — FAISS flat L2 index
    runs/knn/pitch_general.npz     — companion array: next_tokens, pitch_type

Dependencies: pip install faiss-cpu
"""
import argparse
import math
import os
import pickle
import sys
import json
import random

import numpy as np
import torch
import torch.nn as nn


# ---- inline model definition (must match generate_v2.py) ----

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class FactorizedESModel(nn.Module):
    def __init__(self, pad_id, type_names, head_sizes, num_embeddings,
                 d_model, n_heads, n_layers, ff_mult, dropout):
        super().__init__()
        self.pad_id     = pad_id
        self.type_names = type_names
        self.head_sizes = head_sizes
        self.num_types  = len(type_names)

        self.tok_emb = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEmbedding(d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)

        self.type_head   = nn.Linear(d_model, self.num_types, bias=True)
        self.value_heads = nn.ModuleList([nn.Linear(d_model, s, bias=True) for s in head_sizes])

    def forward(self, x):
        B, T = x.shape
        h = self.drop(self.pos_emb(self.tok_emb(x)))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        pad  = (x == self.pad_id)
        h = self.tr(h, mask=mask, src_key_padding_mask=pad)
        return h   # (B, T, d_model) — raw hidden states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True, help="Checkpoint .pt file")
    ap.add_argument("--vocab_json", required=True, help="event_vocab.json used for training")
    ap.add_argument("--train_pkl",  required=True, help="events_train.pkl from pre.py")
    ap.add_argument("--out",        required=True, help="Output path prefix (writes <out>.faiss + <out>.npz)")
    ap.add_argument("--pitch_type", default="PITCH_GENERAL",
                    help="Which PITCH type to index (PITCH_GENERAL or PITCH_DRUMS). "
                         "Run once for each pitch type you want to use KNN on.")
    ap.add_argument("--max_seqs",  type=int, default=5000,
                    help="Max training sequences to process (subsample if larger). "
                         "5000 seqs at 512 tokens ≈ 100K PITCH positions, ~200MB index.")
    ap.add_argument("--ctx",       type=int, default=512, help="Context window (match generate_v2 --ctx)")
    ap.add_argument("--batch",     type=int, default=16,  help="Batch size for forward passes")
    ap.add_argument("--device",    default="auto")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    req = (args.device or "auto").lower()
    if req in ("auto", "best"):
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
                  "cpu")
    else:
        device = req
    print(f"Device: {device}")

    # Load vocab
    with open(args.vocab_json) as f:
        vocab = json.load(f)
    layout = vocab["layout"]
    PAD_ID = layout["PAD"]["start"]
    V      = max(spec["start"] + spec["size"] for spec in layout.values())

    if args.pitch_type not in layout:
        print(f"ERROR: pitch_type '{args.pitch_type}' not found in vocab layout.")
        print(f"  Available types: {list(layout.keys())}")
        sys.exit(1)

    pitch_spec  = layout[args.pitch_type]
    pitch_start = pitch_spec["start"]
    pitch_size  = pitch_spec["size"]
    print(f"Pitch type: {args.pitch_type}  start={pitch_start}  size={pitch_size}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg  = ckpt.get("model_config") or ckpt.get("config")
    fact = ckpt["factored_meta"]
    type_names = fact["type_names"]
    head_sizes = fact["head_sizes"]

    model = FactorizedESModel(
        pad_id=PAD_ID, type_names=type_names, head_sizes=head_sizes, num_embeddings=V,
        d_model=cfg["D_MODEL"], n_heads=cfg["N_HEADS"], n_layers=cfg["N_LAYERS"],
        ff_mult=cfg["FF_MULT"], dropout=0.0,   # no dropout at index time
    ).to(device).eval()

    state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    model.load_state_dict(state, strict=False)
    d_model = cfg["D_MODEL"]
    print(f"Model loaded: d_model={d_model}, layers={cfg['N_LAYERS']}")

    # Load training sequences
    print(f"Loading {args.train_pkl} ...")
    with open(args.train_pkl, "rb") as f:
        train_data = pickle.load(f)

    # train_data is a list of dicts with key "tokens" (or may be a list of lists)
    if isinstance(train_data, list) and len(train_data) > 0:
        if isinstance(train_data[0], dict):
            seqs = [item["tokens"] for item in train_data if "tokens" in item]
        elif isinstance(train_data[0], (list, np.ndarray)):
            seqs = [list(s) for s in train_data]
        else:
            print("ERROR: unrecognized training data format.")
            sys.exit(1)
    else:
        print("ERROR: training data is empty or unrecognized format.")
        sys.exit(1)

    print(f"Total training sequences: {len(seqs)}")
    if len(seqs) > args.max_seqs:
        random.shuffle(seqs)
        seqs = seqs[:args.max_seqs]
        print(f"Subsampled to {len(seqs)} sequences")

    # Collect hidden states at PITCH positions
    all_hidden = []   # list of np arrays shape (d_model,)
    all_next   = []   # list of int (local PITCH value index)

    BOS_ID = layout["BOS"]["start"]

    @torch.no_grad()
    def process_batch(batch_seqs):
        # batch_seqs: list of token lists
        max_len = min(args.ctx, max(len(s) for s in batch_seqs))
        padded = np.full((len(batch_seqs), max_len), PAD_ID, dtype=np.int64)
        for i, s in enumerate(batch_seqs):
            chunk = s[:max_len]
            padded[i, :len(chunk)] = chunk

        x = torch.tensor(padded, dtype=torch.long, device=device)
        h = model(x)   # (B, T, d_model)
        h_np = h.cpu().float().numpy()

        hiddens, nexts = [], []
        for i, seq in enumerate(batch_seqs):
            seq_len = min(len(seq), max_len)
            for t in range(seq_len - 1):
                next_tok = seq[t + 1]
                if pitch_start <= next_tok < pitch_start + pitch_size:
                    local_val = next_tok - pitch_start
                    hiddens.append(h_np[i, t])
                    nexts.append(local_val)
        return hiddens, nexts

    print(f"Running model forward passes (batch={args.batch}) ...")
    for batch_start in range(0, len(seqs), args.batch):
        batch = seqs[batch_start:batch_start + args.batch]
        h_batch, n_batch = process_batch(batch)
        all_hidden.extend(h_batch)
        all_next.extend(n_batch)
        done = min(batch_start + args.batch, len(seqs))
        if done % (args.batch * 10) == 0 or done == len(seqs):
            print(f"  {done}/{len(seqs)} seqs  |  {len(all_hidden)} PITCH positions so far")

    if not all_hidden:
        print("ERROR: no PITCH positions found in training data.")
        sys.exit(1)

    vectors = np.array(all_hidden, dtype=np.float32)
    next_tokens = np.array(all_next, dtype=np.int32)
    print(f"Collected {len(vectors)} vectors, shape={vectors.shape}")

    # Build FAISS index
    try:
        import faiss
    except ImportError:
        print("ERROR: faiss not installed. pip install faiss-cpu")
        sys.exit(1)

    faiss.normalize_L2(vectors)   # cosine similarity via inner product on normalized vectors
    index = faiss.IndexFlatIP(d_model)
    index.add(vectors)
    print(f"FAISS index built: {index.ntotal} vectors")

    # Write outputs
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    faiss_path = args.out + ".faiss"
    meta_path  = args.out + ".npz"
    faiss.write_index(index, faiss_path)
    np.savez(meta_path, next_tokens=next_tokens, pitch_type=np.array(args.pitch_type))
    print(f"Wrote {faiss_path}")
    print(f"Wrote {meta_path}")
    print(f"\nDone. Use with generate_v2.py:")
    print(f"  --knn_index {args.out} --knn_k 16 --knn_lambda 0.3")


if __name__ == "__main__":
    main()
