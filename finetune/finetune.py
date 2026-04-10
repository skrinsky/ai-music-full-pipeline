#!/usr/bin/env python3
"""
LoRA-finetune a pre-trained music transformer on your personal MIDI data.

Base model
----------
Default: Natooz/Multitrack-Music-Transformer
  - GPT-2 architecture trained on ~170 k MIDI files (GigaMIDI + Lakh + others)
  - Already understands phrase structure, voice leading, common progressions
  - Vocab = MMT tokenization (same as convert.py produces)

Why LoRA?
  - Your dataset is small relative to the base model's training data.
  - LoRA updates only ~1% of parameters, preserving the music knowledge while
    injecting your compositional style.  Full fine-tuning risks overwriting
    what the base model learned.
  - Adapter is tiny (~few MB); base model weights stay on disk unchanged.

Usage:
    python finetune/finetune.py \\
        --data_dir finetune/runs/my_data \\
        --out_dir  finetune/runs/adapter

    # More epochs / larger LoRA rank for a bigger dataset:
    python finetune/finetune.py \\
        --data_dir finetune/runs/my_data \\
        --out_dir  finetune/runs/adapter \\
        --epochs 15 --lora_r 32 --lr 5e-5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, get_cosine_schedule_with_warmup


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Numpy array of (N, seq_len) int32 token-ID windows."""
    def __init__(self, path: Path):
        self.data = torch.from_numpy(np.load(path).astype(np.int64))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return {"input_ids": ids, "labels": ids.clone()}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

_FALLBACK_NOTE = """
NOTE: Could not load the pre-trained model — falling back to a randomly
initialised GPT-2-medium sized for your vocab.  This is essentially
training from scratch with a larger architecture.  To get the real benefit
(music structure pre-trained in), try:
  1. Confirm the model name on HuggingFace (e.g. Natooz/Multitrack-Music-Transformer)
  2. Check your internet connection / HuggingFace token if the repo is gated
  3. Or pre-train on GigaMIDI first using the existing blues pipeline, then
     pass that checkpoint with --base_model path/to/local/checkpoint
"""


def load_base_model(model_id: str, vocab_size: int, seq_len: int):
    """Load a HuggingFace causal-LM, resizing embeddings if the vocab doesn't match."""
    try:
        print(f"Loading base model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if model.config.vocab_size != vocab_size:
            print(f"Resizing embeddings: {model.config.vocab_size} → {vocab_size}")
            model.resize_token_embeddings(vocab_size)
        return model
    except Exception as exc:
        print(f"Could not load '{model_id}': {exc}")
        print(_FALLBACK_NOTE)
        # GPT-2 medium architecture, fresh weights, sized for the MidiTok vocab
        cfg = AutoConfig.for_model(
            "gpt2",
            vocab_size=vocab_size,
            n_positions=seq_len,
            n_embd=512,
            n_layer=8,
            n_head=8,
        )
        return AutoModelForCausalLM.from_config(cfg)


def apply_lora(model, r: int, alpha: int, dropout: float):
    """Wrap model with LoRA via PEFT."""
    from peft import LoraConfig, get_peft_model, TaskType

    # target_modules for GPT-2-style (c_attn = fused Q,K,V).
    # For LLaMA/Mistral-style models change to ["q_proj", "v_proj"].
    target = ["c_attn"]
    # Probe the actual module names and fall back if needed
    named = {n for n, _ in model.named_modules()}
    if not any("c_attn" in n for n in named):
        # Likely a non-GPT-2 architecture — try common alternatives
        for candidate in [["q_proj", "v_proj"], ["query", "value"], ["q", "v"]]:
            if any(candidate[0] in n for n in named):
                target = candidate
                break

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    ) if args.device == "auto" else args.device
    print(f"Device: {device}")

    meta = json.loads((data_dir / "meta.json").read_text())
    vocab_size = meta["vocab_size"]
    seq_len    = meta["seq_len"]
    print(f"Vocab size: {vocab_size}  Seq len: {seq_len}")

    model = load_base_model(args.base_model, vocab_size, seq_len)
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    model = model.to(device)

    train_ds = WindowDataset(data_dir / "train_ids.npy")
    val_ds   = WindowDataset(data_dir / "val_ids.npy")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"Train batches: {len(train_dl)}  Val batches: {len(val_dl)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_dl) * args.epochs // args.grad_accum)
    scheduler   = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(train_dl):
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss   = model(input_ids=ids, labels=labels).loss / args.grad_accum
            loss.backward()
            train_loss += loss.item() * args.grad_accum

            if (i + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                ids    = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                val_loss += model(input_ids=ids, labels=labels).loss.item()

        avg_train = train_loss / len(train_dl)
        avg_val   = val_loss   / len(val_dl)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            model.save_pretrained(out_dir / "best")
            print(f"         → saved best adapter (val={best_val:.4f})")

    model.save_pretrained(out_dir / "final")
    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Adapter saved → {out_dir}/best/")
    print("\nNext:")
    print(f"  python finetune/generate.py \\")
    print(f"      --base_model {args.base_model} \\")
    print(f"      --adapter {out_dir}/best \\")
    print(f"      --tokenizer_config {data_dir}/tokenizer_config.json \\")
    print(f"      --out_midi finetune/runs/out.mid")


def main():
    ap = argparse.ArgumentParser(description="LoRA-finetune a pre-trained music transformer")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--base_model", default="Natooz/Multitrack-Music-Transformer",
                    help="HuggingFace model ID (must use MMT tokenization)")
    # LoRA
    ap.add_argument("--lora_r",       type=int,   default=16)
    ap.add_argument("--lora_alpha",   type=int,   default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    # Training
    ap.add_argument("--epochs",       type=int,   default=5)
    ap.add_argument("--batch_size",   type=int,   default=4)
    ap.add_argument("--grad_accum",   type=int,   default=4,
                    help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int,   default=50)
    ap.add_argument("--device",       default="auto")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
