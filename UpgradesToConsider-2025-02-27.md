# Chorale Pipeline Upgrades to Close the Gap with TonicNet

**Date:** 2025-02-27
**Context:** Our chorale-cascade-train results are improved but not great. TonicNet (LSTM on the same JSB Chorales dataset) produces better results. This document analyzes why and proposes upgrades.

## Root Cause Analysis

### 1. Information Density (biggest factor)

**TonicNet**: 1 token per voice per 16th note. A full chorale (~120 timesteps) = ~600 tokens. Fits in one sequence.

**Our pipeline**: 5-6 tokens per note (TIME_SHIFT + INST + VEL + PITCH + DUR). A 512-token window captures only a fragment of a chorale. The model never sees full phrases, cadences, or harmonic arcs.

### 2. Data Augmentation: 13x vs 1x

**TonicNet**: Transposes each chorale by -6 to +6 semitones = 13 copies per chorale, yielding ~4000 training examples from 305 chorales.

**Our pipeline**: Normalizes all keys to C major/A minor and then *disables augmentation*. Only 305 training examples.

### 3. Explicit Harmony

**TonicNet**: A Music21-analyzed chord token (50 chord classes) at every timestep provides direct harmonic supervision. The model knows the current harmony *before* predicting any voice.

**Our cascade pipeline**: Heuristic chord labels injected only at beat boundaries, not at every timestep.

### 4. Wasted Capacity on Uniform Dimensions

All chorale notes have velocity=80 and known durations on a 16th-note grid, but our pipeline spends tokens on VEL (8 bins) and DUR (17 bins) that carry zero information for this dataset.

### 5. Continuation Tracking

**TonicNet**: Continuation embedding (32-dim) tells the model how long each voice has been holding its current note. This is critical for sustaining notes across multiple timesteps.

**Our pipeline**: No equivalent. Duration is a one-time token at note onset, but mid-hold context is lost.

## Side-by-Side Comparison

| Aspect | TonicNet | Our Pipeline |
|--------|----------|--------------|
| Model | 3-layer GRU, 256 hidden | 4-layer Transformer, D=192, 6 heads |
| Vocab size | 98 tokens | ~2400 tokens |
| Tokens per note | 1 (pitch only, on fixed grid) | 5-6 (TIME+INST+VEL+PITCH+DUR) |
| Sequence covers | Full chorale (~600 tokens) | Fragment (512-token window) |
| Training examples | ~4000 (305 chorales x 13 transpositions) | ~305 (no augmentation) |
| Chord modeling | Music21-analyzed, every timestep | Heuristic, beat boundaries only |
| Continuation | 32-dim embedding per voice | None |
| Output heads | Single softmax over 98 tokens | Factored: type head + per-type value heads |
| Optimizer | SGD + OneCycleLR, 60 epochs | AdamW, 200 epochs |
| Voice structure | Fixed interleave [chord, S, B, A, T] per timestep | Event stream sorted by onset time |

## Proposed Upgrades

### Tier 1 — High Impact

| # | Change | Expected Impact | Effort |
|---|--------|----------------|--------|
| 1 | **Dense chorale tokenization** — fixed 16th-note grid, 4 voices interleaved per timestep, ~50-token vocab | Biggest single win: full chorales in one sequence, 5-10x fewer tokens | Medium |
| 2 | **Transposition augmentation** — undo key normalization, transpose +/-6 semitones (like TonicNet) | 13x training data | Low |
| 3 | **Chord tokens** — Music21-analyzed chord at each timestep | Direct harmonic supervision | Low-Medium |

### Tier 2 — Medium Impact

| # | Change | Expected Impact | Effort |
|---|--------|----------------|--------|
| 4 | **Continuation embedding** — auxiliary embedding tracking held-note duration per voice | Better sustained notes | Low |
| 5 | **Full-sequence training** — SEQ_LEN=3000+ for chorales (once dense tokenization is in) | End-to-end phrase learning | Low |
| 6 | **Simplify output heads** — single cross-entropy over small vocab instead of factored type+value | Less overhead for simple task | Low |

### Tier 3 — Worth Trying

| # | Change | Expected Impact | Effort |
|---|--------|----------------|--------|
| 7 | **GRU option** — 3-layer GRU with variational dropout (305 chorales may be too small for transformer) | Simpler model may generalize better on small data | Medium |
| 8 | **SGD + OneCycleLR** — TonicNet's optimizer setup instead of AdamW | Possibly better convergence | Low |

## Recommended Implementation Path

**Phase 1** (items 1+2+3): Create a new `pre_chorale_dense.py` that produces TonicNet-style dense token sequences with chord tokens and transposition augmentation. This is the core change that addresses root causes 1-4 simultaneously.

**Phase 2** (items 4+5+6): Add continuation embeddings, enable longer sequences, and simplify the output head for the dense representation.

**Phase 3** (item 7): Try a GRU if the transformer still underperforms on this small dataset.

## TonicNet Reference Details

- **Repo**: github.com/omarperacha/TonicNet
- **Voice ordering**: [chord, soprano, bass, alto, tenor] per timestep
- **Pitch range**: MIDI 36-81 + Rest = 48 pitch tokens
- **Chord types**: 50 classes (12 roots x 4 qualities + other + none)
- **Variational dropout**: Same dropout mask across all timesteps (Gal & Ghahramani 2016)
- **Generation**: Autoregressive with strict cyclic voice ordering; random sampling or beam search (width=10)
- **Post-processing**: `smooth_rhythm()` merges consecutive identical pitches into held notes
