# Architecture Review: Pipeline vs TonicNet for Bach Chorales

## Overview

This document compares two approaches to modeling Bach chorales:

- **ai-music-full-pipeline** — general-purpose transformer with factored event-stream encoding
- **TonicNet** — GRU with domain-specific interleaved voice+chord encoding

Both train on the same ~350 JSB Chorales dataset.

## Representation Comparison

### Token encoding per timestep

**TonicNet: 5 tokens**
```
CHORD  SOPRANO  BASS  ALTO  TENOR     ← complete harmonic snapshot
```

**Pipeline: ~17 tokens**
```
TIME_SHIFT  INST VEL PITCH DUR  INST VEL PITCH DUR  INST VEL PITCH DUR  INST VEL PITCH DUR
            ──── soprano ────   ──── alto ────────   ──── tenor ───────   ──── bassvox ─────
```

3.4x token overhead for the same musical information.

### Effective context

|                              | TonicNet              | Pipeline                  |
|------------------------------|-----------------------|---------------------------|
| Context window               | 2880 tokens           | 512 tokens                |
| Timesteps per window         | ~576 (full chorale)   | ~30 (a few bars)          |
| Vocab size                   | 98                    | ~96                       |
| Tokens to encode one chorale | ~800–1500             | ~4000–5000                |
| Training data augmentation   | 12x (all keys)        | 1x (normalized to C/Am)   |

TonicNet trains on entire chorales. The pipeline sees only ~30-chord sliding windows.

## Why TonicNet's design fits chorales better

1. **Explicit chord tokens.** Bach chorales are harmonized chord progressions. TonicNet puts the chord label in the sequence — the model knows the harmonic context when predicting each voice. The pipeline has no harmonic representation; it must infer harmony from pitch co-occurrence.

2. **Fixed time grid.** Chorales sit on a 16th-note grid. TonicNet encodes exactly this — one slot per 16th note, no wasted tokens. The pipeline's 1/24-QN resolution (6x finer than a 16th note) buys nothing for chorales.

3. **Interleaved voices = implicit voice leading.** The fixed order (Chord → S → B → A → T) means soprano conditions on the chord, bass on chord+soprano, alto on all three, etc. This mirrors how chorale harmonization is taught — outer voices first, inner voices fill.

4. **12x data augmentation.** TonicNet transposes every chorale through all 12 keys, turning ~350 chorales into ~4200 training examples. The pipeline normalizes to C major/A minor with no augmentation.

5. **Full-sequence GRU state.** The hidden state carries voice-leading context across the entire piece. For short, structurally coherent pieces like chorales, this persistent memory is sufficient.

## Where the pipeline has theoretical advantages (that don't help here)

- **Self-attention** captures long-range dependencies — but chorales are short enough that a GRU handles them fine.
- **Variable duration encoding** — but chorales are metrically regular.
- **Velocity encoding** — but chorale MIDIs have uniform velocity.
- **Factored type+value heads** — clever for a 2400-token heterogeneous vocab, unnecessary for 98 tokens.
- **Grammar-constrained generation** — guarantees well-formed output, but TonicNet's rigid 5-token cycle is an even stronger constraint.

## Proposed improvements to the pipeline

### Tier 1 — Easy wins, big impact

#### 1. Increase context window for chorales

The single biggest gap. At 512 tokens the model never sees a full phrase. TonicNet sees the entire piece.

- Bump `SEQ_LEN` to 2048 for chorale training — most chorales fit in ~1500 pipeline tokens
- The transformer architecture already supports variable sequence lengths
- Requires passing a chorale-specific seq_len through auto-scaler and training config
- Cost: more GPU memory per batch, but chorales are a small dataset

#### 2. Restore transposition augmentation

We normalized to C/Am and dropped augmentation — the model sees only one key center. TonicNet sees all 12. Two options:

- **(a)** Undo normalization, augment all 12 keys (matches TonicNet exactly)
- **(b)** Keep normalization, add ±3 semitone augmentation (7x data, narrower pitch range)

Option (b) is probably better: concentrated pitch vocabulary + meaningful data multiplier.

### Tier 2 — Moderate effort, high impact

#### 3. Add chord tokens to the event stream

The biggest representational gap. Detect chords during preprocessing (music21 is already a dependency) and inject a `CHORD` event at each beat or harmonic change:

```
TIME_SHIFT → CHORD → INST VEL PITCH DUR → INST VEL PITCH DUR → ...
```

One new event type, ~50 chord values (same space TonicNet uses). Costs 1 token per chord change but gives the model explicit harmonic scaffolding. The factored head architecture supports adding a new type cleanly.

#### 4. Compress redundant tokens for homophonic textures

Three independent sub-ideas:

- **Default velocity.** Chorales have uniform velocity. Add a `VEL_DEFAULT` token or let `INST` imply a default, saving 1 token per note (4 per timestep).
- **Implicit duration.** When note duration equals the grid step (16th note), omit `DUR`. Let absence mean "grid-length". Saves 1 token per grid-aligned note.
- **Combined INST+PITCH.** For a fixed 4-voice setup, a single `SOPRANO_C5` token would collapse 3 tokens (INST+VEL+PITCH) into 1. Breaks generality.

The first two are backward-compatible. Together they cut ~8 tokens per timestep, bringing the ratio from 17:5 down to roughly 9:5.

### Tier 3 — Worth considering later

#### 5. Voice-leading event order

Reorder within-timestep events: soprano → bass → alto → tenor (outer voices first, matching counterpoint pedagogy). Currently ordered by instrument index. Free change in `pre.py`, helps the autoregressive model condition inner voices on outer voices.

#### 6. Two-pass generation: harmony then voices

Generate a chord skeleton first, then fill in voices conditioned on the chords. Mirrors how arrangers work and how TonicNet's interleaving implicitly works. Significant architectural change.

## Recommended implementation order

Changes **1 + 2b + 3** together would close most of the gap without breaking the general pipeline (blues training unaffected):

1. Full-chorale context → matches TonicNet's whole-piece training
2. 7x augmented data → approaches TonicNet's 12x
3. Chord tokens → matches TonicNet's harmonic awareness

## Summary

TonicNet wins on chorales because domain-specific representation beats general-purpose representation when the domain is narrow and well-defined. The pipeline's flexibility (arbitrary instruments, complex rhythms, drums) is dead weight for 4-voice a cappella music on a regular grid. The proposed changes narrow the gap by giving the pipeline chorale-aware features while preserving its generality for other genres.
