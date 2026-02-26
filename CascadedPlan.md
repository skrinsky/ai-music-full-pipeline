# Cascaded Voice Models for Bach Chorales

## Motivation

The current pipeline interleaves all 4 voices into a single event stream
(~17 tokens per timestep, ~4000–5000 tokens per chorale). A 512-token
context window sees only ~30 timesteps — never a full phrase.

By factoring into per-voice models, each voice needs ~300–400 tokens for
a full chorale. The context window problem disappears.

## Architecture: Cascade with conditioning

```
Stage 1:  Soprano model   →  melody (no conditioning)
Stage 2:  Bass model      →  bass line, conditioned on soprano
Stage 3:  Alto model      →  alto, conditioned on soprano + bass
Stage 4:  Tenor model     →  tenor, conditioned on soprano + bass + alto
```

This mirrors compositional practice: outer voices define the harmonic
framework, inner voices fill according to voice-leading rules.

## Conditioning strategy

The central design question: how does voice model N "see" the
already-generated voices 1..N-1?

### Option A: Prefix token blocks

Concatenate completed voices as a token prefix, separated by voice
boundary tokens:

```
[BOS] [VOX_S] s₁ s₂ s₃ ... [VOX_B] b₁ b₂ b₃ ... [VOX_A] a₁ a₂ ... → predict next a
       ────── soprano ──────  ────── bass ─────────  ── alto so far ──
```

Each sᵢ and bᵢ are just the voice's own tokens (TIME_SHIFT + PITCH,
possibly VEL + DUR).  The voice-boundary tokens (`VOX_S`, `VOX_B`,
`VOX_A`) tell the model which voice follows.

**Time alignment is implicit.** The model must learn that soprano's
3rd TIME_SHIFT and bass's 3rd TIME_SHIFT refer to the same musical
moment.  Self-attention can discover this — the TIME_SHIFT values
encode elapsed time, so cumulative sums must match at simultaneous
events — but it's an extra burden on the model.

**Token budget for the alto model (3 conditioning voices):**
~300 (S) + ~300 (B) + ~300 (A so far) = ~900 tokens.  Fits in 1024.

**Pros:** Simple, uses the existing encoder-only transformer unchanged.
**Cons:** Time alignment is learned, not structural.  Prefix grows
linearly with number of conditioning voices.

### Option B: Time-aligned interleaved prefix

Interleave conditioning voices by timestep on a shared time grid, then
generate the target voice timestep by timestep:

```
Context (fixed, from completed voices):
  [t=0 S:C5 B:C3] [t=1 S:D5 B:B2] [t=2 S:E5 B:A2] ...

Target (autoregressive):
  [t=0 A:E4] [t=1 A:F4] [t=2 A:...] → predict next
```

Time alignment is explicit — each timestep groups all voices together.
The model attends to `[t=7 S:__ B:__]` when predicting `[t=7 A:__]`.

This is essentially TonicNet's encoding applied to the conditioning
voices.  It works naturally for chorales (fixed 16th-note grid) but
assumes a shared time grid.

**Token budget:** ~2 tokens per conditioning voice per timestep (voice
tag + pitch), so the alto model sees ~2×150 (context) + ~150 (target)
= ~450 tokens for a full chorale.  Very efficient.

**Pros:** Explicit time alignment, compact, musically transparent.
**Cons:** Requires a shared time grid (fine for chorales, breaks for
free-rhythm music).  Mixes two different token formats (interleaved
context vs sequential target).

### Option C: Cross-attention into conditioning voices

Separate the architecture into an encoder for conditioning voices and
a decoder for the target voice.  The decoder cross-attends into the
encoder's output.

```
┌─────────────────────────────────────────────────┐
│  Conditioning encoder (shared across stages)    │
│                                                 │
│  Input: time-interleaved completed voices       │
│    [t=0 S:C5 B:C3] [t=1 S:D5 B:B2] ...       │
│                                                 │
│  Output: hidden states H_cond (T_c, D)         │
└──────────────────────┬──────────────────────────┘
                       │ cross-attention
┌──────────────────────▼──────────────────────────┐
│  Voice decoder (one per voice, or shared+adapter)│
│                                                 │
│  Self-attention over target voice tokens so far │
│  Cross-attention into H_cond at each layer      │
│                                                 │
│  Output: next token prediction for target voice │
└─────────────────────────────────────────────────┘
```

**How multiple conditioning voices are handled:**

The conditioning encoder takes ALL completed voices interleaved by
timestep, producing a single sequence of hidden states.  The decoder
doesn't need to know how many conditioning voices there are — it just
cross-attends into whatever the encoder produced.

This means the same decoder architecture works for all stages:
- Stage 2 (bass): encoder sees soprano only
- Stage 3 (alto): encoder sees soprano + bass interleaved
- Stage 4 (tenor): encoder sees soprano + bass + alto interleaved

The encoder grows by ~150 tokens per added voice.  The decoder stays
the same size (~300–400 tokens for the full target voice).

**Why cross-attention suits this problem:**

1. **Separation of concerns.** The encoder builds a rich representation
   of the harmonic context.  The decoder focuses on voice-leading for
   its specific voice.  Neither needs to parse the other's token format.

2. **Temporal alignment via attention.** Cross-attention weights
   naturally learn to align: when generating alto at t=7, the decoder
   attends most strongly to encoder positions near t=7.  No explicit
   time grid needed, though having one helps.

3. **Scalable to N voices.** Adding a 5th conditioning voice just adds
   ~150 tokens to the encoder input.  The decoder is unchanged.

4. **Shared encoder weights.** The same encoder can process any subset
   of voices.  Only the decoder (or a small adapter layer) needs to be
   voice-specific.

**Pros:** Cleanest architecture, naturally handles variable numbers of
conditioning voices, good separation of context vs generation.
**Cons:** Encoder-decoder is more complex than encoder-only.  Requires
modifying the current `FactorizedESModel` architecture (which is
encoder-only).

### Recommendation

**Start with Option B (time-aligned interleaving), plan for C.**

Option B is the fastest to implement — it reuses the existing
encoder-only transformer and just changes the tokenization.  The
chorale time grid makes alignment trivial.  If we later need to handle
free-rhythm music or want cleaner separation, we migrate to
cross-attention (Option C) using the same interleaved encoding for the
conditioning side.

## Per-voice token vocabulary

Each voice model needs a much smaller vocabulary than the full pipeline:

| Token type   | Size | Notes                            |
|--------------|------|----------------------------------|
| PAD          | 1    |                                  |
| BOS          | 1    |                                  |
| EOS          | 1    |                                  |
| BAR          | 1    | Bar boundary marker              |
| TIME_SHIFT   | 16   | 16th-note grid, max 1 bar       |
| PITCH        | ~30  | Voice-specific range (see below) |
| REST         | 1    | Silence at this timestep         |
| VOX_S/B/A/T  | 1–3  | Voice boundary tags (Option B)   |
| **Total**    | ~55  | Per-voice model                  |

Voice-specific pitch ranges (from existing `pre.py` config):

| Voice    | MIDI range | Semitones | Pitch tokens |
|----------|------------|-----------|--------------|
| Soprano  | 57–84      | 27        | 28           |
| Alto     | 50–77      | 27        | 28           |
| Tenor    | 43–72      | 29        | 30           |
| Bassvox  | 33–69      | 36        | 37           |

VEL and DUR tokens can likely be omitted for chorales — velocity is
uniform and duration is implied by the grid (note-on to next note-on
or rest).

## Training procedure

**Data preparation:**
1. Parse each chorale MIDI into 4 separate voice sequences
2. Quantize to 16th-note grid (already done in NPZ source data)
3. Encode each voice as: `[BOS] TIME PITCH TIME PITCH ... [EOS]`
4. For voice N, prepend the conditioning context (Option B encoding)
5. Augment: transpose ±3 semitones (7x), checking voice ranges

**Training:**
- Teacher forcing: conditioning voices are ground truth during training
- Each stage can be trained independently and in parallel
- Loss: cross-entropy on target voice tokens only (exclude prefix)
- Shared-weight variant: one model with a voice-ID embedding, trained
  on all (context, target) pairs from all 4 stages simultaneously

**Scheduled sampling (optional, combats error cascade):**
During training, occasionally replace ground-truth conditioning tokens
with model-generated tokens (from an earlier training snapshot).
Teaches the model to be robust to imperfect conditioning inputs.

## Generation procedure

```
1. Generate soprano:
   soprano_model([BOS]) → s₁ s₂ ... [EOS]

2. Encode soprano as context:
   ctx = interleave_by_timestep([soprano])

3. Generate bass:
   bass_model([ctx] [SEP] [BOS]) → b₁ b₂ ... [EOS]

4. Encode soprano + bass as context:
   ctx = interleave_by_timestep([soprano, bass])

5. Generate alto:
   alto_model([ctx] [SEP] [BOS]) → a₁ a₂ ... [EOS]

6. Encode soprano + bass + alto as context:
   ctx = interleave_by_timestep([soprano, bass, alto])

7. Generate tenor:
   tenor_model([ctx] [SEP] [BOS]) → t₁ t₂ ... [EOS]

8. Merge 4 voice sequences → multi-track MIDI
```

Total generation: 4 sequential forward passes.  Each is fast (small
model, short sequences).

## Shared vs independent weights

**Option I: Four independent models**
- Each specializes fully for its voice
- 4× the total parameters, but each is small (~50K?)
- Train in parallel

**Option II: Shared body + voice adapter**
- One transformer body shared across all 4 stages
- Small per-voice adapter (learned voice embedding + output head)
- ~1/3 the parameters of Option I
- Trained on all stages jointly: each batch mixes soprano-only,
  soprano→bass, S+B→alto, S+B+A→tenor examples
- The model learns general voice-leading; adapters specialize

Recommendation: **Start with Option II (shared weights).**  The small
dataset size (~350 chorales × 7 augmentations = ~2450 examples) favors
parameter sharing.  If voices prove too different, split later.

## Relationship to existing pipeline

This cascaded approach is a **separate model architecture**, not a
modification of the current interleaved event-stream model.  It would
live alongside the existing pipeline:

```
training/
  train.py              ← existing interleaved model
  train_cascade.py      ← new cascaded voice model
  pre.py                ← reuse MIDI parsing, add per-voice serialization
  generate.py           ← existing interleaved generation
  generate_cascade.py   ← new cascaded generation
```

The Makefile would get new targets:
```
make chorale-cascade-preprocess
make chorale-cascade-train
make chorale-cascade-generate
```

## Evaluation: how to compare with TonicNet

To fairly compare the cascaded pipeline with TonicNet:

1. **Token-level accuracy** — predict held-out chorale voices given
   ground-truth conditioning (matches TonicNet's 90% metric)
2. **Harmonicity** — % of simultaneities that form valid triads/7ths
3. **Voice-leading violations** — parallel 5ths/8ves, voice crossing,
   range violations (standard chorale metrics)
4. **Listening test** — blind A/B comparison of generated chorales

## Open questions

1. **Grid assumption.** The 16th-note grid works for chorales.  For
   extending to other genres, we'd need to handle variable timing —
   motivating the eventual move to cross-attention (Option C).

2. **Soprano source.** Should the soprano model generate freely, or
   should we seed it with a known hymn tune (as Bach did)?  Seeding
   makes evaluation more comparable to TonicNet's harmonization task.

3. **Chord conditioning.** Should we add an explicit chord stage before
   the soprano?  This creates a 5-stage cascade: Chords → S → B → A → T.
   The chord model could be very simple (HMM or small transformer).

4. **Error propagation.** How badly do soprano mistakes degrade the
   inner voices?  Scheduled sampling during training should help, but
   we'll need to measure this empirically.
