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

### Option B: Chronological merge with time-shifts

Merge all conditioning voices into a single chronological event stream
ordered by onset time, using time-shifts (deltas) between events:

```
Context (soprano + bass merged by onset time):
  Δ0 S:C5  Δ0 B:C3  Δ3 S:D5  Δ1 B:G2  Δ2 S:E5  Δ0 B:C3 ...

Target (alto, autoregressive):
  [SEP] Δ0 A:E4  Δ2 A:F4  Δ4 A:... → predict next
```

Events from all conditioning voices appear in a single timeline.
Time-shifts encode the delta to the next event, regardless of which
voice it belongs to.  This is compact — voices with long held notes
contribute no tokens during the hold.

**The alignment problem:** the target voice's timeline and the
context's timeline must be synchronized.  If the alto model has
generated Δ2+Δ4=6 beats into the piece, it needs to attend to context
events near cumulative time = 6.  Three sub-approaches:

**B1. Learned alignment.** Trust the model to learn cumulative
time-shift tracking through self-attention.  Transformers can learn
this, but it's an implicit burden — the model must compute running
sums to discover which context events are "now."

**B2. Absolute time annotations.** Add absolute beat-position tokens
alongside time-shifts:

```
Δ0 @0 S:C5   Δ0 @0 B:C3   Δ3 @3 S:D5   Δ1 @4 B:G2 ...
   [SEP]
Δ0 @0 A:E4   Δ2 @2 A:F4   Δ4 @6 A:...
```

The `@N` tokens are redundant (computable from cumulative Δ) but give
the model explicit anchors.  When generating alto at `@6`, attention
can directly match `@6` in the context.  Costs ~1 extra token per
event.

**B3. Musical time embedding (recommended).** Instead of discrete
time tokens, add a continuous sinusoidal embedding indexed by
cumulative musical time — analogous to positional encoding but on a
musical clock rather than token position:

```python
# Standard positional encoding: token index i
pe[i] = sin(i / 10000^(2k/d))

# Musical time encoding: cumulative beat position t
te[i] = sin(t_i / 10000^(2k/d))
```

Each token gets both embeddings:
```python
token_repr = token_embedding + token_position_encoding + musical_time_encoding
```

Events at the same musical time get similar time embeddings regardless
of where they sit in the token sequence.  No extra tokens, no grid
assumption.  The `musical_time` values are precomputed by accumulating
time-shifts during preprocessing and stored alongside the token IDs.

**Why B3 is the best alignment mechanism:**
- Zero token overhead (embedded, not tokenized)
- Works for arbitrary rhythms and rubato, not just fixed grids
- Alignment is structural (same beat → similar embedding) not learned
- Naturally extends to cross-attention (Option C): encoder and decoder
  share the same musical time embedding space, so cross-attention
  weights can align by beat position without any explicit annotation

**Token budget:** compact because only note onsets appear — a voice
sustaining a half note contributes 0 tokens during the hold.  Typical
alto model context: ~150 (S events) + ~150 (B events) + ~200 (A so
far) = ~500 tokens.  Fits in 512 or 1024.

**Pros:** Compact (no rest/sustain tokens), handles irregular rhythms,
musical time embedding gives structural alignment for free.
**Cons:** Slightly more complex preprocessing (accumulate times, merge
and sort events across voices).

### Option C: Cross-attention into conditioning voices

Separate the architecture into an encoder for conditioning voices and
a decoder for the target voice.  The decoder cross-attends into the
encoder's output.  Both sides use the same musical time embedding
from B3, so cross-attention weights naturally align by beat position.

```
┌─────────────────────────────────────────────────┐
│  Conditioning encoder (shared across stages)    │
│                                                 │
│  Input: chronological merge of completed voices │
│    Δ0 S:C5  Δ0 B:C3  Δ3 S:D5  Δ1 B:G2 ...   │
│  Embeddings: token + token_position + mus_time  │
│                                                 │
│  Output: hidden states H_cond (T_c, D)         │
└──────────────────────┬──────────────────────────┘
                       │ cross-attention
┌──────────────────────▼──────────────────────────┐
│  Voice decoder (one per voice, or shared+adapter)│
│                                                 │
│  Self-attention over target voice tokens so far │
│  Embeddings: token + token_position + mus_time  │
│  Cross-attention into H_cond at each layer      │
│                                                 │
│  Output: next token prediction for target voice │
└─────────────────────────────────────────────────┘
```

**How multiple conditioning voices are handled:**

The conditioning encoder takes ALL completed voices as a chronological
merge (same format as Option B), producing a single sequence of hidden
states.  The decoder doesn't need to know how many conditioning voices
there are — it just cross-attends into whatever the encoder produced.

This means the same decoder architecture works for all stages:
- Stage 2 (bass): encoder sees soprano only
- Stage 3 (alto): encoder sees soprano + bass merged
- Stage 4 (tenor): encoder sees soprano + bass + alto merged

The encoder grows by ~150 tokens per added voice.  The decoder stays
the same size (~200–300 tokens for the full target voice).

**Why cross-attention suits this problem:**

1. **Separation of concerns.** The encoder builds a rich representation
   of the harmonic context.  The decoder focuses on voice-leading for
   its specific voice.  Neither needs to parse the other's token format.

2. **Temporal alignment via musical time embedding.** Both encoder and
   decoder use the same sinusoidal musical time embedding.  When the
   decoder generates a note at beat 6, its musical time embedding is
   similar to encoder tokens near beat 6.  Cross-attention weights
   align by beat position automatically — no explicit time grid needed.

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

**Start with Option B + B3 (chronological merge with musical time
embedding), plan for C.**

Option B reuses the existing encoder-only transformer — just changes
the tokenization and adds a musical time embedding.  The B3 embedding
provides structural alignment without a fixed grid, so the approach
works for both chorales and free-rhythm music from the start.

If we later want cleaner separation of conditioning from generation,
we migrate to cross-attention (Option C).  The chronological merge
format and musical time embedding carry over unchanged — only the
model architecture changes (encoder-only → encoder-decoder).

## Per-voice token vocabulary

Each voice model needs a much smaller vocabulary than the full pipeline:

| Token type   | Size | Notes                                 |
|--------------|------|---------------------------------------|
| PAD          | 1    |                                       |
| BOS          | 1    |                                       |
| EOS          | 1    |                                       |
| SEP          | 1    | Boundary between context and target   |
| BAR          | 1    | Bar boundary marker                   |
| TIME_SHIFT   | 48   | 1/24-QN steps, max 2 QN (half note)   |
| PITCH        | ~30  | Voice-specific range (see below)      |
| REST         | 1    | Explicit silence event                |
| VOX_S/B/A/T  | 4    | Voice tags in conditioning context    |
| **Total**    | ~90  | Shared across all voice models        |

TIME_SHIFT tokens encode deltas (time to next event), not grid
positions.  The 1/24-QN resolution matches the existing pipeline and
handles both straight and triplet rhythms.

Voice-specific pitch ranges (from existing `pre.py` config):

| Voice    | MIDI range | Semitones | Pitch tokens |
|----------|------------|-----------|--------------|
| Soprano  | 57–84      | 27        | 28           |
| Alto     | 50–77      | 27        | 28           |
| Tenor    | 43–72      | 29        | 30           |
| Bassvox  | 33–69      | 36        | 37           |

VEL and DUR tokens can likely be omitted for chorales — velocity is
uniform and duration is implicit (note-on to next note-on or rest in
the same voice).

### Musical time embedding

In addition to the token vocabulary, each token carries a precomputed
musical time value (cumulative beat position as a float).  This is
encoded as a sinusoidal embedding added to the token representation:

```python
token_repr = token_embedding[tok_id]
           + position_encoding[tok_pos]     # token sequence position
           + musical_time_encoding[mus_t]   # cumulative beat position
```

The `musical_time_encoding` uses the same sinusoidal formula as
standard positional encoding but indexed by continuous musical time
(in quarter notes) rather than discrete token index.  This provides
structural temporal alignment across context and target voice streams
without any extra tokens or grid assumptions.

## Training procedure

**Data preparation:**
1. Parse each chorale MIDI into 4 separate voice sequences
2. Encode each voice as: `Δt PITCH Δt PITCH ... [EOS]`
   (time-shifts between consecutive note onsets in that voice)
3. Compute cumulative musical time for each token
4. For voice N, build conditioning context by chronologically merging
   voices 1..N-1 with voice tags: `Δ0 [S] PITCH Δ0 [B] PITCH Δ3 [S] PITCH ...`
5. Concatenate: `[BOS] context [SEP] target_voice [EOS]`
6. Augment: transpose ±3 semitones (7x), checking voice ranges

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
1. Generate soprano (no conditioning):
   model([BOS] [SEP]) → Δt₁ P₁ Δt₂ P₂ ... [EOS]
   Compute cumulative musical times for soprano tokens.

2. Build context from soprano:
   ctx = chronological_merge([soprano], voice_tags=[[S]])

3. Generate bass (conditioned on soprano):
   model([BOS] ctx [SEP]) → Δt₁ P₁ Δt₂ P₂ ... [EOS]
   Musical time embedding keeps bass aligned with soprano context.

4. Build context from soprano + bass:
   ctx = chronological_merge([soprano, bass], voice_tags=[[S],[B]])

5. Generate alto (conditioned on soprano + bass):
   model([BOS] ctx [SEP]) → Δt₁ P₁ ... [EOS]

6. Build context from soprano + bass + alto:
   ctx = chronological_merge([soprano, bass, alto], voice_tags=[[S],[B],[A]])

7. Generate tenor (conditioned on all three):
   model([BOS] ctx [SEP]) → Δt₁ P₁ ... [EOS]

8. Convert each voice's time-shift + pitch sequence to absolute
   note events, merge into multi-track MIDI.
```

Total generation: 4 sequential forward passes.  Each is fast (small
model, short sequences).  Musical time embedding is recomputed for
the context at each stage (cumulative sum of the merged time-shifts).

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

1. **Musical time embedding resolution.** The sinusoidal encoding has
   a natural frequency range.  Do we need to tune the base frequency
   for musical timescales (beats, bars) vs the default 10000 used in
   NLP positional encodings?  Likely yes — musical time spans ~0–60 QN,
   not thousands of tokens.

2. **Soprano source.** Should the soprano model generate freely, or
   should we seed it with a known hymn tune (as Bach did)?  Seeding
   makes evaluation more comparable to TonicNet's harmonization task.

3. **Chord conditioning.** Should we add an explicit chord stage before
   the soprano?  This creates a 5-stage cascade: Chords → S → B → A → T.
   The chord model could be very simple (HMM or small transformer).

4. **Error propagation.** How badly do soprano mistakes degrade the
   inner voices?  Scheduled sampling during training should help, but
   we'll need to measure this empirically.

5. **Time-shift quantization for context merge.** When merging voices
   chronologically, simultaneous onsets (Δ=0 between voices) are common
   in chorales.  Should we enforce a canonical voice ordering for
   simultaneous events (S before B before A before T)?  Likely yes, for
   consistency during training.
