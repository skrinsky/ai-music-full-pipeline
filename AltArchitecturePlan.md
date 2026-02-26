# Alternative Multi-Voice Architectures

## Context

`CascadedPlan.md` describes a cascade architecture where voices are
generated sequentially (S → B → A → T), each conditioned on
previously generated voices via a single transformer.  This document
surveys fundamentally different coordination strategies.

All approaches share the same goal: generate 4-voice Bach chorales
where each voice is musically coherent and the vertical harmony is
correct.  They differ in how voices become aware of each other.

## 1. Product of Experts

Train independent pairwise models — each predicts one voice
conditioned on *one* other voice.  At generation time, multiply the
distributions and sample from the product.

```
P(alto | soprano, bass, tenor)
  ∝ P_s(alto | soprano) × P_b(alto | bass) × P_t(alto | tenor)
```

No single model ever sees the full 4-voice texture.  Coordination
emerges from the product of independent opinions.

**Training:** 12 small models (4 target voices × 3 conditioning
voices), each a simple sequence-to-sequence model with musical time
embedding for alignment.  Or 4 models with 3-head output, one head
per conditioning voice.

**Generation:**
```
for each timestep:
    for each target voice V:
        logits_V = sum of log-probs from each pairwise expert
        sample V from softmax(logits_V)
```

All voices can be generated in parallel at each timestep if we use
the previous timestep's values for conditioning (like a Jacobi
iteration).

**Pros:**
- Each model is tiny (one voice in, one voice out)
- Embarrassingly parallel training
- Adding a 5th voice adds pairwise models, not a redesign
- Well-studied framework (Hinton 1999, Product of Experts)

**Cons:**
- Product of experts can be overconfident (peaky distributions) or
  degenerate when experts disagree
- No model reasons about the full vertical sonority holistically
- Normalization of the product distribution is intractable in general
  (though manageable with a 30-token pitch vocabulary)
- Pairwise interactions may miss higher-order constraints (e.g.,
  "this works with soprano and bass individually but creates parallel
  5ths between alto and tenor")

## 2. Parallel Generation with Harmonic Referee

All 4 voices generate simultaneously and independently.  A lightweight
referee model evaluates proposed vertical slices at sync points and
triggers resampling when violations are detected.

```
Every beat (or every 16th note):
    soprano proposes next note(s)
    bass proposes next note(s)
    alto proposes next note(s)
    tenor proposes next note(s)
              ↓
    Referee scores the 4-voice vertical slice:
      - Valid triad or seventh chord?
      - Parallel 5ths or octaves with previous beat?
      - Voice crossing?
      - Range violation?
              ↓
    If score < threshold:
      Resample worst-scoring voice(s), re-evaluate
              ↓
    Accept, advance to next sync point
```

**Referee options:**
- **Rule-based:** Encode standard counterpoint rules directly.
  Fast, interpretable, no training needed.  Covers parallel 5ths,
  voice crossing, doubling rules, chord membership.
- **Learned:** Small classifier trained on (good chorale, bad
  random-voice chorale) pairs.  More flexible, can learn stylistic
  nuances beyond textbook rules.
- **Hybrid:** Rules as hard constraints, learned model for soft
  stylistic preferences.

**Pros:**
- 4x faster than cascade (parallel voice generation)
- Referee can be very simple — even pure rules
- Clean separation: voice models handle melody, referee handles harmony
- Easy to debug: referee decisions are interpretable

**Cons:**
- Rejection sampling can be slow if independent voices are poorly
  coordinated (many rejections per beat)
- Voices don't adapt *to* each other, they just get vetoed
- Resampled voices may repeatedly propose bad notes if they have no
  harmonic awareness at all
- Sync-point granularity matters: too coarse allows drift, too fine
  is expensive

**Mitigation:** Give each voice model a lightweight harmonic hint
(e.g., chord symbol per beat) so proposals are harmonically informed.
The referee then catches remaining violations rather than doing all
the harmonic work.

## 3. Latent Harmonic Plan + Independent Decoders

A two-level architecture that separates *what harmony* from
*which notes*:

```
┌─────────────────────────────────────────────┐
│  Plan model                                 │
│  Generates harmonic sequence:               │
│    h₁  h₂  h₃  h₄ ...  (one per beat)     │
│                                             │
│  Could be:                                  │
│    - Explicit chord symbols (C, Am, G7...)  │
│    - Learned latent vectors (VQ-VAE codes)  │
│    - Hybrid (chord + style embedding)       │
└──────────┬──────────────────────────────────┘
           │ broadcast to all decoders
    ┌──────┴──────┬──────────────┬─────────────┐
    ▼             ▼              ▼             ▼
┌────────┐  ┌────────┐   ┌────────┐   ┌────────┐
│Soprano │  │ Bass   │   │ Alto   │   │ Tenor  │
│Decoder │  │Decoder │   │Decoder │   │Decoder │
│        │  │        │   │        │   │        │
│ h → S  │  │ h → B  │   │ h → A  │   │ h → T  │
└────────┘  └────────┘   └────────┘   └────────┘
```

Each decoder sees the same harmonic plan and independently generates
its voice's pitch sequence.  Voice-leading coherence comes from
the shared plan constraining what notes are appropriate at each beat.

**Plan model variants:**

*Explicit chords:* The plan is a symbolic chord sequence.  Can be
generated by a small transformer, borrowed from TonicNet's chord
labels, or even hand-specified.  ~50-token vocabulary (12 roots ×
4 qualities + other + none).

*Learned latent codes:* Train a VQ-VAE where the encoder compresses
a full 4-voice beat into a discrete code, and per-voice decoders
reconstruct each voice from the code sequence.  The codes learn to
capture whatever harmonic information the decoders need — potentially
richer than chord symbols (e.g., encoding inversion, spacing, style).

*Hybrid:* Chord symbol + a small continuous style vector per beat.
The chord provides hard harmonic guidance; the style vector carries
soft preferences (open vs close voicing, rhythmic density, etc.).

**Pros:**
- Clean factorization of harmony and voice-leading
- Voice decoders are fully parallel (4x faster than cascade)
- Swap decoders for different styles, keep the same harmonic plan
- Plan model captures global structure; decoders are local
- Explicit chord plan is interpretable and controllable
- Natural user interface: specify a chord progression, get a chorale

**Cons:**
- Independent decoders may still produce parallel 5ths or voice
  crossings (the plan constrains pitch class, not specific voicing)
- Learned latent plans are hard to interpret and debug
- Requires either chord labels (available from TonicNet data) or a
  VQ-VAE training stage
- The plan must carry enough information that decoders don't need
  to see each other — otherwise we're back to needing cross-voice
  conditioning

**Mitigation:** Combine with the referee (approach 2) as a
post-processing step.  Plan provides 90% of coordination; referee
catches the remaining voice-leading violations.

## 4. Gibbs Sampling (Iterative Mutual Refinement)

Train a single model to predict any one voice given the other three
(masked prediction).  Generate by iterating:

```
Initialize all 4 voices (random, or from independent models)

Repeat for N rounds:
    Resample soprano given (bass, alto, tenor)
    Resample bass    given (soprano, alto, tenor)
    Resample alto    given (soprano, bass, tenor)
    Resample tenor   given (soprano, bass, alto)

Return final state
```

This is the approach used by **DeepBach** (Hadjeres et al. 2017).
No voice is "first" — all influence each other symmetrically.

**Training:** One model, four masking patterns.  Each training
example masks one voice and predicts it from the other three.  The
model sees all 4 patterns with equal probability.

**Key design choices:**
- *Initialization:* Random notes (slow convergence) vs independent
  voice model output (fast convergence, better starting point)
- *Iteration count:* DeepBach uses ~100 Gibbs steps.  More steps
  improve quality but cost time.
- *Sampling order:* Sequential (S→B→A→T per round) vs random
  (pick a random voice each step).  Random is theoretically better
  for mixing.
- *Temperature schedule:* Start hot (explore), anneal to cold
  (commit).  Simulated annealing flavor.

**Pros:**
- Symmetric — no privileged voice, no error cascade
- Full mutual awareness: every voice conditions on all others
- Single model handles all 4 masking patterns
- Can start from a partial specification (e.g., given soprano,
  resample only inner voices = harmonization task)

**Cons:**
- Slow: N full passes through the model (N ≈ 50–100)
- Convergence not guaranteed — can oscillate
- Training sees ground-truth context, generation sees its own noisy
  output (exposure bias in all directions, not just one as in cascade)
- Hard to know when to stop iterating (no clear convergence criterion)

## 5. Score-Level Diffusion

Treat the full chorale as a 2D piano roll (time × pitch × 4 voice
channels) and use a denoising diffusion model:

```
Forward process:
    ground-truth piano roll → add Gaussian noise → ... → pure noise

Reverse process (generation):
    pure noise → denoise step → ... → coherent 4-voice chorale
```

All voices emerge simultaneously from the denoising process.
Consistency is structural — the model learns the joint distribution
of all four voices, not conditionals.

**Representation:** A binary piano roll of shape
(T_steps × 128_pitches × 4_voices), or more compactly
(T_steps × ~40_pitches × 4_voices) using voice-specific pitch
ranges.

**Architecture:** U-Net or DiT (Diffusion Transformer) operating on
the 2D piano roll.  Convolutional layers capture local patterns
(chords, voice leading); attention captures global structure (phrase
repetition, cadences).

**Pros:**
- Holistic: generates the full joint distribution, not conditionals
- No ordering bias or error cascade
- Recent diffusion results in image/audio are very strong
- Natural inpainting: mask any subset of voices or time regions,
  denoise the rest (harmonization, continuation, fill-in)

**Cons:**
- Very different architecture from the current transformer pipeline
- Continuous representation (piano roll) vs discrete tokens
- Harder to control (can't easily seed with a soprano melody)
- ~350 chorales is likely too small for diffusion to work well
  (diffusion models are data-hungry)
- Inference is slow (many denoising steps)

## Comparison

| Approach | Parallelism | Mutual awareness | Complexity | Data efficiency | Generation speed |
|---|---|---|---|---|---|
| Cascade | Sequential | Unidirectional | Low | Good | 4 passes |
| Product of experts | Parallel | Pairwise only | Low | Good | 1 pass |
| Harmonic referee | Parallel | Via rejection | Medium | Good | 1 pass + retries |
| Latent plan + decoders | Parallel | Via shared plan | Medium | Moderate | 2 passes |
| Gibbs sampling | Iterative | Full symmetric | Medium | Good | 50–100 passes |
| Score diffusion | Simultaneous | Full joint | High | Poor | 50–1000 steps |

## Recommendations

### For ~350 Bach chorales (current dataset)

**Primary: Cascade** (`CascadedPlan.md`).  Simplest, most
data-efficient, compositionally natural.  Musical time embedding
handles flexible timing.

**Secondary: Latent plan + decoders** (approach 3 with explicit
chords).  The most interesting evolution path.  Start with chord
symbols from TonicNet's labels as the plan; per-voice decoders
conditioned on the chord sequence.  Parallel decoding.  If
independent decoders produce voice-leading errors, add a rule-based
referee (approach 2) as post-processing.

### For larger datasets or future expansion

**Gibbs sampling** (approach 4) becomes more attractive with more
data and compute.  The symmetric mutual awareness is theoretically
ideal for polyphony.

**Score diffusion** (approach 5) would be interesting as a research
direction if the dataset grows significantly (e.g., combining JSB
chorales with other SATB repertoire).

### What NOT to build first

**Product of experts** (approach 1) — the pairwise-only awareness
is a fundamental limitation for 4-voice counterpoint where
higher-order interactions matter.  Parallel 5ths between alto and
tenor would go undetected by models that only see (alto|soprano)
and (tenor|soprano).

## Relationship to CascadedPlan.md

The cascade architecture remains the recommended first implementation.
The approaches in this document represent the broader design space
and potential evolution paths.  Key reusable components across all
approaches:

- **Musical time embedding** (from CascadedPlan.md) — useful in
  all approaches that use sequential token representations
- **Per-voice tokenization** — all approaches except score diffusion
  benefit from compact per-voice token streams
- **Voice-specific pitch ranges** — applicable everywhere
- **Transposition augmentation** — applicable everywhere
