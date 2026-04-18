# Representation Research

Notes on fine-tuning pretrained music models on small personal datasets, the latent-space alignment problem, and methods for probing what a frozen pretrained network actually represents.

Compiled from a Claude Code advising session (2026-04-17) in preparation for a meeting with Summer on the Notochord / LoRA finetuning work in `finetune/`.

---

## 1. The core problem

Summer has a small MIDI dataset (tens of pieces) and is trying to push a pretrained model toward her compositional style using two approaches:

- **Notochord fine-tuning** (`finetune/notochord_*.py`) — a GRU with five parallel prediction heads (instrument / pitch / time / velocity / end), pretrained on Lakh MIDI. Current recipe: freeze backbone, small LR, CPU-only (MPS gives NaN loss).
- **LoRA on Multitrack-Music-Transformer** (`finetune/finetune.py`) — GPT-2 architecture, ~170k MIDIs of pretraining, LoRA rank 16 applied to `c_attn`.

JOS's concerns:

1. Insufficient data to push weights far in structurally useful directions.
2. Poor understanding of what the pretrained models' latent representations actually encode.
3. No principled way to know whether those representations are amenable to style-level fine-tuning.

These concerns are well-founded and point directly at the foundational question this document addresses.

---

## 2. Why small-data fine-tuning of large models is fundamentally hard

With N~50 pieces and a model with millions of parameters, you are in a regime where:

- **Full fine-tuning** → catastrophic forgetting (already observed in Summer's commit log).
- **Light fine-tuning** → the model barely moves; it memorizes surface tokens.
- **LoRA with small rank** → constrains the update, but the *direction* of that update is not inherently aligned with "style."

The latent space of a model trained on 170k heterogeneous MIDIs is **entangled** — no single direction corresponds to "bluesy" or "Summer-ish." Most fine-tuning just shifts the *output distribution* slightly toward whatever surface statistics the finetune data has (more triads of this type, more syncopation of that type), not deeper structural patterns.

---

## 3. Assessment of the two current methods

### Notochord

- **Pros**: small enough that real gradient updates happen; native MIDI event representation; real-time inference; discrete heads allow explicit conditioning on instrument.
- **Cons**: GRU has limited long-range context; five independent heads model harmony only implicitly through autoregressive factorization; no explicit structural prior (bars, chord, phrase).

### MMT + LoRA

- **Pros**: Transformer has long-range attention; large pretraining prior.
- **Cons**: LoRA r=16 on `c_attn` is a narrow conditioning channel; MMT vocab is dense (~20k); Summer's signal has to fight the base model's distributional pull.
- **Red flag**: the `_FALLBACK_NOTE` branch in `finetune/finetune.py:63–96` silently returns a *random* GPT-2 if the HuggingFace download fails. Per JOS's "no fallbacks, fail fast" rule this should `exit 1`. If it ever triggered unnoticed, Summer would be "fine-tuning" noise.

### Don't pick between them — use them for different roles

- **Notochord** → real-time, interactive, live-performance (its designed purpose).
- **MMT + LoRA** → offline, longer-form, style transfer with a large prior.

The freeze-backbone-+-low-LR recipe for Notochord is correct but extreme. Worth trying *partial* unfreezing: freeze early layers, allow late layers to move slightly. Late layers are where style-relevant abstractions live, if they live anywhere.

---

## 4. Alternatives worth considering before more fine-tuning

### 4.1 Prompting / seeding (the cheapest win)

Condition generation on 4–16 bars from one of Summer's pieces. Gets most of "style transfer" with zero training and zero risk of catastrophic forgetting. Under-used in music research because everyone reaches for fine-tuning.

### 4.2 Musical data augmentation

- Transpose to all 12 keys → instant 12× data.
- Tempo stretch ±10%, velocity jitter.
- Structural augmentation: extract and recombine motifs, chord progressions.
- 20 pieces becomes 500+ effective training examples.

### 4.3 Classifier-free guidance with a style token

Train a small conditioning embedding ("this is Summer's style") and use CFG at inference. Much more sample-efficient than whole-model fine-tuning. Standard in diffusion, underused in music LMs.

### 4.4 Explicit structural conditioning

Extract Summer's chord progressions and phrase structure from her MIDIs and use them as *inference-time constraints*. Separates "her harmonic language" (explicit, interpretable) from "notes over a chord" (model). More controllable, more debuggable.

### 4.5 Tiny model from scratch on heavily augmented data

Counter-intuitive but worth one experiment: a 200k-param model trained from scratch on augmented Summer-data may capture her idiom more faithfully than a fine-tuned large one — no competing prior to fight.

### 4.6 Hybrid: symbolic model + RAVE

Notochord / MMT generates note events → synthesize → pass through a RAVE decoder fine-tuned on Summer's sonic palette → final audio. Clean separation of composition and timbre.

---

## 5. What RAVE teaches us

**RAVE** (Caillon & Esling, IRCAM) is a variational autoencoder for raw audio. PQMF front-end → strided-conv encoder → low-dim latent → conv decoder with noise injection → PQMF reconstruction. Two-stage training: (1) VAE with reconstruction + KL, (2) freeze encoder and adversarially fine-tune only the decoder. After training, PCA on latents typically yields 4–16 interpretable control axes.

RAVE 2 refinements: Snake activation, improved noise synthesizer, discrete (VQ) latent variant, CachedConv for causal streaming, `nn~` Max/Pure Data integration.

### The apparent "small-data miracle," demystified

RAVE fine-tunes well on ~30 min to 1 hr of audio. But:

- Audio at 48 kHz is enormously information-dense. 30 min = ~86M samples. Summer's MIDI dataset: ~10⁵ tokens. **Three orders of magnitude more supervision signal per minute of material.**
- Reconstruction is a well-defined, dense objective. "Match the input waveform" is unambiguous. "What note comes next" is massively underdetermined.
- RAVE models *timbre*, not *composition*. It does not learn phrase structure, harmony, or voice leading — those aren't in its objective.

### Lesson

> **When data is scarce, choose a model whose objective is tightly coupled to a dense, unambiguous signal in that data.**

This reframes Summer's problem. Fruitful directions:

1. Switch modalities (audio → dense signal → RAVE-style methods).
2. Make the MIDI supervision denser (explicit chord labels, phrase labels, structure tokens).
3. Use retrieval/prompting (no data efficiency required).

---

## 6. GANs on MIDI — why the field moved on

### Notable attempts

- **C-RNN-GAN** (Mogren 2016) — RNN generator + discriminator for classical MIDI.
- **MidiNet** (Yang et al. 2017) — CNN bar-by-bar melody GAN, chord-conditioned.
- **MuseGAN** (Dong et al. 2018) — multi-track polyphonic generation from piano-roll tensors.
- **SeqGAN** and descendants (Yu et al. 2017) — policy gradient to work around discrete-sampling non-differentiability.

### Three structural problems

1. **Discrete tokens break gradient flow.** Workarounds (Gumbel-Softmax, REINFORCE, piano-roll, VAE-GAN) all introduce bias, high variance, or loss of structure.
2. **Mode collapse is lethal for music** — manifests as repetitive four-bar loops or always-the-same-progressions.
3. **No long-range coherence** — music has structure at beats → bars → phrases → sections; GANs generate "in one shot."

### Where adversarial ideas *do* work in music

- Audio vocoders (MelGAN, HiFiGAN, BigVGAN) — continuous, dense, native GAN territory.
- Stage-2 adversarial fine-tuning as a *perceptual loss* (RAVE's trick).
- Niche style / genre translation (CycleGAN-inspired).

### Relevance to Summer

A GAN-as-generator approach to N~50 MIDI files would be actively worse than what she has. One tractable adversarial idea: a small discriminator that judges "sounds like Summer vs generic Lakh" used as a light auxiliary reward on top of existing LoRA fine-tuning. Closer to RLHF than to classical GANs. Worth a week's experiment, not a research direction to bet on.

---

## 7. Understanding the representation at the freeze boundary

This is the foundational question for any fine-tuning strategy. When you freeze the first N layers and adapt on top, the layer-N representation is the *entire interface* the fine-tuning can see. If a feature you care about isn't present there, no adapter on top can conjure it.

### 7.1 Linear probing

Train a single linear layer on frozen activations to predict a musical property: key, tempo, composer, chord, phrase boundary, instrumentation, genre. If linearly decodable → it's in the representation. If not → it may be nonlinearly encoded (try a tiny MLP) or absent.

Always run against a **label-shuffled control task** (Hewitt & Liang 2019). A sufficiently expressive probe can find structure in random features; the gap between real and shuffled probe accuracy tells you if the feature is genuinely present.

### 7.2 MDL / information-theoretic probing (Voita & Titov 2020)

Instead of probe accuracy, measure the minimum *description length* to encode labels given the representation. Controls for probe capacity automatically — can't cheat by using a bigger probe.

### 7.3 Structural probes (Hewitt & Manning 2019)

For hierarchical structure: learn a linear projection such that pairwise distances match tree distances. Originally for syntax trees; maps onto phrase structure and harmonic hierarchy in music.

### 7.4 CKA (Centered Kernel Alignment)

Kornblith et al. 2019. Compares the *geometry* of two representation spaces. Answers: "does this layer respond to Summer's MIDI the same way it responds to Lakh?" High CKA → model already treats them as similar; fine-tuning has little room. Low CKA → her data is out of distribution; harder but more distinctive.

### 7.5 Dimensionality reduction

PCA / UMAP / t-SNE of activations at a chosen layer, colored by source (Summer vs Lakh vs genre labels). Qualitative but fast, often the most informative first step.

### 7.6 Logit lens / tuned lens

Project intermediate layers' activations through the frozen output head to "decode" what the model is predicting at each depth. For causal-LM music models, watch the pitch-class distribution sharpen layer by layer.

### 7.7 Sparse autoencoders (SAEs) on activations

Anthropic-style superposition work. Train an SAE on a layer's activations to decompose them into a dictionary of interpretable features ("activates on minor-seventh chords," "activates on syncopated onsets"). Heavy to run, most fine-grained answer available.

### 7.8 Causal interventions (activation patching)

Replace a hidden state with one from a different input; measure output change. Shows a feature is *causally used*, not just encoded. Overkill for a first pass, valuable for a deep study.

---

## 8. A concrete probing recipe for Summer

Pre-fine-tuning diagnostic, ~one week, high value.

1. **Extract hidden states at every layer** of MMT-GPT2 for
   (a) Summer's MIDIs,
   (b) a matched Lakh subset (same size, same instrumentation, same tempo band).
2. **Auto-label** both sets with standard MIR tools (`music21`, `pretty_midi`, `madmom`) for key, tempo, chord, density, meter.
3. **For each layer, compute:**
   - Linear probe accuracy on each musical property, against a label-shuffled control.
   - CKA between Summer-activations and Lakh-activations.
   - UMAP scatter colored by source.
4. **Output:** a layer-by-layer table showing which musical properties are decodable where, and where Summer's distribution is most distinct from Lakh.
5. **Decide the freeze boundary:** put LoRA (or unfreeze) at the layer where (a) style-relevant features are maximally decodable and (b) Summer's data is most distinct from the prior.

Currently LoRA targets `c_attn` on *every* layer uniformly — that spreads limited adaptation capacity thin. Concentrating it where it matters would be more sample-efficient.

### Secondary use: post-finetune probing

Run the same probes *after* fine-tuning. If the probe for "Summer-ness" improves, the LoRA did something structural. If it doesn't move, the fine-tune is cosmetic.

---

## 9. The publishable angle

The music-ML community mostly runs fine-tunes and reports sample quality. A thorough probing-first methodology paper — *"What do pretrained music transformers actually represent, and where?"* — would be genuinely novel. Summer could contribute:

- Layer-wise probing of MMT and/or Notochord for standard MIR properties.
- CKA geometry of personal/in-style data vs the pretraining distribution.
- Validation: predict fine-tuning success from pre-finetune probe outcomes.

This reframes Summer's practical problem as a research contribution to an under-examined area.

---

## References

### RAVE and neural audio synthesis
- Caillon & Esling, *RAVE: A variational autoencoder for fast and high-quality neural audio synthesis* (2021). arXiv:2111.05011 — https://arxiv.org/abs/2111.05011
- IRCAM RAVE repository — https://github.com/acids-ircam/RAVE
- Shepardson, Caillon & Esling, *The Notochord: a Flexible Concurrent Transformer Model for Real-Time Probabilistic Music Modelling* (AIMC 2023).
- Notochord repository — https://github.com/victor-shepardson/notochord
- Kong et al., *HiFi-GAN* (2020). arXiv:2010.05646 — https://arxiv.org/abs/2010.05646
- Kumar et al., *MelGAN* (2019). arXiv:1910.06711 — https://arxiv.org/abs/1910.06711
- Lee et al., *BigVGAN* (2022). arXiv:2206.04658 — https://arxiv.org/abs/2206.04658

### GANs for symbolic music
- Mogren, *C-RNN-GAN* (2016). arXiv:1611.09904 — https://arxiv.org/abs/1611.09904
- Yang, Chou & Yang, *MidiNet* (2017). arXiv:1703.10847 — https://arxiv.org/abs/1703.10847
- Dong, Hsiao, Yang & Yang, *MuseGAN* (AAAI 2018). arXiv:1709.06298 — https://arxiv.org/abs/1709.06298
- Yu et al., *SeqGAN* (AAAI 2017). arXiv:1609.05473 — https://arxiv.org/abs/1609.05473

### Transformer music models
- Huang et al., *Music Transformer* (2018). arXiv:1809.04281 — https://arxiv.org/abs/1809.04281
- Payne, *MuseNet* (OpenAI blog, 2019) — https://openai.com/research/musenet
- Multitrack Music Transformer (Natooz) — https://huggingface.co/Natooz

### Probing and interpretability

**Linear probes and control tasks**
- Alain & Bengio, *Understanding intermediate layers using linear classifier probes* (2016). arXiv:1610.01644 — https://arxiv.org/abs/1610.01644
- Conneau et al., *What you can cram into a single vector: Probing sentence embeddings for linguistic properties* (ACL 2018). arXiv:1805.01070 — https://arxiv.org/abs/1805.01070
- Hewitt & Liang, *Designing and interpreting probes with control tasks* (EMNLP 2019). arXiv:1909.03368 — https://arxiv.org/abs/1909.03368

**MDL / information-theoretic probes**
- Voita & Titov, *Information-theoretic probing with minimum description length* (EMNLP 2020). arXiv:2003.12298 — https://arxiv.org/abs/2003.12298
- Pimentel et al., *Information-theoretic probing for linguistic structure* (ACL 2020). arXiv:2004.03061 — https://arxiv.org/abs/2004.03061

**Structural probes**
- Hewitt & Manning, *A structural probe for finding syntax in word representations* (NAACL 2019) — https://aclanthology.org/N19-1419/

**Representational geometry**
- Kornblith, Norouzi, Lee & Hinton, *Similarity of neural network representations revisited* (ICML 2019). arXiv:1905.00414 — https://arxiv.org/abs/1905.00414
- Kriegeskorte, Mur & Bandettini, *Representational similarity analysis — connecting the branches of systems neuroscience* (Frontiers in Systems Neuroscience 2008) — https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008

**Logit / tuned lens**
- nostalgebraist, *interpreting GPT: the logit lens* (LessWrong 2020) — https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Belrose et al., *Eliciting latent predictions from transformers with the tuned lens* (2023). arXiv:2303.08112 — https://arxiv.org/abs/2303.08112

**Sparse autoencoders and mechanistic interpretability**
- Bricken et al. (Anthropic), *Towards monosemanticity: decomposing language models with dictionary learning* (Transformer Circuits Thread, 2023) — https://transformer-circuits.pub/2023/monosemantic-features
- Templeton et al. (Anthropic), *Scaling monosemanticity: extracting interpretable features from Claude 3 Sonnet* (Transformer Circuits, 2024) — https://transformer-circuits.pub/2024/scaling-monosemanticity
- Elhage et al., *Toy models of superposition* (2022). arXiv:2209.10652 — https://arxiv.org/abs/2209.10652

**Causal interventions**
- Meng et al., *Locating and editing factual associations in GPT* (ROME; NeurIPS 2022). arXiv:2202.05262 — https://arxiv.org/abs/2202.05262
- Geiger et al., *Causal abstraction for faithful model interpretation* (2023). arXiv:2301.04709 — https://arxiv.org/abs/2301.04709

### LoRA and parameter-efficient fine-tuning
- Hu et al., *LoRA: Low-rank adaptation of large language models* (ICLR 2022). arXiv:2106.09685 — https://arxiv.org/abs/2106.09685
- Houlsby et al., *Parameter-efficient transfer learning for NLP* (adapters; ICML 2019). arXiv:1902.00751 — https://arxiv.org/abs/1902.00751

### Classifier-free guidance
- Ho & Salimans, *Classifier-free diffusion guidance* (2022). arXiv:2207.12598 — https://arxiv.org/abs/2207.12598

### MIR / music analysis tooling
- `music21` — https://web.mit.edu/music21/
- `pretty_midi` — https://github.com/craffel/pretty-midi
- `madmom` — https://github.com/CPJKU/madmom
- `mir_eval` — https://github.com/craffel/mir_eval
