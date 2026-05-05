"""Sub-block Peak Energy (SPE) transient features for the note discriminator.

Inspired by Fan et al. "Transient Detection Methods for Audio Coding,"
AES 155th Convention 2023. Adapted here for bass/guitar stem onset scoring:
instead of codec block-switching, we use SPE ratios as scalar features
that tell the discriminator whether a real pluck transient exists near
each basic-pitch detection.

Core idea: high-pass filter the stem, then compare peak amplitude of each
sub-block to the immediately preceding one. A large ratio = real transient.
This is completely timbre-agnostic — it doesn't matter what the bass sounds
like, only whether there is a rapid amplitude rise at the note onset.
"""

import numpy as np
import scipy.signal

# Tuned for demucs-separated bass/guitar stems (800 Hz keeps attack energy
# while rejecting sustained fundamentals; zero_thr kept low since demucs
# stems are quieter than the full mix).
_HPF_HZ         = 800.0
_BLOCK          = 1024          # half-buffer size in samples
_THRESHOLDS     = (0.4, 0.4, 0.07, 0.07)   # per-layer T: curr * T > prev fires
_ZERO_THR       = 0.001
_WINDOW_MS      = 50.0          # ±ms around onset to look for transient


def spe_transients(
    audio: np.ndarray,
    sr: int,
    hpf_hz: float = _HPF_HZ,
    block_size: int = _BLOCK,
    thresholds: tuple = _THRESHOLDS,
) -> tuple:
    """Run SPE on a mono audio array.

    Returns
    -------
    fired_times : (K,) float  — times in seconds where a transient fired
    max_ratios  : (K,) float  — peak curr/prev ratio at each firing block
    """
    nyq = sr / 2.0
    sos = scipy.signal.butter(4, min(hpf_hz / nyq, 0.9999), btype="high", output="sos")
    hp  = scipy.signal.sosfilt(sos, audio.astype(np.float64))

    # Adaptive zero threshold: 5% of HPF-signal RMS so it scales with stem level.
    # The ratio condition (curr*T > prev) is already scale-invariant; only this
    # absolute floor needs to adapt to whatever level demucs outputs.
    rms_hp   = float(np.sqrt(np.mean(hp ** 2)))
    zero_thr = max(rms_hp * 0.05, 1e-6)

    N    = block_size * 2
    half = block_size
    n_blocks = int(np.ceil(len(hp) / half))

    fired_times: list = []
    max_ratios:  list = []
    prev = np.zeros(half, dtype=np.float64)

    for bi in range(n_blocks):
        s = bi * half
        e = min(s + half, len(hp))
        chunk = np.zeros(half, dtype=np.float64)
        chunk[: e - s] = hp[s:e]

        buf = np.concatenate([prev, chunk])

        best_ratio = 0.0
        fired      = False
        for j in range(1, 5):
            sub    = N >> j
            n_curr = 1 << (j - 1)
            T      = thresholds[j - 1]
            prev_peak = float(np.max(np.abs(buf[half - sub: half])))
            for k in range(n_curr):
                curr_peak = float(np.max(np.abs(buf[half + k * sub: half + (k + 1) * sub])))
                ratio = curr_peak / max(prev_peak, 1e-10)
                if ratio > best_ratio:
                    best_ratio = ratio
                if curr_peak > zero_thr and curr_peak * T > prev_peak:
                    fired = True
                prev_peak = curr_peak

        if fired:
            fired_times.append(bi * half / sr)
            max_ratios.append(best_ratio)

        prev = chunk

    return np.array(fired_times, dtype=np.float32), np.array(max_ratios, dtype=np.float32)


def spe_note_features(
    audio: np.ndarray,
    sr: int,
    onsets_s: np.ndarray,
    window_ms: float = _WINDOW_MS,
) -> np.ndarray:
    """Compute 3 SPE-based features per note onset.

    Parameters
    ----------
    audio    : mono float32 audio array (raw stem, before any augmentation)
    sr       : sample rate
    onsets_s : (N,) onset times in seconds

    Returns
    -------
    (N, 3) float32 array:
      col 0  spe_fired       — 1.0 if a transient fired within window_ms
      col 1  spe_max_ratio   — peak curr/prev ratio at nearest transient (0 if none)
      col 2  spe_nearest_norm — normalised distance to nearest transient
                                (0 = at onset, 1 = at or beyond window)
    """
    fired_times, ratios = spe_transients(audio, sr)
    window_s = window_ms / 1000.0
    out = np.zeros((len(onsets_s), 3), dtype=np.float32)

    for i, onset in enumerate(onsets_s):
        if len(fired_times) == 0:
            out[i] = [0.0, 0.0, 1.0]
            continue
        dists = np.abs(fired_times - onset)
        idx   = int(np.argmin(dists))
        dist  = float(dists[idx])
        if dist <= window_s:
            out[i] = [1.0, float(ratios[idx]), dist / window_s]
        else:
            out[i] = [0.0, 0.0, 1.0]

    return out
