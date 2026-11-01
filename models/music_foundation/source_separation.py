"""Music source separation: mask estimation, Wiener filtering, U-Net, HPSS, REPET.

Implements the core signal-separation primitives needed by a music foundation
model — from classical spectrogram-masking methods to learned mask estimators.

Methods:
  - STFT / iSTFT (overlap-add with Hann window)
  - Ideal binary mask (IBM) and ideal ratio mask (IRM)
  - Feedforward mask estimator (2-layer MLP, He init)
  - Iterative Wiener filtering (EM-style refinement)
  - Simplified U-Net for spectrogram-domain separation
  - Harmonic-percussive separation via median filtering (Fitzgerald 2010)
  - REPET: repetition-based separation (Rafii & Pardo 2013)
  - BSS_eval metrics: SDR, SIR, SAR (Vincent et al. 2006)

References:
  - Fitzgerald (2010), Harmonic/Percussive Separation using Median Filtering
  - Rafii & Pardo (2013), REPeating Pattern Extraction Technique
  - Vincent et al. (2006), Performance measurement in blind audio source separation

Pure numpy/scipy. No external ML frameworks.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── configuration ──────────────────────────────────────────────────────

@dataclass
class SeparationConfig:
    n_sources: int = 4
    source_names: List[str] = field(
        default_factory=lambda: ["vocals", "drums", "bass", "other"]
    )
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    mask_type: str = "soft"      # "soft" (IRM) or "binary" (IBM)
    wiener_iterations: int = 5
    mask_estimator_hidden: int = 512
    sample_rate: int = 22050


# ── STFT utilities ─────────────────────────────────────────────────────

def _periodic_hann(n: int) -> np.ndarray:
    """Periodic Hann window (COLA-compliant for hop_length = n/4)."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


def stft(signal: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Short-time Fourier transform with periodic Hann window.

    Pads by n_fft // 2 on both sides (centered framing) so the first
    frame is centered at sample 0 and iSTFT can perfectly reconstruct.

    Args:
        signal: (N,) mono audio
        n_fft: FFT size
        hop_length: hop between frames

    Returns:
        Complex spectrogram of shape (n_fft // 2 + 1, T)
    """
    window = _periodic_hann(n_fft)
    half = n_fft // 2
    sig = np.pad(signal, (half, half))
    n_frames = 1 + (len(sig) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        sig,
        shape=(n_frames, n_fft),
        strides=(sig.strides[0] * hop_length, sig.strides[0]),
    ).copy()
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)  # (T, F)
    return spectrum.T  # (F, T)


def istft(spec: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
          length: Optional[int] = None) -> np.ndarray:
    """Inverse STFT via overlap-add synthesis.

    Uses the COLA-normalized weighted overlap-add:
        x[n] = Σ_t w[n - tH] * IFFT(X_t)[n - tH] / Σ_t w²[n - tH]

    Args:
        spec: Complex spectrogram (F, T) where F = n_fft // 2 + 1
        n_fft: FFT size
        hop_length: hop between frames
        length: desired output length (truncated/padded)

    Returns:
        Reconstructed signal (N,)
    """
    window = _periodic_hann(n_fft)
    half = n_fft // 2
    n_frames = spec.shape[1]
    out_len = n_fft + (n_frames - 1) * hop_length
    output = np.zeros(out_len)
    norm = np.zeros(out_len)

    for t in range(n_frames):
        frame = np.fft.irfft(spec[:, t], n=n_fft)
        start = t * hop_length
        output[start:start + n_fft] += frame * window
        norm[start:start + n_fft] += window ** 2

    # COLA normalization — only divide where we have coverage
    valid = norm > 1e-8
    output[valid] /= norm[valid]

    # remove the centering padding added in stft
    output = output[half:]

    if length is not None:
        if len(output) >= length:
            output = output[:length]
        else:
            output = np.pad(output, (0, length - len(output)))
    return output


# ── ideal masks ────────────────────────────────────────────────────────

def ideal_binary_mask(source_mags: List[np.ndarray]) -> List[np.ndarray]:
    """Ideal binary mask — each T-F bin assigned to the loudest source.

    Args:
        source_mags: list of K magnitude spectrograms, each (F, T)

    Returns:
        list of K binary masks (F, T), values in {0, 1}
    """
    stacked = np.stack(source_mags, axis=0)  # (K, F, T)
    winners = np.argmax(stacked, axis=0)     # (F, T)
    return [(winners == k).astype(np.float64) for k in range(len(source_mags))]


def ideal_ratio_mask(source_mags: List[np.ndarray],
                     power: float = 2.0) -> List[np.ndarray]:
    """Ideal ratio mask (Wiener-style soft mask).

    mask_k = |S_k|^p / Σ_j |S_j|^p

    Args:
        source_mags: list of K magnitude spectrograms (F, T)
        power: exponent (2.0 = Wiener mask, 1.0 = amplitude ratio)

    Returns:
        list of K soft masks (F, T), values in [0, 1]
    """
    powered = [np.abs(s) ** power for s in source_mags]
    total = sum(powered)
    total = np.maximum(total, 1e-10)
    return [p / total for p in powered]


# ── feedforward mask estimator ─────────────────────────────────────────

class MaskEstimator:
    """Two-layer MLP that predicts per-source masks from magnitude frames.

    Input: magnitude spectrogram frame (F,)
    Output: n_sources masks each of shape (F,), values in [0, 1]
    """

    def __init__(self, n_freq: int, n_sources: int = 4,
                 hidden: int = 512, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n_freq = n_freq
        self.n_sources = n_sources

        # He initialization
        self.W1 = rng.randn(n_freq, hidden) * np.sqrt(2.0 / n_freq)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = rng.randn(hidden, n_freq * n_sources) * np.sqrt(2.0 / hidden)
        self.b3 = np.zeros(n_freq * n_sources)

    def forward(self, mag_frame: np.ndarray) -> np.ndarray:
        """Predict masks for a single frame.

        Args:
            mag_frame: (F,) magnitude values

        Returns:
            (n_sources, F) mask values in [0, 1]
        """
        h = mag_frame @ self.W1 + self.b1
        h = np.maximum(h, 0)  # ReLU
        h = h @ self.W2 + self.b2
        h = np.maximum(h, 0)  # ReLU
        out = h @ self.W3 + self.b3
        # sigmoid activation
        out = 1.0 / (1.0 + np.exp(-np.clip(out, -30, 30)))
        return out.reshape(self.n_sources, self.n_freq)

    def estimate_masks(self, magnitude: np.ndarray) -> List[np.ndarray]:
        """Estimate masks for full spectrogram.

        Args:
            magnitude: (F, T) magnitude spectrogram

        Returns:
            list of n_sources masks, each (F, T)
        """
        F, T = magnitude.shape
        masks = [np.zeros((F, T)) for _ in range(self.n_sources)]
        for t in range(T):
            frame_masks = self.forward(magnitude[:, t])  # (n_sources, F)
            for k in range(self.n_sources):
                masks[k][:, t] = frame_masks[k]
        return masks


# ── Wiener filter ──────────────────────────────────────────────────────

class WienerFilter:
    """Iterative Wiener filtering for source refinement.

    Given initial masks and mixture spectrogram, iteratively refine
    source estimates via the EM-style update:
        mask_k = |estimate_k|^2 / Σ_j |estimate_j|^2
    """

    def __init__(self, n_iterations: int = 5):
        self.n_iterations = n_iterations

    def filter(self, mixture_spec: np.ndarray,
               initial_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Refine source separation masks.

        Args:
            mixture_spec: complex mixture spectrogram (F, T)
            initial_masks: list of K initial masks (F, T)

        Returns:
            list of K refined complex source estimates (F, T)
        """
        mag = np.abs(mixture_spec)
        estimates = [mask * mixture_spec for mask in initial_masks]

        for _ in range(self.n_iterations):
            powers = [np.abs(est) ** 2 for est in estimates]
            total_power = sum(powers)
            total_power = np.maximum(total_power, 1e-10)
            masks = [p / total_power for p in powers]
            estimates = [m * mixture_spec for m in masks]

        return estimates


# ── batch normalization (for U-Net) ────────────────────────────────────

class BatchNorm2D:
    """Batch normalization for 2D feature maps."""

    def __init__(self, n_channels: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma = np.ones(n_channels)
        self.beta = np.zeros(n_channels)
        self.running_mean = np.zeros(n_channels)
        self.running_var = np.ones(n_channels)
        self.momentum = momentum
        self.eps = eps
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (C, H, W)"""
        if self.training:
            mean = x.mean(axis=(1, 2))
            var = x.var(axis=(1, 2))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean[:, None, None]) / np.sqrt(var[:, None, None] + self.eps)
        return self.gamma[:, None, None] * x_norm + self.beta[:, None, None]


# ── 2D convolution primitives ──────────────────────────────────────────

def _conv2d(x: np.ndarray, W: np.ndarray, b: np.ndarray,
            pad: int = 1) -> np.ndarray:
    """2D convolution via im2col-style loops (simple, pure numpy).

    Args:
        x: (C_in, H, W)
        W: (C_out, C_in, kH, kW)
        b: (C_out,)
        pad: zero-padding

    Returns:
        (C_out, H_out, W_out)
    """
    C_in, H, Wi = x.shape
    C_out, _, kH, kW = W.shape
    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    H_p, W_p = x.shape[1], x.shape[2]
    H_out = H_p - kH + 1
    W_out = W_p - kW + 1
    out = np.zeros((C_out, H_out, W_out))
    for co in range(C_out):
        for ci in range(C_in):
            for kh in range(kH):
                for kw in range(kW):
                    out[co] += W[co, ci, kh, kw] * x[ci, kh:kh + H_out, kw:kw + W_out]
        out[co] += b[co]
    return out


def _max_pool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """2x2 max pooling.

    Args:
        x: (C, H, W) where H and W are divisible by pool_size

    Returns:
        (C, H // pool_size, W // pool_size)
    """
    C, H, W = x.shape
    # trim to multiple of pool_size
    H_t = (H // pool_size) * pool_size
    W_t = (W // pool_size) * pool_size
    x = x[:, :H_t, :W_t]
    reshaped = x.reshape(C, H_t // pool_size, pool_size, W_t // pool_size, pool_size)
    return reshaped.max(axis=(2, 4))


def _upsample2d(x: np.ndarray, scale: int = 2) -> np.ndarray:
    """Nearest-neighbor upsampling.

    Args:
        x: (C, H, W)

    Returns:
        (C, H * scale, W * scale)
    """
    return x.repeat(scale, axis=1).repeat(scale, axis=2)


# ── U-Net separator ───────────────────────────────────────────────────

class UNetSeparator:
    """Simplified U-Net for spectrogram-domain source separation.

    Encoder: 4 blocks of Conv2D → BN → ReLU → MaxPool
    Bottleneck: Conv2D → ReLU
    Decoder: 4 blocks of Upsample → ConcatSkip → Conv2D → BN → ReLU
    Final: Conv2D → Sigmoid → n_sources masks
    """

    def __init__(self, n_sources: int = 4, base_channels: int = 16,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n_sources = n_sources

        channels = [1, base_channels, base_channels * 2,
                    base_channels * 4, base_channels * 8]
        bottleneck_ch = base_channels * 16

        self.enc_convs = []
        self.enc_bns = []
        for i in range(4):
            c_in, c_out = channels[i], channels[i + 1]
            W = rng.randn(c_out, c_in, 3, 3) * np.sqrt(2.0 / (c_in * 9))
            b = np.zeros(c_out)
            self.enc_convs.append((W, b))
            self.enc_bns.append(BatchNorm2D(c_out))

        # bottleneck
        c_in = channels[4]
        self.bn_W = rng.randn(bottleneck_ch, c_in, 3, 3) * np.sqrt(2.0 / (c_in * 9))
        self.bn_b = np.zeros(bottleneck_ch)

        self.dec_convs = []
        self.dec_bns = []
        dec_channels = list(reversed(channels[1:]))  # [8*b, 4*b, 2*b, b]
        dec_in_channels = [bottleneck_ch + channels[4],
                           dec_channels[0] + channels[3],
                           dec_channels[1] + channels[2],
                           dec_channels[2] + channels[1]]
        for i in range(4):
            c_in = dec_in_channels[i]
            c_out = dec_channels[i] if i < 3 else dec_channels[3]
            W = rng.randn(c_out, c_in, 3, 3) * np.sqrt(2.0 / (c_in * 9))
            b = np.zeros(c_out)
            self.dec_convs.append((W, b))
            self.dec_bns.append(BatchNorm2D(c_out))

        # final 1x1 conv → n_sources masks
        final_in = dec_channels[3]
        self.final_W = rng.randn(n_sources, final_in, 1, 1) * np.sqrt(2.0 / final_in)
        self.final_b = np.zeros(n_sources)

    def forward(self, magnitude: np.ndarray) -> List[np.ndarray]:
        """Predict n_sources masks from magnitude spectrogram.

        Args:
            magnitude: (F, T) magnitude spectrogram

        Returns:
            list of n_sources masks, each (F, T)
        """
        F, T = magnitude.shape
        # pad to multiples of 16
        pad_f = (16 - F % 16) % 16
        pad_t = (16 - T % 16) % 16
        x = np.pad(magnitude, ((0, pad_f), (0, pad_t)))
        x = x[np.newaxis, :, :]  # (1, F', T')

        skips = []
        for i in range(4):
            W, b = self.enc_convs[i]
            x = _conv2d(x, W, b, pad=1)
            x = self.enc_bns[i].forward(x)
            x = np.maximum(x, 0)  # ReLU
            skips.append(x)
            x = _max_pool2d(x, 2)

        # bottleneck
        x = _conv2d(x, self.bn_W, self.bn_b, pad=1)
        x = np.maximum(x, 0)

        for i in range(4):
            x = _upsample2d(x, 2)
            skip = skips[3 - i]
            # match shapes
            min_h = min(x.shape[1], skip.shape[1])
            min_w = min(x.shape[2], skip.shape[2])
            x = np.concatenate([x[:, :min_h, :min_w],
                                skip[:, :min_h, :min_w]], axis=0)
            W, b = self.dec_convs[i]
            x = _conv2d(x, W, b, pad=1)
            x = self.dec_bns[i].forward(x)
            x = np.maximum(x, 0)

        # final 1x1 conv
        masks_raw = _conv2d(x, self.final_W, self.final_b, pad=0)
        # sigmoid
        masks_raw = 1.0 / (1.0 + np.exp(-np.clip(masks_raw, -30, 30)))

        # extract per-source masks, crop to original size
        return [masks_raw[k, :F, :T] for k in range(self.n_sources)]


# ── harmonic-percussive separation ─────────────────────────────────────

class HarmonicPercussiveSeparation:
    """Harmonic/percussive separation via median filtering (Fitzgerald 2010).

    Harmonic components are stable across time (horizontal continuity),
    percussive components are stable across frequency (vertical impulse).
    Median filtering along each axis extracts the respective component.
    """

    def __init__(self, harmonic_kernel: int = 31, percussive_kernel: int = 31,
                 margin_harmonic: float = 1.0, margin_percussive: float = 1.0):
        self.harmonic_kernel = harmonic_kernel
        self.percussive_kernel = percussive_kernel
        self.margin_harmonic = margin_harmonic
        self.margin_percussive = margin_percussive

    def _median_filter_horizontal(self, S: np.ndarray, k: int) -> np.ndarray:
        """Median filter along time axis (axis=1)."""
        F, T = S.shape
        pad = k // 2
        padded = np.pad(S, ((0, 0), (pad, pad)), mode='reflect')
        out = np.zeros_like(S)
        for t in range(T):
            out[:, t] = np.median(padded[:, t:t + k], axis=1)
        return out

    def _median_filter_vertical(self, S: np.ndarray, k: int) -> np.ndarray:
        """Median filter along frequency axis (axis=0)."""
        F, T = S.shape
        pad = k // 2
        padded = np.pad(S, ((pad, pad), (0, 0)), mode='reflect')
        out = np.zeros_like(S)
        for f in range(F):
            out[f, :] = np.median(padded[f:f + k, :], axis=0)
        return out

    def separate(self, signal: np.ndarray, n_fft: int = 2048,
                 hop_length: int = 512) -> Dict[str, np.ndarray]:
        """Separate signal into harmonic and percussive components.

        Returns dict with keys 'harmonic', 'percussive', and 'residual'.
        """
        spec = stft(signal, n_fft, hop_length)
        mag = np.abs(spec)
        phase = np.angle(spec)

        H = self._median_filter_horizontal(mag, self.harmonic_kernel)
        P = self._median_filter_vertical(mag, self.percussive_kernel)

        # soft masks with margin
        Hm = H ** (2 * self.margin_harmonic)
        Pm = P ** (2 * self.margin_percussive)
        total = Hm + Pm
        total = np.maximum(total, 1e-10)

        mask_h = Hm / total
        mask_p = Pm / total

        harm_spec = mask_h * mag * np.exp(1j * phase)
        perc_spec = mask_p * mag * np.exp(1j * phase)

        harmonic = istft(harm_spec, n_fft, hop_length, length=len(signal))
        percussive = istft(perc_spec, n_fft, hop_length, length=len(signal))
        residual = signal - harmonic - percussive

        return {
            "harmonic": harmonic,
            "percussive": percussive,
            "residual": residual,
            "mask_harmonic": mask_h,
            "mask_percussive": mask_p,
        }


# ── REPET separation ──────────────────────────────────────────────────

class RepetitionSeparation:
    """REPET: REPeating Pattern Extraction Technique (Rafii & Pardo 2013).

    Finds the repeating period in the mixture via autocorrelation of the
    beat spectrum, builds a repeating spectrogram model, and applies a
    soft mask to separate repeating (accompaniment) from non-repeating
    (vocals/foreground).
    """

    def __init__(self, min_period: float = 0.8, max_period: float = 8.0):
        self.min_period = min_period
        self.max_period = max_period

    def _beat_spectrum(self, mag: np.ndarray) -> np.ndarray:
        """Beat spectrum via autocorrelation of the power spectrogram."""
        power = mag ** 2
        # autocorrelation along time for each frequency, then sum
        T = power.shape[1]
        beat = np.zeros(T)
        for lag in range(T):
            if lag == 0:
                beat[0] = np.sum(power ** 2)
            else:
                beat[lag] = np.sum(power[:, :T - lag] * power[:, lag:])
        return beat / (beat[0] + 1e-10)

    def _find_period(self, beat: np.ndarray, sr: int,
                     hop_length: int) -> int:
        """Find repeating period from beat spectrum peaks."""
        min_lag = max(1, int(self.min_period * sr / hop_length))
        max_lag = min(len(beat) - 1, int(self.max_period * sr / hop_length))
        if min_lag >= max_lag:
            return max(1, min_lag)

        segment = beat[min_lag:max_lag + 1]
        peak_idx = np.argmax(segment)
        return min_lag + peak_idx

    def separate(self, signal: np.ndarray, sr: int = 22050,
                 n_fft: int = 2048, hop_length: int = 512) -> Dict[str, np.ndarray]:
        """Separate repeating background from non-repeating foreground.

        Returns dict with keys 'repeating' (accompaniment) and
        'non_repeating' (vocals/foreground).
        """
        spec = stft(signal, n_fft, hop_length)
        mag = np.abs(spec)
        phase = np.angle(spec)

        beat = self._beat_spectrum(mag)
        period = self._find_period(beat, sr, hop_length)

        # build repeating model: median of period-aligned frames
        F, T = mag.shape
        n_reps = T // period
        if n_reps < 2:
            return {
                "repeating": np.zeros_like(signal),
                "non_repeating": signal.copy(),
                "period_frames": period,
                "period_seconds": period * hop_length / sr,
            }

        # stack period-aligned segments and take median
        rep_model = np.zeros((F, period))
        for f in range(F):
            segments = []
            for r in range(n_reps):
                start = r * period
                segments.append(mag[f, start:start + period])
            rep_model[f] = np.median(np.array(segments), axis=0)

        # tile the model to cover the full spectrogram
        rep_full = np.tile(rep_model, (1, (T // period) + 1))[:, :T]

        # soft mask: min(rep_model, mixture) / mixture
        rep_mag = np.minimum(rep_full, mag)
        mask = rep_mag / np.maximum(mag, 1e-10)

        rep_spec = mask * mag * np.exp(1j * phase)
        fg_spec = (1 - mask) * mag * np.exp(1j * phase)

        repeating = istft(rep_spec, n_fft, hop_length, length=len(signal))
        non_repeating = istft(fg_spec, n_fft, hop_length, length=len(signal))

        return {
            "repeating": repeating,
            "non_repeating": non_repeating,
            "period_frames": period,
            "period_seconds": period * hop_length / sr,
            "beat_spectrum": beat,
        }


# ── BSS_eval metrics ──────────────────────────────────────────────────

def _project(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Project estimate onto reference subspace."""
    if np.dot(reference, reference) < 1e-10:
        return np.zeros_like(reference)
    return np.dot(estimate, reference) / np.dot(reference, reference) * reference


def evaluate_bss(reference: np.ndarray, estimate: np.ndarray,
                 other_references: Optional[List[np.ndarray]] = None
                 ) -> Dict[str, float]:
    """BSS_eval metrics: SDR, SIR, SAR (Vincent et al. 2006).

    Decomposition: estimate = s_target + e_interf + e_noise + e_artif
    SDR = 10 log10(||s_target||^2 / ||e_interf + e_noise + e_artif||^2)
    SIR = 10 log10(||s_target||^2 / ||e_interf||^2)
    SAR = 10 log10(||s_target + e_interf||^2 / ||e_artif||^2)

    Args:
        reference: (N,) target source signal
        estimate: (N,) estimated source signal
        other_references: list of (N,) other source signals (for interference)

    Returns:
        dict with 'sdr', 'sir', 'sar' in dB
    """
    min_len = min(len(reference), len(estimate))
    ref = reference[:min_len].astype(np.float64)
    est = estimate[:min_len].astype(np.float64)

    # s_target: projection of estimate onto reference
    s_target = _project(ref, est)

    if other_references and len(other_references) > 0:
        # build interference subspace: [ref, other1, other2, ...]
        all_refs = [ref] + [r[:min_len].astype(np.float64) for r in other_references]
        # project onto full source subspace
        # simple Gram-Schmidt-style projection
        s_sources = np.zeros_like(est)
        for r in all_refs:
            s_sources += _project(r, est)
        e_interf = s_sources - s_target
    else:
        e_interf = np.zeros_like(est)

    e_total = est - s_target  # total error
    e_artif = e_total - e_interf

    def _ratio_db(signal: np.ndarray, noise: np.ndarray) -> float:
        s_energy = np.sum(signal ** 2)
        n_energy = np.sum(noise ** 2)
        if n_energy < 1e-10:
            return 100.0  # effectively infinite
        if s_energy < 1e-10:
            return -100.0
        return 10.0 * np.log10(s_energy / n_energy)

    sdr = _ratio_db(s_target, e_total)
    sir = _ratio_db(s_target, e_interf) if np.sum(e_interf ** 2) > 1e-10 else 100.0
    sar = _ratio_db(s_target + e_interf, e_artif) if np.sum(e_artif ** 2) > 1e-10 else 100.0

    return {"sdr": sdr, "sir": sir, "sar": sar}


def evaluate_separation_batch(
    estimates: Dict[str, np.ndarray],
    references: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Evaluate BSS metrics for all sources.

    Args:
        estimates: {source_name: estimated_signal}
        references: {source_name: reference_signal}

    Returns:
        {source_name: {"sdr": ..., "sir": ..., "sar": ...}}
    """
    results = {}
    source_names = list(references.keys())
    for name in source_names:
        if name not in estimates:
            continue
        others = [references[n] for n in source_names if n != name]
        results[name] = evaluate_bss(references[name], estimates[name], others)
    return results


# ── high-level separator API ───────────────────────────────────────────

class SourceSeparator:
    """High-level source separation combining multiple methods.

    Methods:
      - "mask_estimator": MLP-based mask estimation
      - "wiener": mask estimation + Wiener refinement
      - "unet": U-Net spectrogram-domain separation
      - "hpss": harmonic-percussive separation (2 sources only)
      - "repet": repetition-based separation (2 sources only)
    """

    def __init__(self, config: Optional[SeparationConfig] = None,
                 seed: int = 42):
        self.config = config or SeparationConfig()
        self.seed = seed
        self._mask_est: Optional[MaskEstimator] = None
        self._wiener: Optional[WienerFilter] = None
        self._unet: Optional[UNetSeparator] = None
        self._hpss: Optional[HarmonicPercussiveSeparation] = None
        self._repet: Optional[RepetitionSeparation] = None

    def _get_mask_estimator(self) -> MaskEstimator:
        if self._mask_est is None:
            n_freq = self.config.n_fft // 2 + 1
            self._mask_est = MaskEstimator(
                n_freq, self.config.n_sources,
                self.config.mask_estimator_hidden, self.seed,
            )
        return self._mask_est

    def _get_wiener(self) -> WienerFilter:
        if self._wiener is None:
            self._wiener = WienerFilter(self.config.wiener_iterations)
        return self._wiener

    def _get_unet(self) -> UNetSeparator:
        if self._unet is None:
            self._unet = UNetSeparator(self.config.n_sources, seed=self.seed)
        return self._unet

    def separate(self, mixture: np.ndarray,
                 method: str = "mask_estimator") -> Dict[str, np.ndarray]:
        """Separate mixture into source signals.

        Args:
            mixture: (N,) mono audio signal
            method: separation method

        Returns:
            dict mapping source names to estimated signals
        """
        n_fft = self.config.n_fft
        hop = self.config.hop_length
        names = self.config.source_names

        if method == "hpss":
            if self._hpss is None:
                self._hpss = HarmonicPercussiveSeparation()
            result = self._hpss.separate(mixture, n_fft, hop)
            return {"harmonic": result["harmonic"],
                    "percussive": result["percussive"]}

        if method == "repet":
            if self._repet is None:
                self._repet = RepetitionSeparation()
            result = self._repet.separate(
                mixture, self.config.sample_rate, n_fft, hop)
            return {"repeating": result["repeating"],
                    "non_repeating": result["non_repeating"]}

        spec = stft(mixture, n_fft, hop)
        mag = np.abs(spec)

        if method == "mask_estimator":
            est = self._get_mask_estimator()
            masks = est.estimate_masks(mag)
        elif method == "wiener":
            est = self._get_mask_estimator()
            initial_masks = est.estimate_masks(mag)
            wiener = self._get_wiener()
            estimates = wiener.filter(spec, initial_masks)
            result = {}
            for k, name in enumerate(names[:len(estimates)]):
                result[name] = istft(estimates[k], n_fft, hop, length=len(mixture))
            return result
        elif method == "unet":
            unet = self._get_unet()
            masks = unet.forward(mag)
        else:
            raise ValueError(f"Unknown method: {method}")

        phase = np.angle(spec)
        result = {}
        for k, name in enumerate(names[:len(masks)]):
            source_spec = masks[k] * mag * np.exp(1j * phase)
            result[name] = istft(source_spec, n_fft, hop, length=len(mixture))
        return result

    def separate_and_remix(self, mixture: np.ndarray,
                           gains: Dict[str, float],
                           method: str = "mask_estimator") -> np.ndarray:
        """Separate and remix with per-source gain control.

        Args:
            mixture: (N,) mono audio
            gains: {source_name: gain_factor}
            method: separation method

        Returns:
            Remixed signal (N,)
        """
        sources = self.separate(mixture, method)
        remixed = np.zeros_like(mixture)
        for name, signal in sources.items():
            gain = gains.get(name, 1.0)
            remixed += gain * signal
        return remixed

    def evaluate_separation(
        self, estimates: Dict[str, np.ndarray],
        references: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate separation quality via BSS_eval metrics."""
        return evaluate_separation_batch(estimates, references)


# ── self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # synthetic sources: pure tones at different frequencies
    src_low = 0.5 * np.sin(2 * np.pi * 110 * t)     # bass (A2)
    src_mid = 0.4 * np.sin(2 * np.pi * 440 * t)     # mid (A4)
    src_high = 0.3 * np.sin(2 * np.pi * 1760 * t)   # high (A6)
    src_perc = np.zeros_like(t)
    # percussive clicks every 0.25s
    for onset in np.arange(0, duration, 0.25):
        idx = int(onset * sr)
        src_perc[idx:idx + 200] = 0.6 * np.random.randn(min(200, len(t) - idx))

    mixture = src_low + src_mid + src_high + src_perc

    print("=== Source Separation Self-Test ===\n")

    # 1. STFT round-trip
    spec = stft(mixture, 2048, 512)
    recon = istft(spec, 2048, 512, length=len(mixture))
    rms_err = np.sqrt(np.mean((mixture - recon) ** 2))
    print(f"STFT round-trip RMS error: {rms_err:.2e}")
    assert rms_err < 1e-6, "STFT round-trip failed"

    # 2. Ideal masks
    mags = [np.abs(stft(s, 2048, 512)) for s in [src_low, src_mid, src_high, src_perc]]
    ibm = ideal_binary_mask(mags)
    irm = ideal_ratio_mask(mags)
    print(f"IBM masks: {len(ibm)}, shapes: {ibm[0].shape}")
    print(f"IRM masks: {len(irm)}, sum check: {sum(m.mean() for m in irm):.4f}")

    # 3. Mask estimator
    n_freq = 2048 // 2 + 1
    est = MaskEstimator(n_freq, n_sources=4, hidden=128, seed=42)
    mag = np.abs(stft(mixture, 2048, 512))
    masks = est.estimate_masks(mag)
    print(f"Mask estimator: {len(masks)} masks, shape {masks[0].shape}")

    # 4. HPSS
    hpss = HarmonicPercussiveSeparation()
    hpss_result = hpss.separate(mixture)
    print(f"HPSS: harmonic={hpss_result['harmonic'].shape}, "
          f"percussive={hpss_result['percussive'].shape}")

    # 5. REPET
    repet = RepetitionSeparation()
    repet_result = repet.separate(mixture, sr=sr)
    print(f"REPET: period={repet_result['period_seconds']:.3f}s, "
          f"repeating={repet_result['repeating'].shape}")

    # 6. BSS_eval on ideal separation
    ref_spec = stft(src_low, 2048, 512)
    est_spec = irm[0] * np.abs(stft(mixture, 2048, 512)) * np.exp(1j * np.angle(stft(mixture, 2048, 512)))
    est_sig = istft(est_spec, 2048, 512, length=len(mixture))
    bss = evaluate_bss(src_low, est_sig, [src_mid, src_high, src_perc])
    print(f"BSS_eval (IRM bass): SDR={bss['sdr']:.1f}dB, "
          f"SIR={bss['sir']:.1f}dB, SAR={bss['sar']:.1f}dB")

    # 7. High-level API
    sep = SourceSeparator(SeparationConfig(n_sources=4, n_fft=2048, hop_length=512))
    sources = sep.separate(mixture, method="mask_estimator")
    print(f"SourceSeparator: {list(sources.keys())}")

    remix = sep.separate_and_remix(
        mixture, {"vocals": 0.0, "drums": 1.0, "bass": 1.5, "other": 0.5})
    print(f"Remix shape: {remix.shape}")

    print("\n=== All source separation tests passed ===")
