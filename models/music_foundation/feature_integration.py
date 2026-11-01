"""Multi-scale feature integration for the frontier music model.

Fuses heterogeneous audio features (mel, CQT, chroma, MFCC, spectral
descriptors, onset strength, pitch contours) into a unified representation
suitable for the transformer encoder.

Architecture:
    Each feature stream → per-stream linear projection → temporal alignment
    → gated fusion (FiLM or attention-based) → combined representation

This lets the frontier model attend over musically meaningful features
alongside raw spectrogram features, analogous to MERT's multi-scale
pre-training but with explicit feature engineering integration.

References:
    - Li et al. (2023). MERT: Acoustic Music Understanding Model with
      Large-Scale Self-Supervised Training.
    - McFee et al. (2015). librosa: Audio and Music Signal Analysis in Python.
    - Perez et al. (2018). FiLM: Visual Reasoning with a General Conditioning
      Layer.

Pure numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FeatureStreamConfig:
    """Configuration for a single feature stream."""
    name: str
    input_dim: int
    output_dim: int = 256
    temporal_pool: str = "none"  # "none", "mean", "max", "stride"
    stride: int = 1


@dataclass
class FeatureIntegrationConfig:
    """Configuration for multi-stream feature fusion."""
    d_model: int = 256
    n_heads: int = 4
    fusion_method: str = "gated"  # "concat", "add", "gated", "attention"
    gate_activation: str = "sigmoid"
    dropout: float = 0.0
    max_streams: int = 8
    temporal_alignment: str = "interpolate"  # "interpolate", "pad", "crop"
    seed: int = 42


# ---------- feature extractors (pure numpy) ----------

def compute_mfcc(
    signal: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> np.ndarray:
    """Compute MFCCs from raw audio. Returns (n_mfcc, T)."""
    # Mel spectrogram
    n_frames = max(1, 1 + (len(signal) - n_fft) // hop_length)
    window = np.hanning(n_fft)
    power = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frame = np.zeros(n_fft)
        end = min(start + n_fft, len(signal))
        frame[:end - start] = signal[start:end]
        frame *= window
        power[:, i] = np.abs(np.fft.rfft(frame)) ** 2

    # Mel filterbank
    mel_high = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_pts = np.linspace(0, mel_high, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            if bins[m + 1] > bins[m]:
                fb[m, k] = (k - bins[m]) / (bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            if bins[m + 2] > bins[m + 1]:
                fb[m, k] = (bins[m + 2] - k) / (bins[m + 2] - bins[m + 1])

    mel_spec = np.log(fb @ power + 1e-9)

    # DCT-II (type 2) for MFCCs
    N = n_mels
    dct_matrix = np.zeros((n_mfcc, N))
    for k in range(n_mfcc):
        for n in range(N):
            dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    dct_matrix *= np.sqrt(2.0 / N)
    dct_matrix[0, :] /= np.sqrt(2.0)

    return dct_matrix @ mel_spec


def compute_chroma(
    signal: np.ndarray,
    sr: int = 22050,
    n_fft: int = 4096,
    hop_length: int = 512,
    n_chroma: int = 12,
) -> np.ndarray:
    """Compute chromagram from raw audio. Returns (12, T)."""
    n_frames = max(1, 1 + (len(signal) - n_fft) // hop_length)
    window = np.hanning(n_fft)
    power = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frame = np.zeros(n_fft)
        end = min(start + n_fft, len(signal))
        frame[:end - start] = signal[start:end]
        frame *= window
        power[:, i] = np.abs(np.fft.rfft(frame)) ** 2

    # Map FFT bins to chroma
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    freqs[0] = 1e-6  # avoid log(0)
    chroma_map = np.round(12.0 * np.log2(freqs / 440.0)) % 12
    chroma_map = chroma_map.astype(int)

    chroma = np.zeros((n_chroma, n_frames))
    for b in range(1, n_fft // 2 + 1):
        chroma[chroma_map[b]] += power[b]

    # Normalize per frame
    norms = chroma.sum(axis=0, keepdims=True) + 1e-10
    return chroma / norms


def compute_spectral_contrast(
    power_spectrum: np.ndarray,
    n_bands: int = 6,
    quantile: float = 0.02,
) -> np.ndarray:
    """Spectral contrast: difference between peaks and valleys per subband.

    Input: (n_freq, T). Returns: (n_bands, T).
    """
    n_freq, T = power_spectrum.shape
    band_edges = np.linspace(0, n_freq, n_bands + 1, dtype=int)
    contrast = np.zeros((n_bands, T))

    for b in range(n_bands):
        lo, hi = band_edges[b], band_edges[b + 1]
        if hi <= lo:
            continue
        band = power_spectrum[lo:hi, :]
        n_q = max(1, int(quantile * (hi - lo)))
        sorted_band = np.sort(band, axis=0)
        valley = sorted_band[:n_q].mean(axis=0)
        peak = sorted_band[-n_q:].mean(axis=0)
        contrast[b] = np.log(peak + 1e-10) - np.log(valley + 1e-10)

    return contrast


def compute_onset_strength(
    signal: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Onset strength envelope via spectral flux. Returns (T,)."""
    n_frames = max(1, 1 + (len(signal) - n_fft) // hop_length)
    window = np.hanning(n_fft)
    mag = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frame = np.zeros(n_fft)
        end = min(start + n_fft, len(signal))
        frame[:end - start] = signal[start:end]
        frame *= window
        mag[:, i] = np.abs(np.fft.rfft(frame))

    # Spectral flux (half-wave rectified first-order difference)
    flux = np.zeros(n_frames)
    for i in range(1, n_frames):
        diff = mag[:, i] - mag[:, i - 1]
        flux[i] = np.sum(np.maximum(diff, 0))

    # Normalize
    peak = flux.max()
    if peak > 1e-10:
        flux /= peak
    return flux


def compute_spectral_centroid(
    power_spectrum: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
) -> np.ndarray:
    """Spectral centroid per frame. Returns (T,)."""
    n_freq, T = power_spectrum.shape
    freqs = np.arange(n_freq) * sr / n_fft
    total = power_spectrum.sum(axis=0) + 1e-10
    centroid = (freqs[:, np.newaxis] * power_spectrum).sum(axis=0) / total
    return centroid


def compute_spectral_bandwidth(
    power_spectrum: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
) -> np.ndarray:
    """Spectral bandwidth (weighted std of frequencies). Returns (T,)."""
    n_freq, T = power_spectrum.shape
    freqs = np.arange(n_freq) * sr / n_fft
    total = power_spectrum.sum(axis=0) + 1e-10
    centroid = (freqs[:, np.newaxis] * power_spectrum).sum(axis=0) / total
    deviation = freqs[:, np.newaxis] - centroid[np.newaxis, :]
    bandwidth = np.sqrt((deviation ** 2 * power_spectrum).sum(axis=0) / total)
    return bandwidth


def compute_spectral_rolloff(
    power_spectrum: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    rolloff_percent: float = 0.85,
) -> np.ndarray:
    """Frequency below which rolloff_percent of energy is concentrated. Returns (T,)."""
    n_freq, T = power_spectrum.shape
    freqs = np.arange(n_freq) * sr / n_fft
    cumsum = np.cumsum(power_spectrum, axis=0)
    total = cumsum[-1:, :] + 1e-10
    threshold = rolloff_percent * total
    rolloff = np.zeros(T)
    for t in range(T):
        idx = np.searchsorted(cumsum[:, t], threshold[0, t])
        idx = min(idx, n_freq - 1)
        rolloff[t] = freqs[idx]
    return rolloff


def compute_zero_crossing_rate(
    signal: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Zero-crossing rate per frame. Returns (T,)."""
    n_frames = max(1, 1 + (len(signal) - frame_length) // hop_length)
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(signal))
        frame = signal[start:end]
        if len(frame) < 2:
            continue
        crossings = np.sum(np.abs(np.diff(np.sign(frame))) > 0)
        zcr[i] = crossings / (2.0 * len(frame))
    return zcr


# ---------- temporal alignment ----------

def _align_temporal(
    feature: np.ndarray,
    target_length: int,
    method: str = "interpolate",
) -> np.ndarray:
    """Align a feature array to a target temporal length.

    feature: (D, T) or (T,)
    Returns same shape with T → target_length.
    """
    is_1d = feature.ndim == 1
    if is_1d:
        feature = feature[np.newaxis, :]

    D, T = feature.shape
    if T == target_length:
        return feature.squeeze() if is_1d else feature

    if method == "interpolate":
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_length)
        aligned = np.zeros((D, target_length))
        for d in range(D):
            aligned[d] = np.interp(x_new, x_old, feature[d])
    elif method == "pad":
        if T >= target_length:
            aligned = feature[:, :target_length]
        else:
            aligned = np.zeros((D, target_length))
            aligned[:, :T] = feature
    elif method == "crop":
        if T >= target_length:
            start = (T - target_length) // 2
            aligned = feature[:, start:start + target_length]
        else:
            aligned = np.zeros((D, target_length))
            start = (target_length - T) // 2
            aligned[:, start:start + T] = feature
    else:
        aligned = feature[:, :target_length] if T >= target_length else np.pad(
            feature, ((0, 0), (0, target_length - T))
        )

    return aligned.squeeze() if is_1d else aligned


# ---------- stream projections ----------

class FeatureProjection:
    """Linear projection for a single feature stream."""

    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = rng.randn(input_dim, output_dim) * scale
        self.b = np.zeros(output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project (D_in, T) → (D_out, T)."""
        return (self.W.T @ x) + self.b[:, np.newaxis]


class GatedFusion:
    """Gated fusion of multiple feature streams via learned gates.

    gate_i = σ(W_g · x_i + b_g)
    output = Σ_i gate_i ⊙ x_i
    """

    def __init__(self, d_model: int, n_streams: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (d_model + 1))
        self.W_gates = [rng.randn(d_model, 1) * scale for _ in range(n_streams)]
        self.b_gates = [np.zeros(1) for _ in range(n_streams)]
        self.d_model = d_model
        self.n_streams = n_streams

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def forward(self, streams: List[np.ndarray]) -> np.ndarray:
        """Fuse streams. Each stream: (D, T). Returns (D, T)."""
        gates = []
        for i, stream in enumerate(streams):
            g = self.W_gates[i].T @ stream + self.b_gates[i][:, np.newaxis]
            gates.append(self._sigmoid(g))

        # Normalize gates to sum to 1
        gate_sum = sum(gates)
        gate_sum = np.maximum(gate_sum, 1e-10)
        gates = [g / gate_sum for g in gates]

        output = np.zeros_like(streams[0])
        for g, s in zip(gates, streams):
            output += g * s
        return output


class AttentionFusion:
    """Cross-attention fusion: one stream queries, others provide context.

    Primary stream is typically the mel spectrogram; auxiliary streams
    (chroma, MFCC, onset, etc.) provide musically informed context.
    """

    def __init__(self, d_model: int, n_heads: int = 4, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q = rng.randn(d_model, d_model) * scale
        self.W_k = rng.randn(d_model, d_model) * scale
        self.W_v = rng.randn(d_model, d_model) * scale
        self.W_o = rng.randn(d_model, d_model) * scale
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, query: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Cross-attend. query: (D, T_q), context: (D, T_c). Returns (D, T_q)."""
        T_q = query.shape[1]
        T_c = context.shape[1]
        d_k = self.d_model // self.n_heads

        Q = self.W_q.T @ query  # (D, T_q)
        K = self.W_k.T @ context
        V = self.W_v.T @ context

        # Reshape for multi-head: (H, d_k, T)
        Q_h = Q.reshape(self.n_heads, d_k, T_q)
        K_h = K.reshape(self.n_heads, d_k, T_c)
        V_h = V.reshape(self.n_heads, d_k, T_c)

        # Attention: Q^T K / sqrt(d_k)
        scores = np.zeros((self.n_heads, T_q, T_c))
        for h in range(self.n_heads):
            scores[h] = Q_h[h].T @ K_h[h] / np.sqrt(d_k)

        # Softmax
        scores_stable = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores_stable)
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-10)

        # Weighted sum
        out = np.zeros((self.n_heads, d_k, T_q))
        for h in range(self.n_heads):
            out[h] = V_h[h] @ attn[h].T

        out_cat = out.reshape(self.d_model, T_q)
        return self.W_o.T @ out_cat


# ---------- multi-stream integrator ----------

class MultiStreamFeatureIntegrator:
    """Integrate multiple feature streams into a unified representation.

    Supports heterogeneous features with different dimensionalities
    and temporal resolutions.  All streams are projected to d_model,
    temporally aligned, and fused via the configured method.

    Usage:
        integrator = MultiStreamFeatureIntegrator(config)
        integrator.add_stream("mel", input_dim=128)
        integrator.add_stream("chroma", input_dim=12)
        integrator.add_stream("mfcc", input_dim=13)

        features = {
            "mel": mel_spec,       # (128, T1)
            "chroma": chroma,      # (12, T2)
            "mfcc": mfcc,          # (13, T3)
        }
        unified = integrator.forward(features)  # (d_model, T)
    """

    def __init__(self, config: Optional[FeatureIntegrationConfig] = None):
        self.config = config or FeatureIntegrationConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.streams: Dict[str, FeatureStreamConfig] = {}
        self.projections: Dict[str, FeatureProjection] = {}
        self._fusion = None

    def add_stream(
        self,
        name: str,
        input_dim: int,
        temporal_pool: str = "none",
        stride: int = 1,
    ) -> None:
        cfg = FeatureStreamConfig(
            name=name,
            input_dim=input_dim,
            output_dim=self.config.d_model,
            temporal_pool=temporal_pool,
            stride=stride,
        )
        self.streams[name] = cfg
        self.projections[name] = FeatureProjection(
            input_dim, self.config.d_model,
            seed=self.rng.randint(0, 2**31),
        )
        self._fusion = None  # rebuild on next forward

    def _build_fusion(self) -> None:
        n = len(self.streams)
        if self.config.fusion_method == "gated":
            self._fusion = GatedFusion(
                self.config.d_model, n,
                seed=self.rng.randint(0, 2**31),
            )
        elif self.config.fusion_method == "attention":
            self._fusion = AttentionFusion(
                self.config.d_model, self.config.n_heads,
                seed=self.rng.randint(0, 2**31),
            )

    def forward(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse feature streams into unified (d_model, T) representation."""
        if not self.streams:
            raise ValueError("No feature streams registered")

        if self._fusion is None:
            self._build_fusion()

        projected = []
        target_T = None

        # Project each stream
        for name in self.streams:
            if name not in features:
                continue
            feat = features[name]
            if feat.ndim == 1:
                feat = feat[np.newaxis, :]
            proj = self.projections[name].forward(feat)
            projected.append((name, proj))
            if target_T is None:
                target_T = proj.shape[1]

        if not projected:
            raise ValueError("No matching features provided")

        # Temporal alignment
        aligned = []
        for name, proj in projected:
            a = _align_temporal(proj, target_T, self.config.temporal_alignment)
            aligned.append(a)

        # Fusion
        if self.config.fusion_method == "concat":
            return np.concatenate(aligned, axis=0)[:self.config.d_model, :]
        elif self.config.fusion_method == "add":
            return sum(aligned) / len(aligned)
        elif self.config.fusion_method == "gated":
            return self._fusion.forward(aligned)
        elif self.config.fusion_method == "attention":
            primary = aligned[0]
            if len(aligned) > 1:
                context = np.concatenate(aligned[1:], axis=1)
                return self._fusion.forward(primary, context)
            return primary
        else:
            return sum(aligned) / len(aligned)


# ---------- full audio feature extractor ----------

class AudioFeatureExtractor:
    """Extract all standard audio features from raw audio.

    Returns a dict of feature name → (D, T) arrays that can be
    directly fed to MultiStreamFeatureIntegrator.
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 13,
        n_chroma: int = 12,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma

    def extract(
        self,
        signal: np.ndarray,
        features: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract specified features (or all by default).

        Available: mel, mfcc, chroma, onset, centroid, bandwidth,
                   rolloff, zcr, contrast
        """
        available = features or [
            "mel", "mfcc", "chroma", "onset",
            "centroid", "bandwidth", "rolloff", "zcr", "contrast",
        ]
        result = {}

        # Shared power spectrum (computed once)
        n_frames = max(1, 1 + (len(signal) - self.n_fft) // self.hop_length)
        window = np.hanning(self.n_fft)
        power = np.zeros((self.n_fft // 2 + 1, n_frames))
        for i in range(n_frames):
            start = i * self.hop_length
            frame = np.zeros(self.n_fft)
            end = min(start + self.n_fft, len(signal))
            frame[:end - start] = signal[start:end]
            frame *= window
            power[:, i] = np.abs(np.fft.rfft(frame)) ** 2

        if "mel" in available:
            mel_high = 2595.0 * np.log10(1.0 + (self.sr / 2.0) / 700.0)
            mel_pts = np.linspace(0, mel_high, self.n_mels + 2)
            hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
            bins = np.floor((self.n_fft + 1) * hz_pts / self.sr).astype(int)
            bins = np.clip(bins, 0, self.n_fft // 2)
            fb = np.zeros((self.n_mels, self.n_fft // 2 + 1))
            for m in range(self.n_mels):
                for k in range(bins[m], bins[m + 1]):
                    if bins[m + 1] > bins[m]:
                        fb[m, k] = (k - bins[m]) / (bins[m + 1] - bins[m])
                for k in range(bins[m + 1], bins[m + 2]):
                    if bins[m + 2] > bins[m + 1]:
                        fb[m, k] = (bins[m + 2] - k) / (bins[m + 2] - bins[m + 1])
            result["mel"] = np.log(fb @ power + 1e-9)

        if "mfcc" in available:
            result["mfcc"] = compute_mfcc(
                signal, self.sr, self.n_mfcc, self.n_fft, self.hop_length, self.n_mels,
            )

        if "chroma" in available:
            result["chroma"] = compute_chroma(
                signal, self.sr, self.n_fft, self.hop_length, self.n_chroma,
            )

        if "onset" in available:
            result["onset"] = compute_onset_strength(
                signal, self.sr, self.n_fft, self.hop_length,
            )

        if "centroid" in available:
            result["centroid"] = compute_spectral_centroid(
                power, self.sr, self.n_fft,
            )

        if "bandwidth" in available:
            result["bandwidth"] = compute_spectral_bandwidth(
                power, self.sr, self.n_fft,
            )

        if "rolloff" in available:
            result["rolloff"] = compute_spectral_rolloff(
                power, self.sr, self.n_fft,
            )

        if "zcr" in available:
            result["zcr"] = compute_zero_crossing_rate(
                signal, self.n_fft, self.hop_length,
            )

        if "contrast" in available:
            result["contrast"] = compute_spectral_contrast(power, n_bands=6)

        return result


def build_default_integrator(d_model: int = 256, seed: int = 42) -> MultiStreamFeatureIntegrator:
    """Build a default integrator with standard audio feature streams."""
    config = FeatureIntegrationConfig(d_model=d_model, fusion_method="gated", seed=seed)
    integrator = MultiStreamFeatureIntegrator(config)
    integrator.add_stream("mel", input_dim=128)
    integrator.add_stream("mfcc", input_dim=13)
    integrator.add_stream("chroma", input_dim=12)
    integrator.add_stream("onset", input_dim=1)
    integrator.add_stream("centroid", input_dim=1)
    integrator.add_stream("bandwidth", input_dim=1)
    integrator.add_stream("contrast", input_dim=6)
    return integrator
