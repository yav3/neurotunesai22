"""Spectrogram encoder/decoder: mel, CQT, PCEN, gammatone, multi-resolution.

Bridges raw audio (via STFT/mel/CQT) into the foundation model's latent space,
producing per-frame embeddings that the decoder can reconstruct or generate from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .encoder import LinearLayer, gelu, layer_norm, MultiHeadAttention
from .positional import sinusoidal_encoding, TemporalEncoding


@dataclass
class SpectrogramEncoderConfig:
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    hidden_dim: int = 256
    latent_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_frames: int = 1024
    pool: str = "mean"


def _hann_window(n: int) -> np.ndarray:
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


# ──────────────────────────────────────────────────────────────────────
# Core STFT and mel filterbank
# ──────────────────────────────────────────────────────────────────────

def _compute_mel_spectrogram(
    audio: np.ndarray, sr: int, n_fft: int, hop_length: int, n_mels: int,
) -> np.ndarray:
    w = _hann_window(n_fft)
    pad_len = n_fft // 2
    padded = np.pad(audio, (pad_len, pad_len), mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    n_freq = n_fft // 2 + 1

    power = np.zeros((n_freq, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        segment = padded[start:start + n_fft] * w
        spectrum = np.fft.rfft(segment)
        power[:, i] = np.abs(spectrum) ** 2

    f_max = sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_freq))
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, center):
            if center > left:
                fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fb[m, k] = (right - k) / (right - center)

    mel = fb @ power
    return np.log1p(mel).T  # (n_frames, n_mels)


# ──────────────────────────────────────────────────────────────────────
# STFT (complex) for phase-based features
# ──────────────────────────────────────────────────────────────────────

def compute_stft(
    audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
) -> np.ndarray:
    """Returns complex STFT matrix (n_freq, n_frames)."""
    w = _hann_window(n_fft)
    pad_len = n_fft // 2
    padded = np.pad(audio, (pad_len, pad_len), mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    n_freq = n_fft // 2 + 1
    S = np.zeros((n_freq, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop_length
        segment = padded[start:start + n_fft] * w
        S[:, i] = np.fft.rfft(segment)
    return S


def istft(
    S: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
) -> np.ndarray:
    """Griffin-Lim-compatible inverse STFT via overlap-add."""
    n_freq, n_frames = S.shape
    w = _hann_window(n_fft)
    out_len = n_fft + (n_frames - 1) * hop_length
    output = np.zeros(out_len)
    window_sum = np.zeros(out_len)
    for i in range(n_frames):
        start = i * hop_length
        frame = np.fft.irfft(S[:, i], n=n_fft)
        output[start:start + n_fft] += frame * w
        window_sum[start:start + n_fft] += w ** 2
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]
    pad = n_fft // 2
    return output[pad:out_len - pad]


def griffin_lim(
    magnitude: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
    n_iter: int = 32,
) -> np.ndarray:
    """Griffin-Lim phase reconstruction from magnitude spectrogram."""
    rng = np.random.RandomState(0)
    phase = np.exp(2j * np.pi * rng.random(magnitude.shape))
    for _ in range(n_iter):
        S = magnitude * phase
        audio = istft(S, n_fft, hop_length)
        S_new = compute_stft(audio, n_fft, hop_length)
        n_freq = min(S_new.shape[0], magnitude.shape[0])
        n_frames = min(S_new.shape[1], magnitude.shape[1])
        S_new = S_new[:n_freq, :n_frames]
        mag = np.abs(S_new) + 1e-12
        phase_new = S_new / mag
        phase = np.zeros_like(magnitude, dtype=np.complex128)
        phase[:n_freq, :n_frames] = phase_new
    return istft(magnitude * phase, n_fft, hop_length)


# ──────────────────────────────────────────────────────────────────────
# PCEN — Per-Channel Energy Normalization (Wang et al., 2017)
# ──────────────────────────────────────────────────────────────────────

def pcen(
    mel: np.ndarray,
    alpha: float = 0.98,
    delta: float = 2.0,
    r: float = 0.5,
    s: float = 0.025,
    eps: float = 1e-6,
) -> np.ndarray:
    """PCEN on log-mel features: adaptive gain + dynamic compression.

    mel: (n_frames, n_mels) — linear (not log) mel energies.
    """
    n_frames, n_mels = mel.shape
    M = np.zeros_like(mel)
    M[0] = mel[0]
    for t in range(1, n_frames):
        M[t] = (1 - s) * M[t - 1] + s * mel[t]
    smooth = np.maximum(M, eps)
    gain = (mel / (smooth ** alpha + delta)) ** r
    return gain


# ──────────────────────────────────────────────────────────────────────
# CQT — Constant-Q Transform (Brown, 1991)
# ──────────────────────────────────────────────────────────────────────

def compute_cqt(
    audio: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    fmin: float = 32.7,
    n_bins: int = 84,
    bins_per_octave: int = 12,
) -> np.ndarray:
    """Simplified CQT via windowed DFT at geometrically-spaced frequencies.

    Returns (n_bins, n_frames) magnitude matrix.
    """
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)

    pad_len = int(sr * Q / fmin) // 2
    padded = np.pad(audio, (pad_len, pad_len), mode="reflect")
    n_frames = max(1, (len(padded) - 1) // hop_length)

    cqt = np.zeros((n_bins, n_frames))
    for b in range(n_bins):
        f = freqs[b]
        N_k = int(np.ceil(Q * sr / f))
        N_k = min(N_k, len(padded))
        window = _hann_window(N_k)
        kernel = window * np.exp(-2j * np.pi * f * np.arange(N_k) / sr)
        for t in range(n_frames):
            start = t * hop_length
            end = min(start + N_k, len(padded))
            seg = padded[start:end]
            if len(seg) < N_k:
                seg = np.pad(seg, (0, N_k - len(seg)))
            cqt[b, t] = np.abs(np.dot(seg, kernel.conj())) / N_k
    return cqt


# ──────────────────────────────────────────────────────────────────────
# Gammatone filterbank (Patterson et al., 1992)
# ──────────────────────────────────────────────────────────────────────

def erb_scale(f: np.ndarray) -> np.ndarray:
    return 9.265 * np.log(1.0 + f / (24.7 * 9.265))


def gammatone_filterbank(
    n_filters: int, n_fft: int, sr: int = 22050, fmin: float = 50.0,
) -> np.ndarray:
    """Gammatone-shaped filterbank weights (n_filters, n_fft//2+1)."""
    n_freq = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freq)
    erb_min = erb_scale(np.array([fmin]))[0]
    erb_max = erb_scale(np.array([sr / 2.0]))[0]
    center_erbs = np.linspace(erb_min, erb_max, n_filters)
    center_hz = 24.7 * 9.265 * (np.exp(center_erbs / 9.265) - 1.0)

    fb = np.zeros((n_filters, n_freq))
    for i in range(n_filters):
        cf = center_hz[i]
        erb = 24.7 * (4.37 * cf / 1000.0 + 1.0)
        b = 1.019 * 2 * np.pi * erb
        diff = freqs - cf
        response = (1.0 + (2 * diff / erb) ** 2) ** (-2)
        fb[i] = response
    fb /= fb.max(axis=1, keepdims=True) + 1e-12
    return fb


def compute_gammatone_spectrogram(
    audio: np.ndarray, sr: int = 22050, n_fft: int = 2048,
    hop_length: int = 512, n_filters: int = 64,
) -> np.ndarray:
    """Gammatone spectrogram: (n_frames, n_filters)."""
    S = compute_stft(audio, n_fft, hop_length)
    power = np.abs(S) ** 2
    fb = gammatone_filterbank(n_filters, n_fft, sr)
    return np.log1p(fb @ power).T


# ──────────────────────────────────────────────────────────────────────
# Multi-resolution STFT (Wang et al., 2019 — MR-STFT loss)
# ──────────────────────────────────────────────────────────────────────

def multi_resolution_stft_loss(
    target: np.ndarray,
    predicted: np.ndarray,
    fft_sizes: list[int] | None = None,
    hop_sizes: list[int] | None = None,
) -> float:
    """Multi-resolution STFT distance (spectral convergence + log magnitude)."""
    if fft_sizes is None:
        fft_sizes = [512, 1024, 2048]
    if hop_sizes is None:
        hop_sizes = [s // 4 for s in fft_sizes]

    sc_loss = 0.0
    mag_loss = 0.0
    for n_fft, hop in zip(fft_sizes, hop_sizes):
        S_t = compute_stft(target, n_fft, hop)
        S_p = compute_stft(predicted, n_fft, hop)
        n_f = min(S_t.shape[0], S_p.shape[0])
        n_fr = min(S_t.shape[1], S_p.shape[1])
        mag_t = np.abs(S_t[:n_f, :n_fr])
        mag_p = np.abs(S_p[:n_f, :n_fr])
        sc_loss += float(np.linalg.norm(mag_t - mag_p)) / (float(np.linalg.norm(mag_t)) + 1e-8)
        mag_loss += float(np.mean(np.abs(np.log(mag_t + 1e-8) - np.log(mag_p + 1e-8))))

    n = len(fft_sizes)
    return (sc_loss + mag_loss) / (2 * n)


# ──────────────────────────────────────────────────────────────────────
# Phase features
# ──────────────────────────────────────────────────────────────────────

def instantaneous_frequency(
    audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
) -> np.ndarray:
    """Instantaneous frequency from phase derivative (n_freq, n_frames)."""
    S = compute_stft(audio, n_fft, hop_length)
    phase = np.angle(S)
    dphase = np.diff(phase, axis=1, prepend=phase[:, :1])
    dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
    return dphase / (2 * np.pi * hop_length / len(audio)) if len(audio) > 0 else dphase


def group_delay(
    audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
) -> np.ndarray:
    """Group delay: negative derivative of phase w.r.t. frequency."""
    S = compute_stft(audio, n_fft, hop_length)
    phase = np.angle(S)
    gd = -np.diff(phase, axis=0, prepend=phase[:1, :])
    return gd


# ──────────────────────────────────────────────────────────────────────
# Spectrogram augmentation in frequency domain
# ──────────────────────────────────────────────────────────────────────

def spec_augment(
    spec: np.ndarray,
    n_freq_masks: int = 2,
    freq_mask_width: int = 10,
    n_time_masks: int = 2,
    time_mask_width: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """SpecAugment (Park et al., 2019) on (n_frames, n_freq) spectrogram."""
    rng = np.random.RandomState(seed)
    out = spec.copy()
    n_frames, n_freq = out.shape
    for _ in range(n_freq_masks):
        f0 = rng.randint(0, max(n_freq - freq_mask_width, 1))
        out[:, f0:f0 + freq_mask_width] = 0.0
    for _ in range(n_time_masks):
        t0 = rng.randint(0, max(n_frames - time_mask_width, 1))
        out[t0:t0 + time_mask_width, :] = 0.0
    return out


# ──────────────────────────────────────────────────────────────────────
# Transformer blocks
# ──────────────────────────────────────────────────────────────────────

class SpectrogramTransformerBlock:
    def __init__(self, dim: int, n_heads: int, seed: int = 0) -> None:
        self.attn = MultiHeadAttention(dim, n_heads, seed)
        self.ff1 = LinearLayer(dim, dim * 4, seed + 100)
        self.ff2 = LinearLayer(dim * 4, dim, seed + 200)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = layer_norm(x + self.attn.forward(x))
        ff = self.ff2.forward(gelu(self.ff1.forward(h)))
        return layer_norm(h + ff)


# ──────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────

class SpectrogramEncoder:
    """Encodes mel-spectrograms into frame-level latent representations."""

    def __init__(self, config: SpectrogramEncoderConfig | None = None, seed: int = 0) -> None:
        self.config = config or SpectrogramEncoderConfig()
        c = self.config
        self.input_proj = LinearLayer(c.n_mels, c.hidden_dim, seed)
        self.layers = [
            SpectrogramTransformerBlock(c.hidden_dim, c.n_heads, seed + i * 300)
            for i in range(c.n_layers)
        ]
        self.output_proj = LinearLayer(c.hidden_dim, c.latent_dim, seed + 999)
        self._pos_enc = None

    def _get_pos_encoding(self, n_frames: int) -> np.ndarray:
        return sinusoidal_encoding(n_frames, self.config.hidden_dim)

    def encode_mel(self, mel: np.ndarray) -> np.ndarray:
        if mel.ndim == 1:
            mel = mel.reshape(1, -1)
        n_frames = min(mel.shape[0], self.config.max_frames)
        mel = mel[:n_frames]

        h = gelu(self.input_proj.forward(mel))
        h = h + self._get_pos_encoding(n_frames)
        for block in self.layers:
            h = block.forward(h)
        return self.output_proj.forward(h)

    def encode_audio(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        mel = _compute_mel_spectrogram(
            audio.astype(np.float64), sr,
            self.config.n_fft, self.config.hop_length, self.config.n_mels,
        )
        return self.encode_mel(mel)

    def encode_to_single(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        frames = self.encode_audio(audio, sr)
        if self.config.pool == "mean":
            return frames.mean(axis=0)
        elif self.config.pool == "max":
            return frames.max(axis=0)
        return frames[-1]

    def params(self) -> dict[str, np.ndarray]:
        p = {}
        p.update({f"spec_input_{k}": v for k, v in self.input_proj.params.items()})
        for i, block in enumerate(self.layers):
            p.update({f"spec_block{i}_attn_{k}": v for k, v in block.attn.Wq.params.items()})
            p.update({f"spec_block{i}_ff1_{k}": v for k, v in block.ff1.params.items()})
            p.update({f"spec_block{i}_ff2_{k}": v for k, v in block.ff2.params.items()})
        p.update({f"spec_output_{k}": v for k, v in self.output_proj.params.items()})
        return p


# ──────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────

class SpectrogramDecoder:
    """Decodes frame-level latents back to mel-spectrogram space."""

    def __init__(self, config: SpectrogramEncoderConfig | None = None, seed: int = 5000) -> None:
        self.config = config or SpectrogramEncoderConfig()
        c = self.config
        self.input_proj = LinearLayer(c.latent_dim, c.hidden_dim, seed)
        self.layers = [
            SpectrogramTransformerBlock(c.hidden_dim, c.n_heads, seed + i * 300)
            for i in range(c.n_layers)
        ]
        self.output_proj = LinearLayer(c.hidden_dim, c.n_mels, seed + 999)

    def decode(self, z: np.ndarray) -> np.ndarray:
        if z.ndim == 1:
            z = z.reshape(1, -1)
        h = gelu(self.input_proj.forward(z))
        h = h + sinusoidal_encoding(z.shape[0], self.config.hidden_dim)
        for block in self.layers:
            h = block.forward(h)
        return self.output_proj.forward(h)

    def reconstruct_mel(self, z: np.ndarray) -> np.ndarray:
        log_mel = self.decode(z)
        return np.expm1(np.maximum(log_mel, 0.0))


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    rng = np.random.RandomState(0)
    sr = 16000
    dur = 0.5
    audio = rng.randn(int(sr * dur))

    # Mel spectrogram
    mel = _compute_mel_spectrogram(audio, sr, 1024, 256, 64)
    assert mel.ndim == 2, f"Mel ndim={mel.ndim}"
    assert mel.shape[1] == 64, f"Mel bands={mel.shape[1]}"

    # STFT roundtrip
    S = compute_stft(audio, 1024, 256)
    recon = istft(S, 1024, 256)
    n = min(len(audio), len(recon))
    err = np.mean((audio[:n] - recon[:n]) ** 2)
    assert err < 1e-8, f"STFT roundtrip MSE={err}"

    # Griffin-Lim
    mag = np.abs(S)
    gl_audio = griffin_lim(mag, 1024, 256, n_iter=5)
    assert len(gl_audio) > 0, "Griffin-Lim output"

    # PCEN
    mel_linear = np.abs(mel)
    pcen_out = pcen(mel_linear)
    assert pcen_out.shape == mel_linear.shape, f"PCEN shape"
    assert np.all(np.isfinite(pcen_out)), "PCEN finite"

    # CQT
    cqt = compute_cqt(audio, sr, 256, n_bins=36, bins_per_octave=12)
    assert cqt.shape[0] == 36, f"CQT bins={cqt.shape[0]}"
    assert cqt.shape[1] > 0, "CQT frames"

    # Gammatone
    gt = compute_gammatone_spectrogram(audio, sr, 1024, 256, 32)
    assert gt.shape[1] == 32, f"Gammatone filters={gt.shape[1]}"

    # Multi-resolution loss
    audio2 = audio + 0.01 * rng.randn(len(audio))
    mr_loss = multi_resolution_stft_loss(audio, audio2, [256, 512], [64, 128])
    assert mr_loss > 0, f"MR-STFT loss={mr_loss}"

    # Phase features
    ifreq = instantaneous_frequency(audio, 1024, 256)
    assert ifreq.shape[0] == 513, f"IF shape={ifreq.shape}"
    gd = group_delay(audio, 1024, 256)
    assert gd.shape[0] == 513, f"GD shape={gd.shape}"

    # SpecAugment
    aug = spec_augment(mel, n_freq_masks=1, freq_mask_width=5, n_time_masks=1, time_mask_width=5)
    assert aug.shape == mel.shape, "SpecAugment shape"
    assert np.sum(aug == 0) > np.sum(mel == 0), "SpecAugment masks"

    # Gammatone filterbank
    fb = gammatone_filterbank(32, 1024, sr)
    assert fb.shape == (32, 513), f"Gammatone FB shape={fb.shape}"

    parts = [
        "Mel", "STFT-roundtrip", "Griffin-Lim", "PCEN", "CQT",
        "Gammatone", "MR-STFT", "IF", "GroupDelay", "SpecAugment", "FB",
    ]
    print(f"Self-test passed: {', '.join(parts)}.")
