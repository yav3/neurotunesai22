"""Spectrogram encoder: mel-spectrogram → frame-level embeddings.

Bridges raw audio (via STFT/mel) into the foundation model's latent space,
producing per-frame embeddings that the decoder can reconstruct or generate from.
"""
from __future__ import annotations

from dataclasses import dataclass
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


class SpectrogramTransformerBlock:
    def __init__(self, dim: int, n_heads: int, seed: int = 0) -> None:
        self.attn = MultiHeadAttention(dim, n_heads, seed)
        self.ff1 = LinearLayer(dim, dim * 4, seed + 100)
        self.ff2 = LinearLayer(dim * 4, dim, seed + 200)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = layer_norm(x + self.attn.forward(x))
        ff = self.ff2.forward(gelu(self.ff1.forward(h)))
        return layer_norm(h + ff)


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
