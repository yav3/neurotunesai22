"""Neural music synthesis: causal convolutions, autoregressive generation, spectral synthesis.

Implements forward-pass neural synthesis components for music generation:
- WaveNet-style causal dilated convolution (van den Oord et al., 2016)
- SampleRNN hierarchical generation (Mehri et al., 2017)
- Autoregressive sampling: temperature, nucleus/top-p, repetition penalty
- Spectral synthesis: additive, subtractive, noise-band
- Neural vocoder: Griffin-Lim, overlap-add, window functions

Pure numpy/scipy. Forward-pass only (no autograd).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# Configuration dataclasses
# ============================================================

@dataclass
class WaveNetConfig:
    """Configuration for a WaveNet-style network (van den Oord et al., 2016)."""
    n_layers: int = 10
    n_stacks: int = 2
    residual_channels: int = 32
    skip_channels: int = 64
    kernel_size: int = 2
    dilation_base: int = 2

    @property
    def receptive_field(self) -> int:
        """Total receptive field in samples."""
        per_stack = sum(self.dilation_base ** i for i in range(self.n_layers))
        return self.n_stacks * per_stack * (self.kernel_size - 1) + 1


@dataclass
class SampleRNNConfig:
    """Configuration for SampleRNN hierarchical generation (Mehri et al., 2017)."""
    frame_size: int = 16
    n_frame_tiers: int = 2
    frame_hidden: int = 256
    sample_hidden: int = 256
    n_quantization: int = 256


@dataclass
class VocoderConfig:
    """Configuration for neural vocoder / Griffin-Lim reconstruction."""
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_iter: int = 32
    window: str = "hann"
    sr: int = 22050


@dataclass
class SynthesisResult:
    """Container for synthesis output."""
    audio: np.ndarray
    sample_rate: int = 22050
    metadata: dict = field(default_factory=dict)


# ============================================================
# Causal dilated convolution (WaveNet core)
# ============================================================

def causal_pad(x: np.ndarray, kernel_size: int, dilation: int) -> np.ndarray:
    """Left-pad signal for causal convolution.

    Args:
        x: (T,) or (T, C) input signal
        kernel_size: convolution kernel width
        dilation: dilation factor

    Returns:
        Padded signal with (kernel_size - 1) * dilation zeros prepended.
    """
    pad_len = (kernel_size - 1) * dilation
    if x.ndim == 1:
        return np.pad(x, (pad_len, 0), mode="constant")
    return np.pad(x, ((pad_len, 0), (0, 0)), mode="constant")


def dilated_conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    dilation: int = 1,
) -> np.ndarray:
    """1-D dilated convolution via gather + matmul.

    Args:
        x: (T, C_in) input
        weight: (kernel_size, C_in, C_out)
        bias: (C_out,) optional
        dilation: dilation factor

    Returns:
        (T, C_out) output (same length as input; caller handles causal padding).
    """
    K, C_in, C_out = weight.shape
    T = x.shape[0]
    out = np.zeros((T, C_out))
    for k in range(K):
        offset = k * dilation
        if offset >= T:
            break
        valid = T - offset
        out[offset:] += x[:valid] @ weight[k]
    if bias is not None:
        out += bias
    return out


def causal_dilated_conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    dilation: int = 1,
) -> np.ndarray:
    """Causal dilated 1-D convolution (van den Oord et al., 2016).

    Pads the input so output[t] depends only on input[..t].

    Args:
        x: (T, C_in)
        weight: (kernel_size, C_in, C_out)
        bias: (C_out,) optional
        dilation: dilation factor

    Returns:
        (T, C_out) causally convolved output.
    """
    K = weight.shape[0]
    padded = causal_pad(x, K, dilation)
    result = dilated_conv1d(padded, weight, bias, dilation)
    return result[: x.shape[0]]


def gated_activation(x_filter: np.ndarray, x_gate: np.ndarray) -> np.ndarray:
    """Gated activation unit: tanh(filter) * sigmoid(gate).

    Core nonlinearity in WaveNet (van den Oord et al., 2016, eq. 2).
    """
    return np.tanh(x_filter) * _sigmoid(x_gate)


def wavenet_residual_block(
    x: np.ndarray,
    W_filter: np.ndarray,
    W_gate: np.ndarray,
    W_res: np.ndarray,
    W_skip: np.ndarray,
    dilation: int,
    b_filter: Optional[np.ndarray] = None,
    b_gate: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """One WaveNet residual block: dilated conv → gated activation → 1x1 projections.

    Args:
        x: (T, residual_channels)
        W_filter, W_gate: (kernel_size, residual_channels, residual_channels)
        W_res: (residual_channels, residual_channels) — 1x1 residual projection
        W_skip: (residual_channels, skip_channels) — 1x1 skip projection
        dilation: dilation factor for this layer

    Returns:
        (residual_output, skip_output) both (T, channels)
    """
    h_filter = causal_dilated_conv1d(x, W_filter, b_filter, dilation)
    h_gate = causal_dilated_conv1d(x, W_gate, b_gate, dilation)

    h = gated_activation(h_filter, h_gate)

    residual = x + h @ W_res
    skip = h @ W_skip
    return residual, skip


def wavenet_forward(
    x: np.ndarray,
    weights: dict,
    config: WaveNetConfig,
) -> np.ndarray:
    """Full WaveNet forward pass.

    Args:
        x: (T, 1) or (T,) input waveform (mu-law quantized or continuous)
        weights: dict with keys 'W_filter_s{s}_l{l}', 'W_gate_s{s}_l{l}',
                 'W_res_s{s}_l{l}', 'W_skip_s{s}_l{l}', 'W_out1', 'W_out2'
        config: WaveNetConfig

    Returns:
        (T, n_classes) logits or (T, 1) continuous output
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.shape[1] == 1:
        W_embed = weights.get("W_embed")
        if W_embed is not None:
            h = x @ W_embed.T if W_embed.ndim == 2 else np.tile(x, (1, config.residual_channels))
        else:
            h = np.tile(x, (1, config.residual_channels))
    else:
        h = x

    skip_sum = np.zeros((h.shape[0], config.skip_channels))

    for s in range(config.n_stacks):
        for l in range(config.n_layers):
            dilation = config.dilation_base ** l
            key = f"s{s}_l{l}"
            h, skip = wavenet_residual_block(
                h,
                weights[f"W_filter_{key}"],
                weights[f"W_gate_{key}"],
                weights[f"W_res_{key}"],
                weights[f"W_skip_{key}"],
                dilation,
            )
            skip_sum += skip

    out = np.maximum(skip_sum, 0)  # ReLU
    out = out @ weights["W_out1"]
    out = np.maximum(out, 0)
    out = out @ weights["W_out2"]
    return out


def init_wavenet_weights(
    config: WaveNetConfig,
    output_dim: int = 256,
    seed: int = 42,
) -> dict:
    """Initialize WaveNet weights (Kaiming / He initialization).

    Returns:
        dict of weight arrays keyed by layer name.
    """
    rng = np.random.RandomState(seed)
    C = config.residual_channels
    S = config.skip_channels
    K = config.kernel_size
    weights = {}

    weights["W_embed"] = rng.randn(C, 1) * np.sqrt(2.0 / 1)

    for s in range(config.n_stacks):
        for l in range(config.n_layers):
            key = f"s{s}_l{l}"
            scale = np.sqrt(2.0 / (K * C))
            weights[f"W_filter_{key}"] = rng.randn(K, C, C) * scale
            weights[f"W_gate_{key}"] = rng.randn(K, C, C) * scale
            weights[f"W_res_{key}"] = rng.randn(C, C) * np.sqrt(2.0 / C)
            weights[f"W_skip_{key}"] = rng.randn(C, S) * np.sqrt(2.0 / C)

    weights["W_out1"] = rng.randn(S, S) * np.sqrt(2.0 / S)
    weights["W_out2"] = rng.randn(S, output_dim) * np.sqrt(2.0 / S)
    return weights


def receptive_field(
    n_layers: int,
    n_stacks: int = 1,
    kernel_size: int = 2,
    dilation_base: int = 2,
) -> int:
    """Compute receptive field of a stack of dilated causal convolutions.

    RF = n_stacks * sum_{l=0}^{L-1} (K-1)*d^l + 1 (van den Oord et al., 2016).
    """
    per_stack = sum((kernel_size - 1) * dilation_base ** l for l in range(n_layers))
    return n_stacks * per_stack + 1


# ============================================================
# SampleRNN hierarchical generation (Mehri et al., 2017)
# ============================================================

def frame_level_rnn(
    frames: np.ndarray,
    W_h: np.ndarray,
    W_x: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Frame-level tier: processes frames of aggregated samples.

    Vanilla RNN: h_t = tanh(W_h h_{t-1} + W_x x_t + b).

    Args:
        frames: (n_frames, frame_size) input frames
        W_h: (hidden, hidden)
        W_x: (frame_size, hidden)
        b: (hidden,)

    Returns:
        (n_frames, hidden) hidden states
    """
    n_frames = frames.shape[0]
    hidden = W_h.shape[0]
    h = np.zeros(hidden)
    states = np.zeros((n_frames, hidden))
    for t in range(n_frames):
        h = np.tanh(W_h @ h + W_x @ frames[t] + b)
        states[t] = h
    return states


def sample_level_mlp(
    samples: np.ndarray,
    conditioning: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Sample-level MLP: predicts next sample given recent samples + frame conditioning.

    Args:
        samples: (T, lookback) recent sample windows
        conditioning: (T, hidden) frame-level conditioning (upsampled)
        W1: (lookback + hidden, mlp_hidden)
        b1: (mlp_hidden,)
        W2: (mlp_hidden, n_quantization)
        b2: (n_quantization,)

    Returns:
        (T, n_quantization) logits
    """
    x = np.concatenate([samples, conditioning], axis=-1)
    h = np.maximum(x @ W1 + b1, 0)  # ReLU
    return h @ W2 + b2


def upsample_conditioning(
    frame_states: np.ndarray,
    frame_size: int,
) -> np.ndarray:
    """Upsample frame-level states to sample-level via repetition.

    Args:
        frame_states: (n_frames, hidden)
        frame_size: samples per frame

    Returns:
        (n_frames * frame_size, hidden)
    """
    return np.repeat(frame_states, frame_size, axis=0)


def samplernn_forward(
    x: np.ndarray,
    weights: dict,
    config: SampleRNNConfig,
) -> np.ndarray:
    """SampleRNN forward pass (Mehri et al., 2017).

    Two-tier: frame-level RNN + sample-level MLP.

    Args:
        x: (T,) quantized audio samples [0, n_quantization)
        weights: dict with 'W_h', 'W_x', 'b_rnn', 'W1', 'b1', 'W2', 'b2'
        config: SampleRNNConfig

    Returns:
        (T, n_quantization) logits for next-sample prediction
    """
    T = len(x)
    fs = config.frame_size
    n_frames = T // fs

    frames = x[: n_frames * fs].reshape(n_frames, fs).astype(np.float64)
    frames = frames / config.n_quantization * 2 - 1  # normalize to [-1, 1]

    frame_states = frame_level_rnn(
        frames, weights["W_h"], weights["W_x"], weights["b_rnn"],
    )

    conditioning = upsample_conditioning(frame_states, fs)

    lookback = min(fs, 4)
    n_samples = n_frames * fs
    sample_windows = np.zeros((n_samples, lookback))
    x_norm = x[:n_samples].astype(np.float64) / config.n_quantization * 2 - 1
    for t in range(n_samples):
        for k in range(lookback):
            if t - k - 1 >= 0:
                sample_windows[t, k] = x_norm[t - k - 1]

    logits = sample_level_mlp(
        sample_windows, conditioning,
        weights["W1"], weights["b1"], weights["W2"], weights["b2"],
    )
    return logits


def init_samplernn_weights(
    config: SampleRNNConfig,
    seed: int = 42,
) -> dict:
    """Initialize SampleRNN weights."""
    rng = np.random.RandomState(seed)
    H = config.frame_hidden
    fs = config.frame_size
    lookback = min(fs, 4)
    Q = config.n_quantization
    mlp_h = config.sample_hidden

    weights = {}
    weights["W_h"] = rng.randn(H, H) * np.sqrt(2.0 / H)
    weights["W_x"] = rng.randn(fs, H) * np.sqrt(2.0 / fs)
    weights["b_rnn"] = np.zeros(H)

    in_dim = lookback + H
    weights["W1"] = rng.randn(in_dim, mlp_h) * np.sqrt(2.0 / in_dim)
    weights["b1"] = np.zeros(mlp_h)
    weights["W2"] = rng.randn(mlp_h, Q) * np.sqrt(2.0 / mlp_h)
    weights["b2"] = np.zeros(Q)
    return weights


# ============================================================
# Autoregressive sampling utilities
# ============================================================

def temperature_sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> int:
    """Sample from logits with temperature scaling.

    Lower temperature → more deterministic; higher → more random.
    """
    rng = np.random.RandomState(seed)
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    probs = _softmax(scaled)
    return int(rng.choice(len(probs), p=probs))


def nucleus_sample(
    logits: np.ndarray,
    top_p: float = 0.9,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> int:
    """Nucleus (top-p) sampling (Holtzman et al., 2020).

    Restricts sampling to the smallest set of tokens with cumulative
    probability >= top_p, then renormalizes.
    """
    rng = np.random.RandomState(seed)
    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits / temperature
    probs = _softmax(scaled)

    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)

    cutoff = np.searchsorted(cumsum, top_p) + 1
    cutoff = min(cutoff, len(probs))

    top_idx = sorted_idx[:cutoff]
    top_probs = probs[top_idx]
    top_probs /= top_probs.sum()

    chosen = rng.choice(top_idx, p=top_probs)
    return int(chosen)


def top_k_sample(
    logits: np.ndarray,
    k: int = 40,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> int:
    """Top-k sampling (Fan et al., 2018)."""
    rng = np.random.RandomState(seed)
    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits / temperature
    top_idx = np.argsort(-scaled)[:k]
    top_logits = scaled[top_idx]
    top_probs = _softmax(top_logits)
    chosen = rng.choice(top_idx, p=top_probs)
    return int(chosen)


def apply_repetition_penalty(
    logits: np.ndarray,
    generated: np.ndarray,
    penalty: float = 1.2,
    window: int = 64,
) -> np.ndarray:
    """Penalize recently generated tokens to reduce repetition.

    Divides logits of recently-seen tokens by the penalty factor
    (Keskar et al., 2019).
    """
    result = logits.copy()
    recent = generated[-window:] if len(generated) > window else generated
    seen = np.unique(recent.astype(int))
    for idx in seen:
        if 0 <= idx < len(result):
            if result[idx] > 0:
                result[idx] /= penalty
            else:
                result[idx] *= penalty
    return result


def mu_law_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Mu-law companding encoding (ITU-T G.711).

    Maps continuous [-1, 1] to quantized [0, mu].
    """
    x = np.clip(x, -1.0, 1.0)
    compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = ((compressed + 1) / 2 * mu + 0.5).astype(int)
    return np.clip(quantized, 0, mu)


def mu_law_decode(y: np.ndarray, mu: int = 255) -> np.ndarray:
    """Mu-law companding decoding.

    Maps quantized [0, mu] back to continuous [-1, 1].
    """
    y = y.astype(np.float64)
    normalized = 2.0 * y / mu - 1.0
    expanded = np.sign(normalized) * ((1 + mu) ** np.abs(normalized) - 1) / mu
    return expanded


# ============================================================
# Spectral synthesis
# ============================================================

def additive_synthesis(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    phases: Optional[np.ndarray] = None,
    n_samples: int = 22050,
    sr: int = 22050,
) -> np.ndarray:
    """Additive synthesis from sinusoidal partials.

    Args:
        frequencies: (n_partials,) partial frequencies in Hz
        amplitudes: (n_partials,) or (n_frames, n_partials) amplitudes
        phases: (n_partials,) initial phases in radians; zero if None
        n_samples: output length
        sr: sample rate

    Returns:
        (n_samples,) synthesized waveform
    """
    n_partials = len(frequencies)
    if phases is None:
        phases = np.zeros(n_partials)

    t = np.arange(n_samples) / sr

    if amplitudes.ndim == 1:
        signal = np.zeros(n_samples)
        for i in range(n_partials):
            signal += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i])
    else:
        n_frames = amplitudes.shape[0]
        frame_len = n_samples / n_frames
        amp_interp = np.zeros((n_samples, n_partials))
        for i in range(n_partials):
            amp_interp[:, i] = np.interp(
                np.arange(n_samples),
                np.linspace(0, n_samples - 1, n_frames),
                amplitudes[:, i],
            )
        signal = np.zeros(n_samples)
        for i in range(n_partials):
            signal += amp_interp[:, i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i])

    return signal


def harmonic_amplitudes(
    f0: float,
    n_harmonics: int = 16,
    rolloff: float = 1.0,
    sr: int = 22050,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate harmonic series with amplitude rolloff.

    Amplitude of k-th harmonic: 1 / k^rolloff (capped at Nyquist).
    """
    harmonics = f0 * np.arange(1, n_harmonics + 1)
    mask = harmonics < sr / 2
    harmonics = harmonics[mask]
    amps = 1.0 / np.arange(1, len(harmonics) + 1) ** rolloff
    return harmonics, amps


def subtractive_synthesis(
    source: np.ndarray,
    filter_coeffs: np.ndarray,
    sr: int = 22050,
) -> np.ndarray:
    """Subtractive synthesis via IIR filtering.

    Applies an all-pole filter (LPC-style) to a source signal.

    Args:
        source: (T,) excitation signal (e.g., impulse train or noise)
        filter_coeffs: (order,) AR/LPC coefficients a[1..p]
        sr: sample rate (for reference)

    Returns:
        (T,) filtered signal
    """
    from scipy.signal import lfilter

    a = np.concatenate([[1.0], -filter_coeffs])
    return lfilter([1.0], a, source)


def noise_band_synthesis(
    center_freqs: np.ndarray,
    bandwidths: np.ndarray,
    amplitudes: np.ndarray,
    n_samples: int = 22050,
    sr: int = 22050,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Noise-band synthesis: sum of bandpass-filtered noise.

    Used for breathy, noisy, or aspirated components.

    Args:
        center_freqs: (n_bands,) center frequencies in Hz
        bandwidths: (n_bands,) bandwidth in Hz
        amplitudes: (n_bands,) band amplitudes
        n_samples: output length
        sr: sample rate

    Returns:
        (n_samples,) synthesized noise signal
    """
    from scipy.signal import butter, lfilter

    rng = np.random.RandomState(seed)
    noise = rng.randn(n_samples)
    signal = np.zeros(n_samples)
    nyquist = sr / 2.0

    for i in range(len(center_freqs)):
        low = max((center_freqs[i] - bandwidths[i] / 2) / nyquist, 0.001)
        high = min((center_freqs[i] + bandwidths[i] / 2) / nyquist, 0.999)
        if low >= high:
            continue
        b, a = butter(2, [low, high], btype="band")
        filtered = lfilter(b, a, noise)
        signal += amplitudes[i] * filtered

    return signal


def pulse_train(
    f0: float,
    n_samples: int = 22050,
    sr: int = 22050,
) -> np.ndarray:
    """Generate an impulse train at fundamental frequency f0.

    Used as excitation for subtractive synthesis of voiced speech/singing.
    """
    period = int(round(sr / f0))
    if period < 1:
        period = 1
    signal = np.zeros(n_samples)
    signal[::period] = 1.0
    return signal


# ============================================================
# Neural vocoder components
# ============================================================

def get_window(name: str, length: int) -> np.ndarray:
    """Get a window function by name."""
    if name == "hann":
        return np.hanning(length)
    elif name == "hamming":
        return np.hamming(length)
    elif name == "blackman":
        return np.blackman(length)
    elif name == "bartlett":
        return np.bartlett(length)
    elif name == "kaiser":
        return np.kaiser(length, beta=14.0)
    else:
        return np.ones(length)


def stft(
    x: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
) -> np.ndarray:
    """Short-time Fourier transform.

    Args:
        x: (T,) time-domain signal
        n_fft: FFT size
        hop_length: hop between frames
        win_length: window length (defaults to n_fft)
        window: window function name

    Returns:
        (n_fft // 2 + 1, n_frames) complex STFT matrix
    """
    if win_length is None:
        win_length = n_fft
    w = get_window(window, win_length)
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        w = np.pad(w, (pad_left, n_fft - win_length - pad_left))

    x_padded = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(x_padded) - n_fft) // hop_length

    result = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop_length
        frame = x_padded[start: start + n_fft] * w
        result[:, i] = np.fft.rfft(frame)
    return result


def istft(
    S: np.ndarray,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    length: Optional[int] = None,
) -> np.ndarray:
    """Inverse STFT via overlap-add.

    Args:
        S: (n_fft // 2 + 1, n_frames) complex STFT
        hop_length: hop between frames
        win_length: window length
        window: window function name
        length: desired output length (truncates or pads)

    Returns:
        (T,) time-domain signal
    """
    n_fft = (S.shape[0] - 1) * 2
    if win_length is None:
        win_length = n_fft
    n_frames = S.shape[1]
    w = get_window(window, win_length)
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        w = np.pad(w, (pad_left, n_fft - win_length - pad_left))

    out_len = n_fft + hop_length * (n_frames - 1)
    signal = np.zeros(out_len)
    window_sum = np.zeros(out_len)

    for i in range(n_frames):
        start = i * hop_length
        frame = np.fft.irfft(S[:, i], n=n_fft)
        signal[start: start + n_fft] += frame * w
        window_sum[start: start + n_fft] += w ** 2

    nonzero = window_sum > 1e-8
    signal[nonzero] /= window_sum[nonzero]

    offset = n_fft // 2
    signal = signal[offset:]

    if length is not None:
        if len(signal) > length:
            signal = signal[:length]
        elif len(signal) < length:
            signal = np.pad(signal, (0, length - len(signal)))
    return signal


def griffin_lim(
    magnitude: np.ndarray,
    n_iter: int = 32,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    length: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Griffin-Lim phase reconstruction (Griffin & Lim, 1984).

    Iteratively estimates phase from magnitude spectrogram via
    alternating STFT ↔ iSTFT projections.

    Args:
        magnitude: (n_freq, n_frames) magnitude spectrogram
        n_iter: number of iterations
        hop_length: STFT hop
        win_length: STFT window length
        window: window function name
        length: desired output length
        seed: random seed for initial phase

    Returns:
        (T,) reconstructed time-domain signal
    """
    rng = np.random.RandomState(seed)
    phase = np.exp(2j * np.pi * rng.rand(*magnitude.shape))
    S = magnitude * phase

    for _ in range(n_iter):
        x = istft(S, hop_length, win_length, window, length)
        S_new = stft(x, (magnitude.shape[0] - 1) * 2, hop_length, win_length, window)
        if S_new.shape[1] > magnitude.shape[1]:
            S_new = S_new[:, :magnitude.shape[1]]
        elif S_new.shape[1] < magnitude.shape[1]:
            pad = magnitude.shape[1] - S_new.shape[1]
            S_new = np.pad(S_new, ((0, 0), (0, pad)))
        phase = np.exp(1j * np.angle(S_new))
        S = magnitude * phase

    return istft(S, hop_length, win_length, window, length)


def overlap_add(
    frames: np.ndarray,
    hop_length: int,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Overlap-add synthesis from windowed frames.

    Args:
        frames: (n_frames, frame_length) windowed frames
        hop_length: hop size
        window: (frame_length,) synthesis window

    Returns:
        (T,) reconstructed signal
    """
    n_frames, frame_len = frames.shape
    out_len = frame_len + hop_length * (n_frames - 1)
    signal = np.zeros(out_len)
    norm = np.zeros(out_len)

    if window is None:
        window = np.hanning(frame_len)

    for i in range(n_frames):
        start = i * hop_length
        signal[start: start + frame_len] += frames[i] * window
        norm[start: start + frame_len] += window ** 2

    nonzero = norm > 1e-8
    signal[nonzero] /= norm[nonzero]
    return signal


def mel_spectrogram_to_audio(
    mel_spec: np.ndarray,
    mel_basis_pinv: np.ndarray,
    config: VocoderConfig,
    length: Optional[int] = None,
) -> np.ndarray:
    """Convert mel spectrogram to audio via pseudo-inverse + Griffin-Lim.

    Args:
        mel_spec: (n_mels, n_frames) mel spectrogram (linear scale)
        mel_basis_pinv: (n_fft // 2 + 1, n_mels) pseudo-inverse of mel filterbank
        config: VocoderConfig
        length: desired output length

    Returns:
        (T,) audio signal
    """
    linear = np.maximum(mel_basis_pinv @ mel_spec, 0)
    return griffin_lim(
        linear,
        n_iter=config.n_iter,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=config.window,
        length=length,
    )


# ============================================================
# Helpers
# ============================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    pos = x >= 0
    z = np.zeros_like(x)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    z[~pos] = exp_x / (1.0 + exp_x)
    return z


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-10)
