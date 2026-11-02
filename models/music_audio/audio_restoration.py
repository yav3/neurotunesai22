"""Audio restoration: noise reduction, declipping, bandwidth extension, dereverberation."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy import signal as sp_signal


@dataclass
class NoiseProfile:
    spectrum: np.ndarray
    floor_db: float
    sr: int


def estimate_noise_profile(
    noise_segment: np.ndarray,
    sr: int,
    n_fft: int = 2048,
) -> NoiseProfile:
    """Estimate noise spectrum from a known-noise segment."""
    seg = np.asarray(noise_segment, dtype=float)
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    hop = n_fft // 4
    n_frames = max(1, (len(seg) - n_fft) // hop + 1)
    accum = np.zeros(n_fft // 2 + 1)
    window = np.hanning(n_fft)
    for i in range(n_frames):
        frame = seg[i * hop : i * hop + n_fft] * window
        spec = np.abs(np.fft.rfft(frame)) ** 2
        accum += spec
    accum /= n_frames
    floor_db = float(10 * np.log10(np.mean(accum) + 1e-12))
    return NoiseProfile(spectrum=accum, floor_db=floor_db, sr=sr)


def spectral_subtraction(
    audio: np.ndarray,
    noise_profile: NoiseProfile,
    oversubtract: float = 2.0,
    floor_factor: float = 0.01,
    n_fft: int = 2048,
) -> np.ndarray:
    """Spectral subtraction noise reduction (Boll 1979).

    Parameters
    ----------
    audio : array
        Noisy input signal.
    noise_profile : NoiseProfile
        Estimated noise power spectrum.
    oversubtract : float
        Over-subtraction factor alpha (>1 reduces more noise at cost of distortion).
    floor_factor : float
        Spectral floor to prevent musical noise artifacts.
    """
    x = np.asarray(audio, dtype=float)
    hop = n_fft // 4
    window = np.hanning(n_fft)
    n_frames = max(1, (len(x) - n_fft) // hop + 1)
    out = np.zeros(len(x))
    win_sum = np.zeros(len(x))
    noise_pow = noise_profile.spectrum
    if len(noise_pow) != n_fft // 2 + 1:
        noise_pow = np.interp(
            np.linspace(0, 1, n_fft // 2 + 1),
            np.linspace(0, 1, len(noise_pow)),
            noise_pow,
        )
    floor_level = floor_factor * noise_pow
    for i in range(n_frames):
        start = i * hop
        frame = np.zeros(n_fft)
        end = min(start + n_fft, len(x))
        frame[:end - start] = x[start:end]
        frame *= window
        X = np.fft.rfft(frame)
        mag_sq = np.abs(X) ** 2
        clean_mag_sq = mag_sq - oversubtract * noise_pow
        clean_mag_sq = np.maximum(clean_mag_sq, floor_level)
        gain = np.sqrt(clean_mag_sq / (mag_sq + 1e-12))
        Y = X * gain
        y_frame = np.fft.irfft(Y, n=n_fft)
        out[start:start + n_fft] += y_frame * window
        win_sum[start:start + n_fft] += window ** 2
    mask = win_sum > 1e-8
    out[mask] /= win_sum[mask]
    return out[:len(audio)]


def wiener_filter(
    audio: np.ndarray,
    noise_profile: NoiseProfile,
    n_fft: int = 2048,
    beta: float = 0.98,
) -> np.ndarray:
    """Frequency-domain Wiener filter for noise reduction.

    H(f) = max(1 - noise_pow / signal_pow, beta_floor)
    """
    x = np.asarray(audio, dtype=float)
    hop = n_fft // 4
    window = np.hanning(n_fft)
    n_frames = max(1, (len(x) - n_fft) // hop + 1)
    out = np.zeros(len(x))
    win_sum = np.zeros(len(x))
    noise_pow = noise_profile.spectrum
    if len(noise_pow) != n_fft // 2 + 1:
        noise_pow = np.interp(
            np.linspace(0, 1, n_fft // 2 + 1),
            np.linspace(0, 1, len(noise_pow)),
            noise_pow,
        )
    beta_floor = 1.0 - beta
    for i in range(n_frames):
        start = i * hop
        frame = np.zeros(n_fft)
        end = min(start + n_fft, len(x))
        frame[:end - start] = x[start:end]
        frame *= window
        X = np.fft.rfft(frame)
        sig_pow = np.abs(X) ** 2
        H = np.maximum(1.0 - noise_pow / (sig_pow + 1e-12), beta_floor)
        Y = X * H
        y_frame = np.fft.irfft(Y, n=n_fft)
        out[start:start + n_fft] += y_frame * window
        win_sum[start:start + n_fft] += window ** 2
    mask = win_sum > 1e-8
    out[mask] /= win_sum[mask]
    return out[:len(audio)]


def detect_clipping(
    audio: np.ndarray,
    threshold: float = 0.99,
    min_run: int = 2,
) -> list[tuple[int, int]]:
    """Detect clipped regions where consecutive samples hit the rail.

    Returns list of (start, end) sample indices of clipped runs.
    """
    x = np.asarray(audio, dtype=float)
    clipped = np.abs(x) >= threshold
    regions = []
    in_run = False
    run_start = 0
    for i in range(len(clipped)):
        if clipped[i] and not in_run:
            run_start = i
            in_run = True
        elif not clipped[i] and in_run:
            if i - run_start >= min_run:
                regions.append((run_start, i))
            in_run = False
    if in_run and len(x) - run_start >= min_run:
        regions.append((run_start, len(x)))
    return regions


def declip_cubic(
    audio: np.ndarray,
    threshold: float = 0.99,
    min_run: int = 2,
) -> np.ndarray:
    """Restore clipped regions using cubic interpolation from boundary samples."""
    x = np.array(audio, dtype=float)
    regions = detect_clipping(x, threshold, min_run)
    for start, end in regions:
        left = max(0, start - 1)
        right = min(len(x) - 1, end)
        if left == start and right == end:
            continue
        n = end - start
        t = np.linspace(0, 1, n + 2)
        y_left = x[left]
        y_right = x[right]
        slope_left = x[left] - x[max(0, left - 1)] if left > 0 else 0
        slope_right = x[min(len(x) - 1, right + 1)] - x[right] if right < len(x) - 1 else 0
        a = 2 * y_left - 2 * y_right + slope_left + slope_right
        b = -3 * y_left + 3 * y_right - 2 * slope_left - slope_right
        c = slope_left
        d = y_left
        interp = a * t[1:-1] ** 3 + b * t[1:-1] ** 2 + c * t[1:-1] + d
        x[start:end] = interp
    return x


def bandwidth_extension(
    audio: np.ndarray,
    sr: int,
    target_bandwidth: float | None = None,
    n_fft: int = 2048,
) -> np.ndarray:
    """Extend bandwidth by synthesizing harmonics from existing content.

    Adds harmonically-related spectral content above the detected cutoff
    frequency using spectral mirroring with decay.
    """
    x = np.asarray(audio, dtype=float)
    if target_bandwidth is None:
        target_bandwidth = sr / 2.0
    X = np.fft.rfft(x, n=max(n_fft, len(x)))
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(max(n_fft, len(x)), 1.0 / sr)
    mean_mag = np.mean(mag)
    cutoff_bin = len(mag) - 1
    for i in range(len(mag) - 1, 0, -1):
        if mag[i] > mean_mag * 0.01:
            cutoff_bin = i
            break
    cutoff_freq = freqs[cutoff_bin] if cutoff_bin < len(freqs) else sr / 2
    if cutoff_freq >= target_bandwidth * 0.95:
        return x
    source_band = mag[cutoff_bin // 2 : cutoff_bin]
    if len(source_band) == 0:
        return x
    target_start = cutoff_bin
    target_end = min(len(mag), int(target_bandwidth / (sr / 2) * len(mag)))
    n_target = target_end - target_start
    if n_target <= 0:
        return x
    extended = np.interp(
        np.linspace(0, 1, n_target),
        np.linspace(0, 1, len(source_band)),
        source_band,
    )
    decay = np.exp(-2.0 * np.linspace(0, 1, n_target))
    extended *= decay * 0.3
    rng = np.random.default_rng(42)
    phases = rng.uniform(-np.pi, np.pi, n_target)
    X_out = X.copy()
    X_out[target_start:target_end] += extended * np.exp(1j * phases)
    result = np.fft.irfft(X_out, n=max(n_fft, len(x)))
    return result[:len(x)]


def estimate_reverb_tail(
    impulse_response: np.ndarray,
    sr: int,
    threshold_db: float = -60.0,
) -> float:
    """Estimate RT60 (time for reverb to decay 60dB) from an impulse response."""
    ir = np.asarray(impulse_response, dtype=float)
    if len(ir) == 0:
        return 0.0
    energy = ir ** 2
    cumsum = np.cumsum(energy[::-1])[::-1]
    cumsum = np.maximum(cumsum, 1e-12)
    db = 10 * np.log10(cumsum / cumsum[0])
    below = np.where(db <= threshold_db)[0]
    if len(below) == 0:
        return len(ir) / sr
    return below[0] / sr


def dereverberate(
    audio: np.ndarray,
    sr: int,
    rt60_estimate: float = 0.5,
    n_fft: int = 2048,
    alpha: float = 0.9,
) -> np.ndarray:
    """Simple late-reverb suppression via exponential decay model.

    Estimates the late reverberation power as an exponentially decayed
    version of earlier frames, then applies Wiener-like suppression.
    """
    x = np.asarray(audio, dtype=float)
    hop = n_fft // 4
    window = np.hanning(n_fft)
    n_frames = max(1, (len(x) - n_fft) // hop + 1)
    n_bins = n_fft // 2 + 1
    specs = np.zeros((n_frames, n_bins), dtype=complex)
    for i in range(n_frames):
        start = i * hop
        frame = np.zeros(n_fft)
        end = min(start + n_fft, len(x))
        frame[:end - start] = x[start:end]
        specs[i] = np.fft.rfft(frame * window)
    mag_sq = np.abs(specs) ** 2
    decay_frames = max(1, int(rt60_estimate * sr / hop / 10))
    gamma = np.exp(-3.0 * np.log(10) / max(decay_frames, 1))
    reverb_est = np.zeros_like(mag_sq)
    for i in range(1, n_frames):
        reverb_est[i] = gamma * reverb_est[i - 1] + (1 - gamma) * mag_sq[i - 1]
    gain = np.maximum(1.0 - alpha * reverb_est / (mag_sq + 1e-12), 0.1)
    out = np.zeros(len(x))
    win_sum = np.zeros(len(x))
    for i in range(n_frames):
        Y = specs[i] * gain[i]
        y_frame = np.fft.irfft(Y, n=n_fft)
        start = i * hop
        out[start:start + n_fft] += y_frame * window
        win_sum[start:start + n_fft] += window ** 2
    mask = win_sum > 1e-8
    out[mask] /= win_sum[mask]
    return out[:len(audio)]


def adaptive_noise_gate(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    hold_ms: float = 20.0,
) -> np.ndarray:
    """Adaptive noise gate with attack/hold/release envelope."""
    x = np.asarray(audio, dtype=float)
    if len(x) == 0:
        return x.copy()
    frame_len = int(sr * 0.01)
    hop = frame_len // 2
    threshold_lin = 10 ** (threshold_db / 20.0)
    attack_samples = int(sr * attack_ms / 1000)
    release_samples = int(sr * release_ms / 1000)
    hold_samples = int(sr * hold_ms / 1000)
    envelope = np.zeros(len(x))
    for i in range(0, len(x) - frame_len, hop):
        rms = np.sqrt(np.mean(x[i:i + frame_len] ** 2))
        level = 1.0 if rms >= threshold_lin else 0.0
        envelope[i:i + frame_len] = np.maximum(envelope[i:i + frame_len], level)
    gate = np.zeros(len(x))
    state = 0.0
    hold_counter = 0
    for i in range(len(x)):
        if envelope[i] > 0.5:
            hold_counter = hold_samples
            if state < 1.0:
                state = min(state + 1.0 / max(attack_samples, 1), 1.0)
        else:
            if hold_counter > 0:
                hold_counter -= 1
            else:
                state = max(state - 1.0 / max(release_samples, 1), 0.0)
        gate[i] = state
    return x * gate
