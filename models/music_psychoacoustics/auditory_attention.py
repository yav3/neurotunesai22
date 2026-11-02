"""Auditory attention and perceptual salience in music.

Models selective auditory attention via salience-driven filtering:
spectral surprise (Itti & Koch 2001 adapted for audio), temporal
novelty detection, stream-based attention switching (Bregman 1990),
and the attentional blink in rapid auditory sequences. Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class SalienceMap:
    times: np.ndarray
    frequencies: np.ndarray
    salience: np.ndarray
    peak_times: np.ndarray
    peak_freqs: np.ndarray


@dataclass
class AttentionState:
    focus_frequency: float
    bandwidth: float
    gain: float
    time_since_switch: float
    fatigue: float


@dataclass
class AttentionalBlinkResult:
    t1_detected: bool
    t2_detected: bool
    lag: int
    deficit: float


def spectral_surprise(spectrum_current: np.ndarray,
                      spectrum_context: np.ndarray,
                      epsilon: float = 1e-10) -> np.ndarray:
    """KL divergence per frequency bin: how surprising current frame is relative to context."""
    p = np.maximum(spectrum_current, epsilon)
    q = np.maximum(spectrum_context, epsilon)
    p = p / p.sum()
    q = q / q.sum()
    return p * np.log2(p / q)


def spectral_contrast_salience(spectrogram: np.ndarray,
                                context_frames: int = 10) -> np.ndarray:
    """Compute salience as spectral contrast against running context.

    Returns (n_freq, n_time) salience map. High values = spectrally novel events.
    """
    n_freq, n_time = spectrogram.shape
    salience = np.zeros_like(spectrogram)
    for t in range(n_time):
        start = max(0, t - context_frames)
        if start == t:
            continue
        context = spectrogram[:, start:t].mean(axis=1)
        salience[:, t] = np.maximum(spectrogram[:, t] - context, 0)
    norm = salience.max()
    if norm > 1e-10:
        salience /= norm
    return salience


def temporal_novelty(feature_sequence: np.ndarray,
                     kernel_size: int = 8) -> np.ndarray:
    """Detect novel events via checkerboard kernel on self-similarity matrix.

    Foote (2000) novelty detection: convolve a Gaussian checkerboard kernel
    along the diagonal of the self-similarity matrix.
    """
    n = len(feature_sequence)
    if n < 2:
        return np.zeros(n)

    if feature_sequence.ndim == 1:
        feature_sequence = feature_sequence[:, np.newaxis]

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = np.linalg.norm(feature_sequence[i] - feature_sequence[j])
            sim[i, j] = np.exp(-d ** 2 / (2.0 * np.median(
                np.abs(np.diff(feature_sequence, axis=0)).ravel() + 1e-10) ** 2))

    half = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            sign = 1.0 if (i < half) == (j < half) else -1.0
            gi = np.exp(-((i - half) ** 2) / (2 * (half / 2) ** 2))
            gj = np.exp(-((j - half) ** 2) / (2 * (half / 2) ** 2))
            kernel[i, j] = sign * gi * gj

    novelty = np.zeros(n)
    for t in range(half, n - half):
        patch = sim[t - half:t + half, t - half:t + half]
        if patch.shape == kernel.shape:
            novelty[t] = np.abs(np.sum(patch * kernel))

    norm = novelty.max()
    if norm > 1e-10:
        novelty /= norm
    return novelty


def onset_salience(onset_strengths: np.ndarray,
                   expected_period: float,
                   sr_onset: float) -> np.ndarray:
    """Salience of onsets: syncopated/unexpected onsets are more salient.

    Onsets that deviate from the expected metrical grid are weighted higher.
    """
    n = len(onset_strengths)
    if expected_period <= 0 or n == 0:
        return onset_strengths.copy()

    period_samples = expected_period * sr_onset
    if period_samples < 1:
        return onset_strengths.copy()

    grid_phase = np.mod(np.arange(n), period_samples) / period_samples
    on_beat = np.minimum(grid_phase, 1.0 - grid_phase) * 2.0
    syncopation_weight = 0.5 + 0.5 * on_beat
    return onset_strengths * syncopation_weight


def attention_filter(spectrum: np.ndarray,
                     frequencies: np.ndarray,
                     state: AttentionState) -> np.ndarray:
    """Apply attentional filter: boost around focus frequency, attenuate elsewhere."""
    distance = np.abs(frequencies - state.focus_frequency)
    filter_shape = np.exp(-0.5 * (distance / max(state.bandwidth, 1.0)) ** 2)
    effective_gain = state.gain * (1.0 - state.fatigue)
    attended = spectrum * (1.0 + effective_gain * filter_shape)
    return attended


def attention_switch_cost(old_freq: float, new_freq: float,
                          time_at_old: float,
                          switch_rate: float = 2.0) -> float:
    """Cost of switching attention between frequency regions.

    Larger frequency distance and shorter dwell time increase cost.
    Based on Cusack et al. (2004) auditory attention switching.
    """
    freq_distance = abs(np.log2(max(new_freq, 1) / max(old_freq, 1)))
    dwell_bonus = 1.0 - np.exp(-time_at_old / switch_rate)
    return float(freq_distance * (1.0 - 0.5 * dwell_bonus))


def update_attention(state: AttentionState,
                     salience_map_frame: np.ndarray,
                     frequencies: np.ndarray,
                     dt: float,
                     fatigue_rate: float = 0.05,
                     recovery_rate: float = 0.02,
                     switch_threshold: float = 0.6) -> AttentionState:
    """Update attention state based on current salience frame.

    Attention fatigues over time at the current focus; switches when
    a competing region's salience exceeds threshold adjusted for switch cost.
    """
    current_idx = np.argmin(np.abs(frequencies - state.focus_frequency))
    current_salience = salience_map_frame[current_idx]

    peak_idx = np.argmax(salience_map_frame)
    peak_salience = salience_map_frame[peak_idx]
    peak_freq = frequencies[peak_idx]

    new_fatigue = min(1.0, state.fatigue + fatigue_rate * dt)

    cost = attention_switch_cost(state.focus_frequency, peak_freq,
                                 state.time_since_switch)
    effective_threshold = switch_threshold + 0.3 * cost

    if (peak_salience > current_salience * (1.0 + effective_threshold) and
            peak_idx != current_idx):
        return AttentionState(
            focus_frequency=peak_freq,
            bandwidth=state.bandwidth,
            gain=state.gain,
            time_since_switch=0.0,
            fatigue=max(0, new_fatigue - recovery_rate * 5),
        )

    return AttentionState(
        focus_frequency=state.focus_frequency,
        bandwidth=state.bandwidth,
        gain=state.gain,
        time_since_switch=state.time_since_switch + dt,
        fatigue=new_fatigue,
    )


def compute_salience_map(spectrogram: np.ndarray,
                         frequencies: np.ndarray,
                         times: np.ndarray,
                         context_frames: int = 10,
                         n_peaks: int = 10) -> SalienceMap:
    """Full salience map from spectrogram with peak extraction."""
    salience = spectral_contrast_salience(spectrogram, context_frames)

    flat_indices = np.argsort(salience.ravel())[-n_peaks:]
    peak_freq_idx, peak_time_idx = np.unravel_index(flat_indices, salience.shape)

    return SalienceMap(
        times=times,
        frequencies=frequencies,
        salience=salience,
        peak_times=times[peak_time_idx] if len(times) > 0 else np.array([]),
        peak_freqs=frequencies[peak_freq_idx] if len(frequencies) > 0 else np.array([]),
    )


def attentional_blink(target_saliencies: np.ndarray,
                      soa_ms: float,
                      t1_threshold: float = 0.5,
                      blink_duration_ms: float = 500.0,
                      blink_depth: float = 0.4) -> List[AttentionalBlinkResult]:
    """Model the attentional blink in rapid auditory presentation.

    After detecting T1, there's a ~200-500ms window where T2 detection
    is impaired (Raymond et al. 1992; Mondor 1998 for auditory).
    """
    results = []
    n = len(target_saliencies)
    blink_samples = int(blink_duration_ms / soa_ms)

    for i in range(n):
        t1_detected = target_saliencies[i] > t1_threshold
        if not t1_detected:
            continue
        for lag in range(1, min(blink_samples + 3, n - i)):
            j = i + lag
            if j >= n:
                break
            blink_factor = blink_depth * np.exp(-((lag * soa_ms - 250) ** 2) /
                                                  (2 * 100 ** 2))
            effective_threshold = t1_threshold * (1.0 + blink_factor)
            t2_detected = target_saliencies[j] > effective_threshold
            deficit = max(0, effective_threshold - t1_threshold) / max(t1_threshold, 1e-10)
            results.append(AttentionalBlinkResult(
                t1_detected=True,
                t2_detected=t2_detected,
                lag=lag,
                deficit=float(deficit),
            ))
    return results


def informational_masking(target_spectrum: np.ndarray,
                          masker_spectrum: np.ndarray,
                          frequencies: np.ndarray,
                          uncertainty: float = 0.5) -> np.ndarray:
    """Informational masking: masking beyond energetic (peripheral) masking.

    High uncertainty about masker composition increases informational masking.
    Durlach et al. (2003).
    """
    energetic = np.maximum(masker_spectrum - target_spectrum, 0)
    spectral_complexity = np.std(masker_spectrum) / (np.mean(masker_spectrum) + 1e-10)
    info_factor = uncertainty * spectral_complexity
    informational = energetic * (1.0 + info_factor)
    return np.minimum(informational, target_spectrum)


def streaming_attention_trajectory(
    frequencies_over_time: np.ndarray,
    amplitudes_over_time: np.ndarray,
    initial_focus: float = 440.0,
    dt: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate attention trajectory over a sequence of spectral frames.

    Returns (focus_frequencies, fatigue_values) over time.
    """
    n_time = frequencies_over_time.shape[1] if frequencies_over_time.ndim == 2 else len(frequencies_over_time)
    freqs = frequencies_over_time[:, 0] if frequencies_over_time.ndim == 2 else frequencies_over_time

    state = AttentionState(
        focus_frequency=initial_focus,
        bandwidth=100.0,
        gain=2.0,
        time_since_switch=0.0,
        fatigue=0.0,
    )

    focus_history = np.zeros(n_time)
    fatigue_history = np.zeros(n_time)

    for t in range(n_time):
        if frequencies_over_time.ndim == 2:
            frame_salience = amplitudes_over_time[:, t]
            frame_freqs = frequencies_over_time[:, 0]
        else:
            frame_salience = amplitudes_over_time
            frame_freqs = frequencies_over_time

        state = update_attention(state, frame_salience, frame_freqs, dt)
        focus_history[t] = state.focus_frequency
        fatigue_history[t] = state.fatigue

    return focus_history, fatigue_history


def perceptual_importance(spectrogram: np.ndarray,
                          frequencies: np.ndarray,
                          a_weighting: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-frame perceptual importance combining loudness weighting and spectral novelty.

    Returns (n_time,) importance curve.
    """
    n_freq, n_time = spectrogram.shape

    if a_weighting is not None:
        weighted = spectrogram * a_weighting[:, np.newaxis]
    else:
        weighted = spectrogram

    loudness = np.sum(weighted ** 2, axis=0)
    loudness_norm = loudness / (loudness.max() + 1e-10)

    novelty = np.zeros(n_time)
    for t in range(1, n_time):
        novelty[t] = np.sum(np.abs(weighted[:, t] - weighted[:, t - 1]))
    novelty_norm = novelty / (novelty.max() + 1e-10)

    return 0.6 * loudness_norm + 0.4 * novelty_norm
