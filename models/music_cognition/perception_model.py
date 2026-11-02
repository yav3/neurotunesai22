"""Deep psychoacoustic perception model: signal → auditory processing → musical cognition.

Implements computational models of human auditory perception grounded in
psychoacoustic research:
  - Gammatone auditory filterbank (Patterson et al., 1992)
  - Pitch perception via summary autocorrelation (Meddis & Hewitt, 1991)
  - Loudness model following ISO 532-1 / Zwicker (1961)
  - Timbre perception: brightness, roughness, sharpness (Grey, 1977; Sethares, 2005)
  - Rhythm: beat induction, metrical hierarchy, groove (Madison et al., 2011)
  - Harmony: sensory consonance (Plomp & Levelt, 1965), tonal tension (Lerdahl, 2001)
  - Auditory scene analysis / stream segregation (Bregman, 1990)

Pure numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------- auditory filterbank ----------

def erb_rate(f: np.ndarray) -> np.ndarray:
    """Frequency (Hz) → ERB-rate scale (Glasberg & Moore, 1990)."""
    return 21.4 * np.log10(4.37e-3 * np.asarray(f, dtype=float) + 1.0)


def erb_rate_inv(e: np.ndarray) -> np.ndarray:
    """ERB-rate → frequency (Hz)."""
    return (10 ** (np.asarray(e, dtype=float) / 21.4) - 1.0) / 4.37e-3


def erb_bandwidth(cf: np.ndarray) -> np.ndarray:
    """ERB bandwidth at center frequency (Glasberg & Moore, 1990)."""
    return 24.7 * (4.37e-3 * np.asarray(cf, dtype=float) + 1.0)


def gammatone_impulse_response(
    t: np.ndarray,
    cf: float,
    order: int = 4,
    bandwidth_factor: float = 1.019,
) -> np.ndarray:
    """Gammatone filter impulse response (Patterson et al., 1992)."""
    b = bandwidth_factor * 2 * np.pi * erb_bandwidth(np.array([cf]))[0]
    return t ** (order - 1) * np.exp(-b * t) * np.cos(2 * np.pi * cf * t)


class AuditoryFilterbank:
    """Patterson/Glasberg ERB gammatone filterbank."""

    def __init__(self, sr: int = 22050, n_channels: int = 64, f_lo: float = 50, f_hi: float = 8000):
        self.sr = sr
        self.n_channels = n_channels
        erb_lo = erb_rate(np.array([f_lo]))[0]
        erb_hi = erb_rate(np.array([f_hi]))[0]
        erb_pts = np.linspace(erb_lo, erb_hi, n_channels)
        self.center_freqs = erb_rate_inv(erb_pts)
        self._build_filters()

    def _build_filters(self, duration: float = 0.025) -> None:
        n = int(self.sr * duration)
        t = np.arange(n) / self.sr
        self.filters = np.zeros((self.n_channels, n))
        for i, cf in enumerate(self.center_freqs):
            ir = gammatone_impulse_response(t, cf)
            self.filters[i] = ir / (np.max(np.abs(ir)) + 1e-10)

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply filterbank. Returns (n_channels, T)."""
        from scipy.signal import fftconvolve
        output = np.zeros((self.n_channels, len(signal)))
        for i in range(self.n_channels):
            filtered = fftconvolve(signal, self.filters[i], mode='same')
            output[i] = filtered
        return output

    @staticmethod
    def neural_transduction(bm_output: np.ndarray, compression: float = 0.3) -> np.ndarray:
        """Half-wave rectification + power-law compression."""
        rectified = np.maximum(bm_output, 0)
        return rectified ** compression

    @staticmethod
    def adaptation(
        excitation: np.ndarray,
        sr: int = 22050,
        tau_fast: float = 5e-3,
        tau_slow: float = 50e-3,
    ) -> np.ndarray:
        """Rapid + short-term adaptation (Westerman & Smith, 1988)."""
        alpha_fast = 1.0 / (tau_fast * sr)
        alpha_slow = 1.0 / (tau_slow * sr)
        n_ch, T = excitation.shape
        adapted = np.zeros_like(excitation)
        state_fast = np.zeros(n_ch)
        state_slow = np.zeros(n_ch)
        for t in range(T):
            state_fast += alpha_fast * (excitation[:, t] - state_fast)
            state_slow += alpha_slow * (excitation[:, t] - state_slow)
            adapted[:, t] = excitation[:, t] / (state_slow + 1e-10)
        return adapted


# ---------- pitch perception ----------

class PitchPerception:
    """Autocorrelation model of pitch (Meddis & Hewitt, 1991)."""

    def __init__(self, sr: int = 22050, fmin: float = 50, fmax: float = 2000):
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.lag_min = int(sr / fmax)
        self.lag_max = int(sr / fmin)

    def summary_autocorrelation(self, nerve_output: np.ndarray) -> np.ndarray:
        """Across-channel summary ACF from auditory nerve output (n_channels, T)."""
        n_ch, T = nerve_output.shape
        max_lag = min(self.lag_max, T - 1)
        sacf = np.zeros(max_lag + 1)
        for ch in range(n_ch):
            x = nerve_output[ch]
            for lag in range(self.lag_min, max_lag + 1):
                sacf[lag] += np.sum(x[:T - lag] * x[lag:])
        sacf /= n_ch * max(T - max_lag, 1)
        return sacf

    def pitch_candidates(self, sacf: np.ndarray) -> List[Tuple[float, float]]:
        """Extract pitch candidates from summary ACF. Returns [(freq, salience), ...]."""
        candidates = []
        for i in range(self.lag_min + 1, min(self.lag_max, len(sacf) - 1)):
            if sacf[i] > sacf[i - 1] and sacf[i] > sacf[i + 1]:
                freq = self.sr / i
                if self.fmin <= freq <= self.fmax:
                    candidates.append((freq, float(sacf[i])))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:5]

    def pitch_strength(self, signal: np.ndarray) -> Tuple[float, float]:
        """Estimate pitch and salience from a signal frame."""
        n = len(signal)
        acf = np.correlate(signal, signal, mode='full')[n - 1:]
        acf /= acf[0] + 1e-10
        candidates = []
        for i in range(self.lag_min, min(self.lag_max, n - 1)):
            if i > 0 and i < len(acf) - 1:
                if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
                    candidates.append((self.sr / i, float(acf[i])))
        if not candidates:
            return 0.0, 0.0
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    @staticmethod
    def virtual_pitch(harmonics: np.ndarray, amplitudes: np.ndarray) -> float:
        """Missing fundamental: estimate f0 from partial frequencies."""
        if len(harmonics) < 2:
            return float(harmonics[0]) if len(harmonics) > 0 else 0.0
        diffs = np.diff(np.sort(harmonics))
        weighted_diff = np.average(diffs, weights=amplitudes[:-1] + amplitudes[1:])
        return float(weighted_diff)

    @staticmethod
    def octave_equivalence(f1: float, f2: float) -> float:
        """Chroma similarity: 1.0 if same pitch class, 0.0 if maximally different."""
        if f1 <= 0 or f2 <= 0:
            return 0.0
        cents = 1200 * np.log2(f2 / f1)
        chroma_diff = cents % 1200
        return float(np.cos(2 * np.pi * chroma_diff / 1200) * 0.5 + 0.5)


# ---------- loudness model ----------

class LoudnessModel:
    """Loudness perception (simplified Zwicker, 1961; Glasberg & Moore, 2002)."""

    @staticmethod
    def bark_scale(f: np.ndarray) -> np.ndarray:
        """Frequency → Bark scale (Traunmüller, 1990)."""
        f = np.asarray(f, dtype=float)
        return 26.81 * f / (1960 + f) - 0.53

    @staticmethod
    def specific_loudness(bark_spectrum: np.ndarray, threshold_quiet: float = 0.0) -> np.ndarray:
        """Specific loudness per critical band (sone/Bark)."""
        excitation = np.maximum(bark_spectrum - threshold_quiet, 0)
        return excitation ** 0.23

    @staticmethod
    def total_loudness(specific_loudness: np.ndarray) -> float:
        """Total loudness in sone by integration over Bark bands."""
        return float(np.sum(specific_loudness) * 0.5)

    def loudness_contour(
        self,
        signal: np.ndarray,
        sr: int,
        hop: int = 512,
        n_fft: int = 2048,
    ) -> np.ndarray:
        """Temporal loudness trajectory in sone."""
        n_bark = 24
        bark_edges = np.linspace(0, 24, n_bark + 1)
        freq_edges = 1960 * (bark_edges + 0.53) / (26.81 - bark_edges - 0.53)
        freq_edges = np.clip(freq_edges, 0, sr / 2)
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
        n_frames = max(1, (len(signal) - n_fft) // hop)
        loudness = np.zeros(n_frames)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        for i in range(n_frames):
            frame = signal[i * hop:i * hop + n_fft] * window
            spectrum = np.abs(np.fft.rfft(frame)) ** 2
            bark_spec = np.zeros(n_bark)
            for b in range(n_bark):
                mask = (freqs >= freq_edges[b]) & (freqs < freq_edges[b + 1])
                bark_spec[b] = spectrum[mask].sum() if mask.any() else 0.0
            sl = self.specific_loudness(bark_spec)
            loudness[i] = self.total_loudness(sl)
        return loudness

    @staticmethod
    def equal_loudness_weight(f: float) -> float:
        """Approximate equal-loudness weighting at 60 phon (ISO 226:2003 simplified)."""
        if f < 20:
            return 0.0
        f2 = f * f
        num = 12194 ** 2 * f2 ** 2
        den = (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2)
        return float(num / (den + 1e-10))

    def dynamic_loudness(
        self,
        signal: np.ndarray,
        sr: int,
        attack_ms: float = 22.0,
        release_ms: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Short-term and long-term loudness (Glasberg & Moore, 2002)."""
        instantaneous = self.loudness_contour(signal, sr)
        hop_s = 512 / sr
        alpha_a = 1 - np.exp(-hop_s / (attack_ms / 1000))
        alpha_r = 1 - np.exp(-hop_s / (release_ms / 1000))
        stl = np.zeros_like(instantaneous)
        ltl = np.zeros_like(instantaneous)
        for i in range(1, len(instantaneous)):
            alpha = alpha_a if instantaneous[i] > stl[i - 1] else alpha_r
            stl[i] = stl[i - 1] + alpha * (instantaneous[i] - stl[i - 1])
            ltl[i] = ltl[i - 1] + 0.05 * (stl[i] - ltl[i - 1])
        return stl, ltl


# ---------- timbre perception ----------

class TimbrePerception:
    """Multidimensional timbre descriptors (Grey, 1977; McAdams et al., 1995)."""

    @staticmethod
    def spectral_centroid(spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """Brightness: amplitude-weighted mean frequency."""
        total = np.sum(spectrum)
        if total < 1e-10:
            return 0.0
        return float(np.sum(freqs * spectrum) / total)

    @staticmethod
    def log_attack_time(envelope: np.ndarray, sr: int, threshold: float = 0.1) -> float:
        """Log10 of attack duration in seconds."""
        peak = np.argmax(envelope)
        above = np.where(envelope[:peak + 1] > threshold * envelope[peak])[0]
        if len(above) < 2:
            return -3.0
        attack_samples = above[-1] - above[0]
        return float(np.log10(max(attack_samples / sr, 1e-6)))

    @staticmethod
    def spectral_irregularity(harmonic_amplitudes: np.ndarray) -> float:
        """Jensen (1999) spectral irregularity."""
        if len(harmonic_amplitudes) < 3:
            return 0.0
        a = harmonic_amplitudes
        irregularity = 0.0
        for k in range(1, len(a) - 1):
            irregularity += abs(a[k] - (a[k - 1] + a[k] + a[k + 1]) / 3)
        return float(irregularity / np.sum(a + 1e-10))

    @staticmethod
    def roughness(signal: np.ndarray, sr: int, n_fft: int = 4096) -> float:
        """Sensory dissonance / roughness (Sethares, 2005).

        Based on interference between frequency components.
        """
        spectrum = np.abs(np.fft.rfft(signal, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        threshold = spectrum.max() * 0.01
        peaks = np.where(spectrum > threshold)[0]
        if len(peaks) < 2:
            return 0.0

        peak_freqs = freqs[peaks]
        peak_amps = spectrum[peaks]
        top_n = min(30, len(peaks))
        top_idx = np.argsort(peak_amps)[-top_n:]
        pf = peak_freqs[top_idx]
        pa = peak_amps[top_idx]

        total_roughness = 0.0
        for i in range(len(pf)):
            for j in range(i + 1, len(pf)):
                diff = abs(pf[i] - pf[j])
                s_star = 0.24 * (0.021 * min(pf[i], pf[j]) + 19)
                if s_star < 1e-6:
                    continue
                x = diff / s_star
                d = np.exp(-3.5 * x) - np.exp(-5.75 * x)
                total_roughness += pa[i] * pa[j] * max(d, 0)

        return float(total_roughness)

    @staticmethod
    def sharpness(specific_loudness: np.ndarray) -> float:
        """Zwicker's sharpness (acum): weighted spectral centroid of specific loudness."""
        n = len(specific_loudness)
        z = np.arange(n) * 24.0 / n
        weights = np.ones(n)
        high = z > 15.0
        weights[high] = 0.066 * np.exp(0.171 * z[high])
        N_total = np.sum(specific_loudness) + 1e-10
        return float(0.11 * np.sum(specific_loudness * weights * z) / N_total)


# ---------- rhythm perception ----------

class RhythmPerception:
    """Rhythm, beat, and groove perception models."""

    @staticmethod
    def onset_strength(signal: np.ndarray, sr: int, hop: int = 512, n_fft: int = 2048) -> np.ndarray:
        """Perceptual onset detection function (spectral flux + weighting)."""
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
        n_frames = max(1, (len(signal) - n_fft) // hop)
        onset = np.zeros(n_frames)
        prev_spec = None
        for i in range(n_frames):
            frame = signal[i * hop:i * hop + n_fft] * window
            spec = np.abs(np.fft.rfft(frame))
            if prev_spec is not None:
                diff = spec - prev_spec
                onset[i] = np.sum(np.maximum(diff, 0))
            prev_spec = spec
        return onset / (onset.max() + 1e-10)

    @staticmethod
    def beat_period_acf(onset_fn: np.ndarray, sr: int, hop: int = 512) -> Tuple[float, np.ndarray]:
        """Estimate beat period via onset function autocorrelation."""
        acf = np.correlate(onset_fn, onset_fn, mode='full')
        acf = acf[len(onset_fn) - 1:]
        acf /= acf[0] + 1e-10
        bpm_min, bpm_max = 40, 240
        lag_min = int(60 * sr / (bpm_max * hop))
        lag_max = int(60 * sr / (bpm_min * hop))
        lag_max = min(lag_max, len(acf) - 1)
        search = acf[lag_min:lag_max + 1]
        if len(search) == 0:
            return 120.0, acf
        best_lag = lag_min + np.argmax(search)
        bpm = 60 * sr / (best_lag * hop)
        return float(bpm), acf

    @staticmethod
    def metrical_strength(position: int, level: int = 4) -> float:
        """Metrical weight at a given position in a hierarchical meter.

        Level 0 = downbeat (strongest), higher = weaker subdivision.
        """
        if position == 0:
            return 1.0
        strength = 0.0
        for lev in range(level, 0, -1):
            period = 2 ** lev
            if position % period == 0:
                strength = lev / level
                break
        return strength

    @staticmethod
    def syncopation_score(
        onsets: np.ndarray,
        meter_period: int = 16,
        levels: int = 4,
    ) -> float:
        """Syncopation: Longuet-Higgins & Lee (1984) model.

        Sum of metrical-weight differences between syncopated notes and following stronger beats.
        """
        score = 0.0
        for i in range(len(onsets)):
            if onsets[i] > 0:
                pos = i % meter_period
                weight_here = RhythmPerception.metrical_strength(pos, levels)
                next_pos = (pos + 1) % meter_period
                weight_next = RhythmPerception.metrical_strength(next_pos, levels)
                if weight_here < weight_next:
                    score += weight_next - weight_here
        return float(score)

    @staticmethod
    def groove_rating(
        onset_times: np.ndarray,
        beat_times: np.ndarray,
    ) -> float:
        """Groove from microtiming deviations (Madison et al., 2011).

        Moderate deviations → high groove; zero or extreme → lower.
        """
        if len(onset_times) == 0 or len(beat_times) == 0:
            return 0.0
        deviations = []
        for onset in onset_times:
            nearest_beat = beat_times[np.argmin(np.abs(beat_times - onset))]
            dev = abs(onset - nearest_beat)
            deviations.append(dev)
        devs = np.array(deviations)
        mean_dev = np.mean(devs)
        optimal_dev = 0.02
        groove = np.exp(-((mean_dev - optimal_dev) ** 2) / (2 * 0.015 ** 2))
        return float(groove)


# ---------- harmony perception ----------

class HarmonyPerception:
    """Consonance, tension, and harmonic expectation models."""

    @staticmethod
    def sensory_consonance(interval_cents: float) -> float:
        """Plomp & Levelt (1965) dissonance curve for a single interval.

        Returns consonance in [0, 1] where 1 = perfectly consonant.
        """
        x = abs(interval_cents) / 100.0
        d = np.exp(-3.5 * x) - np.exp(-5.75 * x)
        return float(1.0 - max(d, 0))

    @staticmethod
    def chord_consonance(pitch_classes: np.ndarray, reference_freq: float = 261.63) -> float:
        """Overall consonance of a chord via pairwise Plomp-Levelt."""
        freqs = reference_freq * 2 ** (pitch_classes / 12.0)
        if len(freqs) < 2:
            return 1.0
        total_consonance = 0.0
        n_pairs = 0
        for i in range(len(freqs)):
            for j in range(i + 1, len(freqs)):
                interval_cents = 1200 * np.log2(freqs[j] / freqs[i])
                total_consonance += HarmonyPerception.sensory_consonance(interval_cents)
                n_pairs += 1
        return float(total_consonance / max(n_pairs, 1))

    @staticmethod
    def tonal_tension(
        chord_pc: np.ndarray,
        key_profile: np.ndarray,
    ) -> float:
        """Simplified tonal tension (Lerdahl, 2001): distance from key profile."""
        chord_dist = np.zeros(12)
        for pc in chord_pc:
            chord_dist[int(pc) % 12] += 1.0
        chord_dist /= chord_dist.sum() + 1e-10
        key_norm = key_profile / (key_profile.sum() + 1e-10)
        distance = np.sqrt(np.sum((chord_dist - key_norm) ** 2))
        return float(distance)

    KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    @staticmethod
    def key_finding(pitch_class_distribution: np.ndarray) -> Tuple[int, str, float]:
        """Krumhansl-Schmuckler key-finding algorithm.

        Returns (tonic_pc, mode, correlation).
        """
        best_corr = -np.inf
        best_key = 0
        best_mode = "major"
        for tonic in range(12):
            rotated = np.roll(pitch_class_distribution, -tonic)
            corr_maj = np.corrcoef(rotated, HarmonyPerception.KRUMHANSL_MAJOR)[0, 1]
            corr_min = np.corrcoef(rotated, HarmonyPerception.KRUMHANSL_MINOR)[0, 1]
            if corr_maj > best_corr:
                best_corr = corr_maj
                best_key = tonic
                best_mode = "major"
            if corr_min > best_corr:
                best_corr = corr_min
                best_key = tonic
                best_mode = "minor"
        return best_key, best_mode, float(best_corr)

    @staticmethod
    def harmonic_expectation(
        context_pcs: List[np.ndarray],
        candidate_pcs: List[np.ndarray],
    ) -> np.ndarray:
        """Simple Bayesian expectation: probability of each candidate chord given context."""
        pc_dist = np.zeros(12)
        for chord in context_pcs:
            for pc in chord:
                pc_dist[int(pc) % 12] += 1.0
        pc_dist /= pc_dist.sum() + 1e-10
        scores = np.zeros(len(candidate_pcs))
        for i, cand in enumerate(candidate_pcs):
            for pc in cand:
                scores[i] += pc_dist[int(pc) % 12]
            scores[i] /= len(cand) + 1e-10
        probs = np.exp(scores)
        probs /= probs.sum() + 1e-10
        return probs


# ---------- stream segregation ----------

class StreamSegregation:
    """Auditory scene analysis (Bregman, 1990)."""

    @staticmethod
    def frequency_proximity(f1: float, f2: float) -> float:
        """Probability of same-stream assignment based on frequency proximity."""
        if f1 <= 0 or f2 <= 0:
            return 0.0
        semitones = 12 * np.log2(max(f1, f2) / min(f1, f2))
        return float(np.exp(-semitones ** 2 / (2 * 3.0 ** 2)))

    @staticmethod
    def temporal_proximity(dt: float) -> float:
        """Same-stream probability based on inter-onset interval."""
        return float(np.exp(-dt / 0.15))

    @staticmethod
    def stream_assignment(
        pitches: np.ndarray,
        onsets: np.ndarray,
    ) -> np.ndarray:
        """Assign notes to streams via greedy nearest-neighbor on pitch × time.

        Returns stream labels (integer array, same length as pitches).
        """
        n = len(pitches)
        if n == 0:
            return np.array([], dtype=int)
        labels = -np.ones(n, dtype=int)
        labels[0] = 0
        n_streams = 1
        stream_last_pitch = {0: pitches[0]}
        stream_last_time = {0: onsets[0]}

        for i in range(1, n):
            best_stream = -1
            best_score = -1.0
            for s in range(n_streams):
                fp = StreamSegregation.frequency_proximity(pitches[i], stream_last_pitch[s])
                tp = StreamSegregation.temporal_proximity(onsets[i] - stream_last_time[s])
                score = fp * tp
                if score > best_score:
                    best_score = score
                    best_stream = s
            if best_score < 0.1:
                best_stream = n_streams
                n_streams += 1
            labels[i] = best_stream
            stream_last_pitch[best_stream] = pitches[i]
            stream_last_time[best_stream] = onsets[i]
        return labels

    @staticmethod
    def fission_boundary(rate: float) -> float:
        """Van Noorden (1975) fission boundary: semitone threshold for stream split."""
        return 3.0 + 2.0 * np.log2(max(rate, 1.0))


# ---------- perceptual distance ----------

class PerceptualDistance:
    """Perceptually-weighted distance metrics between audio representations."""

    @staticmethod
    def spectral_distance(
        spec1: np.ndarray,
        spec2: np.ndarray,
        weighting: Optional[np.ndarray] = None,
    ) -> float:
        """Perceptually-weighted spectral distance."""
        if weighting is None:
            weighting = np.ones(spec1.shape[0])
        diff = (spec1 - spec2) ** 2
        weighted = diff * weighting[:, np.newaxis] if diff.ndim > 1 else diff * weighting
        return float(np.sqrt(np.mean(weighted)))

    @staticmethod
    def melodic_similarity(
        melody1: np.ndarray,
        melody2: np.ndarray,
    ) -> float:
        """Edit distance on interval sequences, normalized to [0, 1]."""
        int1 = np.diff(melody1) if len(melody1) > 1 else np.array([])
        int2 = np.diff(melody2) if len(melody2) > 1 else np.array([])
        n, m = len(int1), len(int2)
        if n == 0 and m == 0:
            return 1.0
        dp = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            dp[i, 0] = i
        for j in range(m + 1):
            dp[0, j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if abs(int1[i - 1] - int2[j - 1]) < 0.5 else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
        max_len = max(n, m)
        return 1.0 - float(dp[n, m] / max_len) if max_len > 0 else 1.0

    @staticmethod
    def rhythmic_similarity(
        pattern1: np.ndarray,
        pattern2: np.ndarray,
    ) -> float:
        """Hamming distance on binary onset patterns, normalized."""
        n = max(len(pattern1), len(pattern2))
        if n == 0:
            return 1.0
        p1 = np.zeros(n)
        p2 = np.zeros(n)
        p1[:len(pattern1)] = pattern1
        p2[:len(pattern2)] = pattern2
        matches = np.sum(p1 == p2)
        return float(matches / n)
