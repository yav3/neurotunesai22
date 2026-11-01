"""Deep automated feature extraction: spectral, rhythmic, harmonic, timbral."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _hann_window(n: int) -> np.ndarray:
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


def _stft_frame(signal: np.ndarray, frame_size: int, hop: int, window: np.ndarray) -> np.ndarray:
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    frames = np.zeros((n_frames, frame_size // 2 + 1))
    for i in range(n_frames):
        start = i * hop
        chunk = signal[start:start + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        windowed = chunk * window
        spectrum = np.abs(np.fft.rfft(windowed))
        frames[i] = spectrum
    return frames


class SpectralFeatureExtractor:
    def spectral_centroid(self, magnitude: np.ndarray, freqs: np.ndarray | None = None) -> float:
        if freqs is None:
            freqs = np.arange(len(magnitude), dtype=float)
        total = np.sum(magnitude)
        if total < 1e-12:
            return 0.0
        return float(np.sum(freqs * magnitude) / total)

    def spectral_bandwidth(self, magnitude: np.ndarray, freqs: np.ndarray | None = None,
                            centroid: float | None = None) -> float:
        if freqs is None:
            freqs = np.arange(len(magnitude), dtype=float)
        total = np.sum(magnitude)
        if total < 1e-12:
            return 0.0
        if centroid is None:
            centroid = self.spectral_centroid(magnitude, freqs)
        return float(np.sqrt(np.sum(magnitude * (freqs - centroid) ** 2) / total))

    def spectral_contrast(self, magnitude: np.ndarray, n_bands: int = 6) -> np.ndarray:
        if len(magnitude) < n_bands:
            return np.zeros(n_bands)
        band_size = len(magnitude) // n_bands
        contrasts = np.zeros(n_bands)
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size if i < n_bands - 1 else len(magnitude)
            band = np.sort(magnitude[start:end])
            n = len(band)
            if n < 2:
                continue
            alpha = max(1, n // 5)
            peak = np.mean(band[-alpha:])
            valley = np.mean(band[:alpha])
            contrasts[i] = float(np.log1p(peak) - np.log1p(valley))
        return contrasts

    def spectral_rolloff(self, magnitude: np.ndarray, percentile: float = 0.85) -> float:
        total = np.sum(magnitude)
        if total < 1e-12:
            return 0.0
        cumsum = np.cumsum(magnitude)
        idx = np.searchsorted(cumsum, percentile * total)
        return float(idx) / len(magnitude) if len(magnitude) > 0 else 0.0

    def spectral_flatness(self, magnitude: np.ndarray) -> float:
        mag = magnitude[magnitude > 0]
        if len(mag) == 0:
            return 0.0
        geo_mean = np.exp(np.mean(np.log(mag)))
        arith_mean = np.mean(mag)
        if arith_mean < 1e-12:
            return 0.0
        return float(geo_mean / arith_mean)

    def spectral_flux(self, spec_t: np.ndarray, spec_t1: np.ndarray) -> float:
        diff = spec_t1 - spec_t
        return float(np.sqrt(np.sum(diff ** 2)))

    def spectral_slope(self, magnitude: np.ndarray, freqs: np.ndarray | None = None) -> float:
        if freqs is None:
            freqs = np.arange(len(magnitude), dtype=float)
        if len(freqs) < 2:
            return 0.0
        f_mean = np.mean(freqs)
        m_mean = np.mean(magnitude)
        num = np.sum((freqs - f_mean) * (magnitude - m_mean))
        den = np.sum((freqs - f_mean) ** 2)
        if abs(den) < 1e-12:
            return 0.0
        return float(num / den)

    def spectral_decrease(self, magnitude: np.ndarray) -> float:
        if len(magnitude) < 2:
            return 0.0
        indices = np.arange(1, len(magnitude))
        diffs = magnitude[1:] - magnitude[0]
        denom = np.sum(1.0 / indices)
        if abs(denom) < 1e-12:
            return 0.0
        return float(np.sum(diffs / indices) / denom)

    def spectral_crest(self, magnitude: np.ndarray) -> float:
        mean_val = np.mean(magnitude)
        if mean_val < 1e-12:
            return 0.0
        return float(np.max(magnitude) / mean_val)

    def mfcc(self, signal: np.ndarray, sr: int, n_mfcc: int = 13,
             n_mels: int = 40, frame_size: int = 1024) -> np.ndarray:
        window = _hann_window(frame_size)
        hop = frame_size // 2
        frames = _stft_frame(signal, frame_size, hop, window)
        n_fft = frame_size // 2 + 1
        mel_filters = self._mel_filterbank(sr, n_fft, n_mels)
        mel_spec = frames @ mel_filters.T
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel = np.log(mel_spec)
        n_frames = log_mel.shape[0]
        mfccs = np.zeros((n_frames, n_mfcc))
        for i in range(n_frames):
            dct_out = self._dct_ii(log_mel[i], n_mfcc)
            mfccs[i] = dct_out
        return mfccs

    def _mel_filterbank(self, sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        low_mel = self._hz_to_mel(0)
        high_mel = self._hz_to_mel(sr / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((n_fft) * hz_points / sr).astype(int)
        filters = np.zeros((n_mels, n_fft))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            for j in range(left, min(center, n_fft)):
                if center > left:
                    filters[i, j] = (j - left) / (center - left)
            for j in range(center, min(right, n_fft)):
                if right > center:
                    filters[i, j] = (right - j) / (right - center)
        return filters

    @staticmethod
    def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    @staticmethod
    def _dct_ii(x: np.ndarray, n_out: int) -> np.ndarray:
        n = len(x)
        result = np.zeros(n_out)
        for k in range(n_out):
            result[k] = np.sum(x * np.cos(np.pi * k * (2 * np.arange(n) + 1) / (2 * n)))
        return result


class RhythmFeatureExtractor:
    def onset_strength(self, signal: np.ndarray, sr: int,
                        frame_size: int = 1024) -> np.ndarray:
        window = _hann_window(frame_size)
        hop = frame_size // 2
        frames = _stft_frame(signal, frame_size, hop, window)
        if frames.shape[0] < 2:
            return np.zeros(max(1, frames.shape[0]))
        diff = np.diff(frames, axis=0)
        diff = np.maximum(diff, 0)
        return np.sum(diff, axis=1)

    def tempo_estimate(self, onset_env: np.ndarray, sr: int,
                        hop_length: int = 512) -> float:
        if len(onset_env) < 4:
            return 0.0
        env = onset_env - np.mean(onset_env)
        corr = np.correlate(env, env, mode="full")
        corr = corr[len(corr) // 2:]
        min_lag = max(1, int(sr * 60 / (250 * hop_length)))
        max_lag = min(len(corr) - 1, int(sr * 60 / (30 * hop_length)))
        if min_lag >= max_lag or max_lag >= len(corr):
            return 0.0
        segment = corr[min_lag:max_lag + 1]
        peak = np.argmax(segment) + min_lag
        if peak == 0:
            return 0.0
        bpm = 60.0 * sr / (peak * hop_length)
        return float(bpm)

    def beat_histogram(self, onset_env: np.ndarray, sr: int,
                        n_bins: int = 64, hop_length: int = 512) -> np.ndarray:
        if len(onset_env) < 4:
            return np.zeros(n_bins)
        env = onset_env - np.mean(onset_env)
        corr = np.correlate(env, env, mode="full")
        corr = corr[len(corr) // 2:]
        bpm_range = np.linspace(30, 250, n_bins)
        hist = np.zeros(n_bins)
        for i, bpm in enumerate(bpm_range):
            lag = int(sr * 60 / (bpm * hop_length))
            if 0 < lag < len(corr):
                hist[i] = max(0, corr[lag])
        total = np.sum(hist)
        if total > 0:
            hist /= total
        return hist

    def pulse_clarity(self, onset_env: np.ndarray) -> float:
        if len(onset_env) < 2:
            return 0.0
        env = onset_env - np.mean(onset_env)
        corr = np.correlate(env, env, mode="full")
        corr = corr[len(corr) // 2:]
        if corr[0] < 1e-12:
            return 0.0
        peaks = []
        for i in range(1, len(corr) - 1):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1]:
                peaks.append(corr[i])
        if not peaks:
            return 0.0
        return float(max(peaks) / corr[0])

    def rhythm_complexity(self, beat_positions: Sequence[float]) -> float:
        if len(beat_positions) < 3:
            return 0.0
        iois = [beat_positions[i+1] - beat_positions[i] for i in range(len(beat_positions) - 1)]
        n = len(iois)
        npvi = 0.0
        for i in range(n - 1):
            a, b = iois[i], iois[i + 1]
            if a + b > 0:
                npvi += abs(a - b) / ((a + b) / 2)
        return float(npvi / (n - 1)) if n > 1 else 0.0

    def syncopation_index(self, onsets: np.ndarray, meter_grid: np.ndarray) -> float:
        if len(onsets) == 0 or len(meter_grid) == 0:
            return 0.0
        on_grid = 0
        off_grid = 0
        for o in onsets:
            if o > 0:
                dists = np.abs(np.arange(len(meter_grid)) - np.argmin(np.abs(meter_grid - o)))
                if np.min(dists) < 2:
                    on_grid += 1
                else:
                    off_grid += 1
        total = on_grid + off_grid
        return float(off_grid / total) if total > 0 else 0.0

    def groove_similarity(self, pattern_a: np.ndarray, pattern_b: np.ndarray) -> float:
        if len(pattern_a) == 0 or len(pattern_b) == 0:
            return 0.0
        min_len = min(len(pattern_a), len(pattern_b))
        a = pattern_a[:min_len].astype(float)
        b = pattern_b[:min_len].astype(float)
        dot = np.sum(a * b)
        mag_a = np.sqrt(np.sum(a ** 2))
        mag_b = np.sqrt(np.sum(b ** 2))
        if mag_a < 1e-12 or mag_b < 1e-12:
            return 0.0
        return float(dot / (mag_a * mag_b))


class HarmonicFeatureExtractor:
    _NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    _MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    _MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    def chroma_vector(self, magnitude: np.ndarray, sr: int, n_chroma: int = 12) -> np.ndarray:
        chroma = np.zeros(n_chroma)
        n_fft = len(magnitude)
        for i in range(1, n_fft):
            freq = i * sr / (2 * n_fft)
            if freq < 20 or freq > 5000:
                continue
            midi = 69 + 12 * math.log2(freq / 440.0)
            pitch_class = int(round(midi)) % n_chroma
            chroma[pitch_class] += magnitude[i] ** 2
        total = np.sum(chroma)
        if total > 0:
            chroma /= total
        return chroma

    def tonnetz_features(self, chroma: np.ndarray) -> np.ndarray:
        if len(chroma) != 12:
            return np.zeros(6)
        phi = np.zeros(6)
        for i in range(12):
            angle_fifth = 2 * np.pi * i * 7 / 12
            angle_minor = 2 * np.pi * i * 3 / 12
            angle_major = 2 * np.pi * i * 4 / 12
            phi[0] += chroma[i] * np.sin(angle_fifth)
            phi[1] += chroma[i] * np.cos(angle_fifth)
            phi[2] += chroma[i] * np.sin(angle_minor)
            phi[3] += chroma[i] * np.cos(angle_minor)
            phi[4] += chroma[i] * np.sin(angle_major)
            phi[5] += chroma[i] * np.cos(angle_major)
        return phi

    def harmonic_change_detection(self, chroma_seq: np.ndarray) -> np.ndarray:
        if chroma_seq.ndim != 2 or chroma_seq.shape[0] < 2:
            return np.zeros(max(1, chroma_seq.shape[0] - 1) if chroma_seq.ndim == 2 else 0)
        changes = np.zeros(chroma_seq.shape[0] - 1)
        for i in range(len(changes)):
            a = chroma_seq[i]
            b = chroma_seq[i + 1]
            dot = np.sum(a * b)
            na = np.sqrt(np.sum(a ** 2))
            nb = np.sqrt(np.sum(b ** 2))
            if na > 1e-12 and nb > 1e-12:
                cos_sim = dot / (na * nb)
                changes[i] = 1.0 - max(-1, min(1, cos_sim))
            else:
                changes[i] = 1.0
        return changes

    def key_estimation(self, chroma: np.ndarray) -> tuple[str, float]:
        if len(chroma) != 12:
            return ("C", 0.0)
        best_key = "C"
        best_corr = -1.0
        best_mode = "major"
        for shift in range(12):
            rotated = np.roll(chroma, -shift)
            for mode, profile in [("major", self._MAJOR_PROFILE), ("minor", self._MINOR_PROFILE)]:
                corr = float(np.corrcoef(rotated, profile)[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_key = self._NOTE_NAMES[shift]
                    best_mode = mode
        return (f"{best_key} {best_mode}", max(0.0, best_corr))

    def tonal_power_ratio(self, magnitude: np.ndarray, f0: float, sr: int) -> float:
        if f0 <= 0 or len(magnitude) == 0:
            return 0.0
        n_fft = len(magnitude)
        total_power = np.sum(magnitude ** 2)
        if total_power < 1e-12:
            return 0.0
        harmonic_power = 0.0
        for h in range(1, 20):
            freq = f0 * h
            bin_idx = int(freq * 2 * n_fft / sr)
            if bin_idx >= n_fft:
                break
            lo = max(0, bin_idx - 2)
            hi = min(n_fft, bin_idx + 3)
            harmonic_power += np.sum(magnitude[lo:hi] ** 2)
        return float(harmonic_power / total_power)


class TimbreFeatureExtractor:
    def zero_crossing_rate(self, signal: np.ndarray) -> float:
        if len(signal) < 2:
            return 0.0
        crossings = np.sum(np.abs(np.diff(np.sign(signal))) > 0)
        return float(crossings / (len(signal) - 1))

    def rms_energy(self, signal: np.ndarray) -> float:
        if len(signal) == 0:
            return 0.0
        return float(np.sqrt(np.mean(signal ** 2)))

    def attack_time(self, envelope: np.ndarray) -> float:
        if len(envelope) == 0:
            return 0.0
        peak_idx = int(np.argmax(envelope))
        threshold = 0.1 * envelope[peak_idx]
        start_idx = 0
        for i in range(peak_idx):
            if envelope[i] >= threshold:
                start_idx = i
                break
        return float(peak_idx - start_idx)

    def temporal_centroid(self, envelope: np.ndarray) -> float:
        total = np.sum(envelope)
        if total < 1e-12:
            return 0.0
        t = np.arange(len(envelope), dtype=float)
        return float(np.sum(t * envelope) / total)

    def log_attack_time(self, envelope: np.ndarray) -> float:
        at = self.attack_time(envelope)
        return float(np.log1p(at))

    def spectral_irregularity(self, harmonics: np.ndarray) -> float:
        if len(harmonics) < 2:
            return 0.0
        diffs = np.diff(harmonics)
        return float(np.sum(diffs ** 2) / np.sum(harmonics ** 2)) if np.sum(harmonics ** 2) > 0 else 0.0

    def tristimulus(self, harmonics: np.ndarray) -> tuple[float, float, float]:
        if len(harmonics) == 0:
            return (0.0, 0.0, 0.0)
        total = np.sum(harmonics)
        if total < 1e-12:
            return (0.0, 0.0, 0.0)
        t1 = float(harmonics[0] / total)
        t2 = float(np.sum(harmonics[1:4]) / total) if len(harmonics) > 1 else 0.0
        t3 = float(np.sum(harmonics[4:]) / total) if len(harmonics) > 4 else 0.0
        return (t1, t2, t3)

    def brightness(self, magnitude: np.ndarray, freqs: np.ndarray,
                    cutoff: float = 1500.0) -> float:
        total = np.sum(magnitude)
        if total < 1e-12:
            return 0.0
        above = np.sum(magnitude[freqs >= cutoff])
        return float(above / total)


class UnifiedDeepExtractor:
    def __init__(self) -> None:
        self.spectral = SpectralFeatureExtractor()
        self.rhythm = RhythmFeatureExtractor()
        self.harmonic = HarmonicFeatureExtractor()
        self.timbre = TimbreFeatureExtractor()

    def feature_names(self) -> list[str]:
        return [
            "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
            "spectral_flatness", "spectral_crest", "spectral_slope",
            "spectral_decrease", "zcr", "rms_energy", "tempo",
            "pulse_clarity", "key", "key_confidence",
            "tpr", "brightness", "log_attack_time",
            "tristimulus_1", "tristimulus_2", "tristimulus_3",
        ] + [f"mfcc_{i}" for i in range(13)] + [f"chroma_{i}" for i in range(12)]

    def extract_all(self, signal: np.ndarray, sr: int,
                     frame_size: int = 1024) -> dict[str, float]:
        window = _hann_window(frame_size)
        n_fft = frame_size // 2 + 1
        chunk = signal[:frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        spec = np.abs(np.fft.rfft(chunk * window))
        freqs = np.linspace(0, sr / 2, n_fft)
        centroid = self.spectral.spectral_centroid(spec, freqs)
        bw = self.spectral.spectral_bandwidth(spec, freqs, centroid)
        rolloff = self.spectral.spectral_rolloff(spec)
        flatness = self.spectral.spectral_flatness(spec)
        crest = self.spectral.spectral_crest(spec)
        slope = self.spectral.spectral_slope(spec, freqs)
        decrease = self.spectral.spectral_decrease(spec)

        zcr = self.timbre.zero_crossing_rate(signal)
        rms = self.timbre.rms_energy(signal)

        onset_env = self.rhythm.onset_strength(signal, sr, frame_size)
        tempo = self.rhythm.tempo_estimate(onset_env, sr, frame_size // 2)
        clarity = self.rhythm.pulse_clarity(onset_env)

        chroma = self.harmonic.chroma_vector(spec, sr)
        key_name, key_conf = self.harmonic.key_estimation(chroma)

        envelope = np.abs(signal)
        lat = self.timbre.log_attack_time(envelope)
        t1, t2, t3 = self.timbre.tristimulus(spec[:20])

        f0_est = centroid if centroid > 0 else 440.0
        tpr = self.harmonic.tonal_power_ratio(spec, f0_est, sr)
        bright = self.timbre.brightness(spec, freqs)

        mfccs = self.spectral.mfcc(signal, sr, n_mfcc=13, frame_size=frame_size)
        mean_mfcc = np.mean(mfccs, axis=0) if mfccs.shape[0] > 0 else np.zeros(13)

        features: dict[str, float] = {
            "spectral_centroid": centroid,
            "spectral_bandwidth": bw,
            "spectral_rolloff": rolloff,
            "spectral_flatness": flatness,
            "spectral_crest": crest,
            "spectral_slope": slope,
            "spectral_decrease": decrease,
            "zcr": zcr,
            "rms_energy": rms,
            "tempo": tempo,
            "pulse_clarity": clarity,
            "key": 0.0,
            "key_confidence": key_conf,
            "tpr": tpr,
            "brightness": bright,
            "log_attack_time": lat,
            "tristimulus_1": t1,
            "tristimulus_2": t2,
            "tristimulus_3": t3,
        }
        for i in range(13):
            features[f"mfcc_{i}"] = float(mean_mfcc[i])
        for i in range(12):
            features[f"chroma_{i}"] = float(chroma[i])
        return features

    def feature_matrix(self, signals: list[np.ndarray], sr: int) -> np.ndarray:
        names = self.feature_names()
        matrix = np.zeros((len(signals), len(names)))
        for i, sig in enumerate(signals):
            feats = self.extract_all(sig, sr)
            for j, name in enumerate(names):
                matrix[i, j] = feats.get(name, 0.0)
        return matrix

    def normalize_features(self, features: dict[str, float],
                            method: str = "z-score",
                            stats: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
        if stats is None:
            return features
        result: dict[str, float] = {}
        for k, v in features.items():
            if k in stats:
                mean, std = stats[k]
                if method == "z-score" and std > 1e-12:
                    result[k] = (v - mean) / std
                elif method == "min-max" and std > mean:
                    result[k] = (v - mean) / (std - mean) if std > mean else 0.0
                else:
                    result[k] = v
            else:
                result[k] = v
        return result
