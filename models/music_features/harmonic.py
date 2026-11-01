from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class HarmonicFeatures:
    key: str = "C"
    key_confidence: float = 0.0
    mode: str = "major"
    mode_confidence: float = 0.0
    chroma_energy: list[float] = field(default_factory=lambda: [0.0] * 12)
    harmonic_change_rate: float = 0.0
    tonal_centroid: list[float] = field(default_factory=lambda: [0.0] * 6)
    dissonance_mean: float = 0.0
    dissonance_std: float = 0.0
    pitch_class_histogram: list[float] = field(default_factory=lambda: [0.0] * 12)
    key_clarity: float = 0.0


class HarmonicExtractor:
    def __init__(self, sr: int = 22050, n_fft: int = 4096, hop_length: int = 512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> HarmonicFeatures:
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        audio = audio.astype(np.float64)

        chroma_seq = self._compute_chromagram(audio)
        chroma_mean = np.mean(chroma_seq, axis=1)
        chroma_total = chroma_mean.sum()
        if chroma_total > 0:
            chroma_norm = chroma_mean / chroma_total
        else:
            chroma_norm = np.ones(12) / 12.0

        key, key_conf, mode, mode_conf = self._estimate_key(chroma_norm)
        key_clarity = self._key_clarity(chroma_norm)
        tonnetz = self._tonnetz_features(chroma_seq)
        hcr = self._harmonic_change_detection(chroma_seq)
        dissonance_mean, dissonance_std = self._dissonance(audio)

        return HarmonicFeatures(
            key=key,
            key_confidence=key_conf,
            mode=mode,
            mode_confidence=mode_conf,
            chroma_energy=[float(x) for x in chroma_mean],
            harmonic_change_rate=hcr,
            tonal_centroid=[float(x) for x in tonnetz],
            dissonance_mean=dissonance_mean,
            dissonance_std=dissonance_std,
            pitch_class_histogram=[float(x) for x in chroma_norm],
            key_clarity=key_clarity,
        )

    def _compute_chromagram(self, audio: np.ndarray) -> np.ndarray:
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        if n_frames < 1:
            n_frames = 1
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        window = np.hanning(self.n_fft)
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)
        chroma = np.zeros((12, n_frames))

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            spec = np.abs(np.fft.rfft(frame * window))
            for j, f in enumerate(freqs):
                if f < 20.0:
                    continue
                pc = (int(round(12.0 * np.log2(f / 440.0))) + 9) % 12
                chroma[pc, i] += spec[j]

        col_sum = chroma.sum(axis=0, keepdims=True)
        col_sum = np.where(col_sum == 0, 1.0, col_sum)
        chroma = chroma / col_sum
        return chroma

    def _estimate_key(self, chroma: np.ndarray) -> tuple[str, float, str, float]:
        best_corr = -2.0
        best_key = 0
        best_mode = "major"

        for shift in range(12):
            rotated = np.roll(chroma, -shift)
            corr_major = float(np.corrcoef(rotated, KRUMHANSL_MAJOR)[0, 1])
            corr_minor = float(np.corrcoef(rotated, KRUMHANSL_MINOR)[0, 1])

            if corr_major > best_corr:
                best_corr = corr_major
                best_key = shift
                best_mode = "major"
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = shift
                best_mode = "minor"

        profile = KRUMHANSL_MAJOR if best_mode == "major" else KRUMHANSL_MINOR
        rotated = np.roll(chroma, -best_key)
        mode_conf = float(np.corrcoef(rotated, profile)[0, 1])
        other_profile = KRUMHANSL_MINOR if best_mode == "major" else KRUMHANSL_MAJOR
        other_conf = float(np.corrcoef(rotated, other_profile)[0, 1])

        return (
            PITCH_CLASSES[best_key],
            float(np.clip(best_corr, 0.0, 1.0)),
            best_mode,
            float(np.clip(mode_conf - other_conf, 0.0, 1.0)),
        )

    def _tonnetz_features(self, chroma: np.ndarray) -> np.ndarray:
        chroma_norm = chroma.copy()
        col_sum = chroma_norm.sum(axis=0, keepdims=True)
        col_sum = np.where(col_sum == 0, 1.0, col_sum)
        chroma_norm = chroma_norm / col_sum

        r = np.arange(12)
        phi = np.zeros((6, 12))
        phi[0] = np.sin(2 * np.pi * r * 7 / 12)  # perfect fifth sin
        phi[1] = np.cos(2 * np.pi * r * 7 / 12)  # perfect fifth cos
        phi[2] = np.sin(2 * np.pi * r * 4 / 12)  # major third sin
        phi[3] = np.cos(2 * np.pi * r * 4 / 12)  # major third cos
        phi[4] = np.sin(2 * np.pi * r * 3 / 12)  # minor third sin
        phi[5] = np.cos(2 * np.pi * r * 3 / 12)  # minor third cos

        tonnetz = phi @ chroma_norm
        return np.mean(tonnetz, axis=1)

    def _harmonic_change_detection(self, chroma_seq: np.ndarray) -> float:
        if chroma_seq.shape[1] < 2:
            return 0.0
        diffs = np.diff(chroma_seq, axis=1)
        changes = np.sqrt((diffs ** 2).sum(axis=0))
        return float(np.mean(changes))

    def _dissonance(self, audio: np.ndarray) -> tuple[float, float]:
        n_fft = min(self.n_fft, len(audio))
        n_frames = max(1, (len(audio) - n_fft) // self.hop_length)
        window = np.hanning(n_fft)
        dissonances = []

        consonant_ratios = [1.0, 2.0, 1.5, 4 / 3, 5 / 4, 5 / 3, 6 / 5, 8 / 5]

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spec = np.abs(np.fft.rfft(frame * window))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

            top_k = min(8, len(spec))
            peak_idx = np.argsort(spec)[-top_k:]
            peak_freqs = freqs[peak_idx]
            peak_freqs = peak_freqs[peak_freqs > 20.0]

            if len(peak_freqs) < 2:
                dissonances.append(0.0)
                continue

            diss = 0.0
            n_pairs = 0
            for a in range(len(peak_freqs)):
                for b in range(a + 1, len(peak_freqs)):
                    ratio = max(peak_freqs[a], peak_freqs[b]) / min(peak_freqs[a], peak_freqs[b])
                    min_dist = min(abs(ratio - cr) for cr in consonant_ratios)
                    diss += min_dist
                    n_pairs += 1
            if n_pairs > 0:
                dissonances.append(diss / n_pairs)

        if not dissonances:
            return 0.0, 0.0
        return float(np.mean(dissonances)), float(np.std(dissonances))

    def _key_clarity(self, chroma: np.ndarray) -> float:
        correlations = []
        for shift in range(12):
            rotated = np.roll(chroma, -shift)
            correlations.append(float(np.corrcoef(rotated, KRUMHANSL_MAJOR)[0, 1]))
            correlations.append(float(np.corrcoef(rotated, KRUMHANSL_MINOR)[0, 1]))
        sorted_corr = sorted(correlations, reverse=True)
        if len(sorted_corr) >= 2:
            return float(np.clip(sorted_corr[0] - sorted_corr[1], 0.0, 1.0))
        return 0.0
