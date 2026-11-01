from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


@dataclass
class SpectralFeatures:
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_flatness: float = 0.0
    spectral_contrast: list[float] = field(default_factory=lambda: [0.0] * 7)
    spectral_entropy: float = 0.0
    spectral_flux: float = 0.0
    zero_crossing_rate: float = 0.0
    mfcc: list[float] = field(default_factory=lambda: [0.0] * 13)
    chroma: list[float] = field(default_factory=lambda: [0.0] * 12)


class SpectralExtractor:
    def __init__(self, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> SpectralFeatures:
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        audio = audio.astype(np.float64)

        if _HAS_LIBROSA:
            return self._extract_librosa(audio)
        return self._extract_numpy(audio)

    def _extract_numpy(self, audio: np.ndarray) -> SpectralFeatures:
        S = self._stft(audio)
        mag = np.abs(S)
        power = mag ** 2
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)

        centroid = self._spectral_centroid(mag, freqs)
        rolloff = self._spectral_rolloff(mag, freqs)
        bandwidth = self._spectral_bandwidth(mag, freqs, centroid)
        flatness = self._spectral_flatness(mag)
        contrast = self._spectral_contrast(mag, freqs)
        entropy = self._spectral_entropy(mag)
        flux = self._spectral_flux(mag)
        zcr = self._zero_crossing_rate(audio)
        mfcc = self._mfcc(power)
        chroma = self._chroma_from_stft(mag)

        return SpectralFeatures(
            spectral_centroid=centroid,
            spectral_rolloff=rolloff,
            spectral_bandwidth=bandwidth,
            spectral_flatness=flatness,
            spectral_contrast=contrast,
            spectral_entropy=entropy,
            spectral_flux=flux,
            zero_crossing_rate=zcr,
            mfcc=mfcc,
            chroma=chroma,
        )

    def _extract_librosa(self, audio: np.ndarray) -> SpectralFeatures:
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        power = S ** 2

        centroid = float(np.mean(librosa.feature.spectral_centroid(
            S=S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(
            S=S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(
            S=S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
        contrast_mat = librosa.feature.spectral_contrast(
            S=S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_bands=6)
        contrast = [float(x) for x in np.mean(contrast_mat, axis=1)]
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio, frame_length=self.n_fft, hop_length=self.hop_length)))
        mfcc_mat = librosa.feature.mfcc(S=librosa.power_to_db(power), sr=self.sr, n_mfcc=13)
        mfcc_vals = [float(x) for x in np.mean(mfcc_mat, axis=1)]
        chroma_mat = librosa.feature.chroma_stft(S=S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        chroma_vals = [float(x) for x in np.mean(chroma_mat, axis=1)]

        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)
        mag_np = S
        entropy = self._spectral_entropy(mag_np)
        flux = self._spectral_flux(mag_np)

        return SpectralFeatures(
            spectral_centroid=centroid,
            spectral_rolloff=rolloff,
            spectral_bandwidth=bandwidth,
            spectral_flatness=flatness,
            spectral_contrast=contrast,
            spectral_entropy=entropy,
            spectral_flux=flux,
            zero_crossing_rate=zcr,
            mfcc=mfcc_vals,
            chroma=chroma_vals,
        )

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        if n_frames < 1:
            padded = np.pad(audio, (0, self.n_fft - len(audio)))
            window = np.hanning(self.n_fft)
            return np.fft.rfft(padded * window).reshape(-1, 1)

        window = np.hanning(self.n_fft)
        frames = np.stack([
            audio[i * self.hop_length: i * self.hop_length + self.n_fft]
            for i in range(n_frames)
        ])
        return np.fft.rfft(frames * window, axis=1).T

    def _spectral_centroid(self, mag: np.ndarray, freqs: np.ndarray) -> float:
        mag_sum = mag.sum(axis=0)
        mag_sum = np.where(mag_sum == 0, 1.0, mag_sum)
        centroids = (freqs[:, None] * mag).sum(axis=0) / mag_sum
        return float(np.mean(centroids))

    def _spectral_rolloff(self, mag: np.ndarray, freqs: np.ndarray, threshold: float = 0.85) -> float:
        cumsum = np.cumsum(mag, axis=0)
        total = mag.sum(axis=0, keepdims=True)
        total = np.where(total == 0, 1.0, total)
        norm_cumsum = cumsum / total
        rolloffs = []
        for frame_idx in range(mag.shape[1]):
            idx = np.searchsorted(norm_cumsum[:, frame_idx], threshold)
            idx = min(idx, len(freqs) - 1)
            rolloffs.append(freqs[idx])
        return float(np.mean(rolloffs))

    def _spectral_bandwidth(self, mag: np.ndarray, freqs: np.ndarray, centroid: float) -> float:
        mag_sum = mag.sum(axis=0)
        mag_sum = np.where(mag_sum == 0, 1.0, mag_sum)
        deviation = freqs[:, None] - centroid
        bw = np.sqrt((mag * deviation ** 2).sum(axis=0) / mag_sum)
        return float(np.mean(bw))

    def _spectral_flatness(self, mag: np.ndarray) -> float:
        power = mag ** 2
        mean_per_frame = power.mean(axis=0)
        mean_per_frame = np.where(mean_per_frame == 0, 1e-10, mean_per_frame)
        log_mean = np.mean(np.log(power + 1e-10), axis=0)
        geo_mean = np.exp(log_mean)
        flatness = geo_mean / mean_per_frame
        return float(np.mean(np.clip(flatness, 0.0, 1.0)))

    def _spectral_contrast(self, mag: np.ndarray, freqs: np.ndarray) -> list[float]:
        n_bands = 6
        fmin = 200.0
        freq_bands = [fmin * (2.0 ** i) for i in range(n_bands + 1)]
        freq_bands[-1] = min(freq_bands[-1], self.sr / 2.0)
        contrasts = []
        for b in range(n_bands + 1):
            if b < n_bands:
                low, high = freq_bands[b], freq_bands[b + 1]
            else:
                low, high = freq_bands[-1], self.sr / 2.0
            mask = (freqs >= low) & (freqs < high)
            if not mask.any():
                contrasts.append(0.0)
                continue
            band_mag = mag[mask, :]
            sorted_mag = np.sort(band_mag, axis=0)
            alpha = max(1, int(0.2 * len(sorted_mag)))
            peak = sorted_mag[-alpha:, :].mean()
            valley = sorted_mag[:alpha, :].mean()
            contrast = float(np.log1p(peak) - np.log1p(valley))
            contrasts.append(contrast)
        return contrasts

    def _spectral_entropy(self, mag: np.ndarray) -> float:
        power = mag ** 2
        total = power.sum(axis=0, keepdims=True)
        total = np.where(total == 0, 1.0, total)
        prob = power / total
        prob = np.where(prob == 0, 1e-10, prob)
        entropy_per_frame = -np.sum(prob * np.log2(prob), axis=0)
        return float(np.mean(entropy_per_frame))

    def _spectral_flux(self, mag: np.ndarray) -> float:
        if mag.shape[1] < 2:
            return 0.0
        diff = np.diff(mag, axis=1)
        flux_per_frame = np.sqrt((diff ** 2).sum(axis=0))
        return float(np.mean(flux_per_frame))

    def _zero_crossing_rate(self, audio: np.ndarray) -> float:
        signs = np.sign(audio)
        crossings = np.abs(np.diff(signs)) > 0
        n_frames = max(1, len(audio) // self.hop_length)
        return float(crossings.sum() / max(len(audio) - 1, 1))

    def _mel_filterbank(self, n_mels: int = 40) -> np.ndarray:
        f_min, f_max = 0.0, self.sr / 2.0
        mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        n_freqs = self.n_fft // 2 + 1
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)
        filterbank = np.zeros((n_mels, n_freqs))

        for m in range(n_mels):
            left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
            for k in range(max(0, left), min(center, n_freqs)):
                if center > left:
                    filterbank[m, k] = (k - left) / (center - left)
            for k in range(max(0, center), min(right, n_freqs)):
                if right > center:
                    filterbank[m, k] = (right - k) / (right - center)
        return filterbank

    def _mfcc(self, power: np.ndarray, n_mfcc: int = 13, n_mels: int = 40) -> list[float]:
        mel_fb = self._mel_filterbank(n_mels)
        mel_spec = mel_fb @ power
        mel_spec = np.where(mel_spec == 0, 1e-10, mel_spec)
        log_mel = np.log(mel_spec)
        mfcc_frames = np.zeros((n_mfcc, log_mel.shape[1]))
        for i in range(n_mfcc):
            for j in range(n_mels):
                mfcc_frames[i] += log_mel[j] * np.cos(np.pi * i * (j + 0.5) / n_mels)
        mfcc_frames *= np.sqrt(2.0 / n_mels)
        return [float(x) for x in np.mean(mfcc_frames, axis=1)]

    def _chroma_from_stft(self, mag: np.ndarray) -> list[float]:
        n_freqs = mag.shape[0]
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)
        chroma = np.zeros((12, mag.shape[1]))
        for i, f in enumerate(freqs):
            if f < 20.0:
                continue
            pitch_class = int(round(12.0 * np.log2(f / 440.0))) % 12
            chroma[pitch_class] += mag[i]
        total = chroma.sum(axis=0, keepdims=True)
        total = np.where(total == 0, 1.0, total)
        chroma = chroma / total
        return [float(x) for x in np.mean(chroma, axis=1)]
