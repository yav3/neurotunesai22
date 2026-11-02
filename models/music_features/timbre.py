from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimbreFeatures:
    brightness: float = 0.0
    warmth: float = 0.0
    roughness: float = 0.0
    sharpness: float = 0.0
    boominess: float = 0.0
    depth: float = 0.0
    hardness: float = 0.0
    spectral_centroid_var: float = 0.0
    mfcc_delta_mean: list[float] = field(default_factory=lambda: [0.0] * 13)
    mfcc_delta2_mean: list[float] = field(default_factory=lambda: [0.0] * 13)
    spectral_contrast_mean: list[float] = field(default_factory=lambda: [0.0] * 7)
    tonnetz_mean: list[float] = field(default_factory=lambda: [0.0] * 6)


class TimbreExtractor:
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray, sr: int = 22050) -> TimbreFeatures:
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        audio = audio.astype(np.float64)

        mag, freqs = self._magnitude_spectrum_frames(audio, sr)

        brightness = self._brightness(mag, freqs, sr)
        warmth = self._warmth(mag, freqs)
        roughness = self._roughness(mag, freqs)
        sharpness = self._sharpness(mag, freqs, sr)
        boominess = self._boominess(mag, freqs)
        depth = self._depth(mag, freqs)
        hardness = self._hardness(mag, freqs, sr)
        centroid_var = self._spectral_centroid_variance(mag, freqs)
        mfcc_delta, mfcc_delta2 = self._mfcc_deltas(audio, sr)
        contrast = self._spectral_contrast_mean(mag, freqs)
        tonnetz = self._tonnetz_mean(mag, freqs)

        return TimbreFeatures(
            brightness=brightness,
            warmth=warmth,
            roughness=roughness,
            sharpness=sharpness,
            boominess=boominess,
            depth=depth,
            hardness=hardness,
            spectral_centroid_var=centroid_var,
            mfcc_delta_mean=mfcc_delta,
            mfcc_delta2_mean=mfcc_delta2,
            spectral_contrast_mean=contrast,
            tonnetz_mean=tonnetz,
        )

    def _magnitude_spectrum_frames(
        self, audio: np.ndarray, sr: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_frames = max(1, 1 + (len(audio) - self.n_fft) // self.hop_length)
        window = np.hanning(self.n_fft)
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / sr)
        n_bins = len(freqs)
        mag = np.zeros((n_bins, n_frames))
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            mag[:, i] = np.abs(np.fft.rfft(frame * window))
        return mag, freqs

    def _brightness(self, mag: np.ndarray, freqs: np.ndarray, sr: int) -> float:
        cutoff = 1500.0
        total_energy = (mag ** 2).sum()
        if total_energy <= 0:
            return 0.0
        high_mask = freqs >= cutoff
        high_energy = (mag[high_mask] ** 2).sum()
        return float(high_energy / total_energy)

    def _warmth(self, mag: np.ndarray, freqs: np.ndarray) -> float:
        bass_mask = freqs < 300.0
        mid_mask = (freqs >= 300.0) & (freqs < 2000.0)
        bass_energy = (mag[bass_mask] ** 2).sum()
        mid_energy = (mag[mid_mask] ** 2).sum()
        if mid_energy <= 0:
            return 0.0
        ratio = bass_energy / mid_energy
        return float(np.clip(ratio / 3.0, 0.0, 1.0))

    def _roughness(self, mag: np.ndarray, freqs: np.ndarray) -> float:
        if mag.shape[1] < 3:
            return 0.0
        am_band = (freqs >= 15.0) & (freqs <= 300.0)
        if not am_band.any():
            return 0.0
        band_mag = mag[am_band, :]
        am_depth = np.std(band_mag, axis=1)
        am_mean = np.mean(band_mag, axis=1)
        am_mean = np.where(am_mean == 0, 1e-10, am_mean)
        modulation = am_depth / am_mean
        return float(np.clip(np.mean(modulation), 0.0, 1.0))

    def _sharpness(self, mag: np.ndarray, freqs: np.ndarray, sr: int) -> float:
        n_bark = 24
        bark_edges = 600.0 * np.sinh(np.arange(n_bark + 1) / 6.0)
        bark_edges = np.clip(bark_edges, 0, sr / 2.0)

        z_weights = np.arange(1, n_bark + 1, dtype=float)
        z_weights[z_weights <= 15] = 1.0
        z_weights[z_weights > 15] = 0.066 * np.exp(0.171 * z_weights[z_weights > 15])

        loudness = np.zeros(n_bark)
        for b in range(n_bark):
            mask = (freqs >= bark_edges[b]) & (freqs < bark_edges[b + 1])
            if mask.any():
                loudness[b] = np.mean(mag[mask] ** 2)

        total_loudness = loudness.sum()
        if total_loudness <= 0:
            return 0.0
        numerator = (z_weights * loudness).sum()
        sharpness = 0.11 * numerator / total_loudness
        return float(np.clip(sharpness, 0.0, 1.0))

    def _boominess(self, mag: np.ndarray, freqs: np.ndarray) -> float:
        sub_mask = freqs < 100.0
        total = (mag ** 2).sum()
        if total <= 0:
            return 0.0
        sub_energy = (mag[sub_mask] ** 2).sum()
        return float(np.clip(sub_energy / total * 5.0, 0.0, 1.0))

    def _depth(self, mag: np.ndarray, freqs: np.ndarray) -> float:
        low_mask = freqs < 500.0
        total = (mag ** 2).sum()
        if total <= 0:
            return 0.0
        low_energy = (mag[low_mask] ** 2).sum()
        return float(np.clip(low_energy / total, 0.0, 1.0))

    def _hardness(self, mag: np.ndarray, freqs: np.ndarray, sr: int) -> float:
        high_mask = freqs >= 4000.0
        total = (mag ** 2).sum()
        if total <= 0:
            return 0.0
        high_energy = (mag[high_mask] ** 2).sum()
        return float(np.clip(high_energy / total * 3.0, 0.0, 1.0))

    def _spectral_centroid_variance(self, mag: np.ndarray, freqs: np.ndarray) -> float:
        mag_sum = mag.sum(axis=0)
        mag_sum = np.where(mag_sum == 0, 1.0, mag_sum)
        centroids = (freqs[:, None] * mag).sum(axis=0) / mag_sum
        return float(np.var(centroids))

    def _mfcc_deltas(self, audio: np.ndarray, sr: int) -> tuple[list[float], list[float]]:
        from models.music_features.spectral import SpectralExtractor
        ext = SpectralExtractor(sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        S = ext._stft(audio)
        power = np.abs(S) ** 2
        mel_fb = ext._mel_filterbank(40)
        mel_spec = mel_fb @ power
        mel_spec = np.where(mel_spec == 0, 1e-10, mel_spec)
        log_mel = np.log(mel_spec)

        n_mfcc = 13
        n_mels = 40
        mfcc_frames = np.zeros((n_mfcc, log_mel.shape[1]))
        for i in range(n_mfcc):
            for j in range(n_mels):
                mfcc_frames[i] += log_mel[j] * np.cos(np.pi * i * (j + 0.5) / n_mels)
        mfcc_frames *= np.sqrt(2.0 / n_mels)

        if mfcc_frames.shape[1] < 3:
            return [0.0] * 13, [0.0] * 13

        delta = np.zeros_like(mfcc_frames)
        for t in range(1, mfcc_frames.shape[1] - 1):
            delta[:, t] = (mfcc_frames[:, t + 1] - mfcc_frames[:, t - 1]) / 2.0

        delta2 = np.zeros_like(delta)
        for t in range(1, delta.shape[1] - 1):
            delta2[:, t] = (delta[:, t + 1] - delta[:, t - 1]) / 2.0

        return (
            [float(x) for x in np.mean(np.abs(delta), axis=1)],
            [float(x) for x in np.mean(np.abs(delta2), axis=1)],
        )

    def _spectral_contrast_mean(self, mag: np.ndarray, freqs: np.ndarray) -> list[float]:
        n_bands = 6
        fmin = 200.0
        freq_bands = [fmin * (2.0 ** i) for i in range(n_bands + 1)]
        contrasts = []
        for b in range(n_bands + 1):
            if b < n_bands:
                low, high = freq_bands[b], freq_bands[b + 1]
            else:
                low, high = freq_bands[-1], freqs[-1]
            mask = (freqs >= low) & (freqs < high)
            if not mask.any():
                contrasts.append(0.0)
                continue
            band = mag[mask, :]
            srt = np.sort(band, axis=0)
            alpha = max(1, int(0.2 * len(srt)))
            peak = srt[-alpha:, :].mean()
            valley = srt[:alpha, :].mean()
            contrasts.append(float(np.log1p(peak) - np.log1p(valley)))
        return contrasts

    def _tonnetz_mean(self, mag: np.ndarray, freqs: np.ndarray) -> list[float]:
        chroma = np.zeros((12, mag.shape[1]))
        for i, f in enumerate(freqs):
            if f < 20.0:
                continue
            pc = (int(round(12.0 * np.log2(f / 440.0))) + 9) % 12
            chroma[pc] += mag[i]
        col_sum = chroma.sum(axis=0, keepdims=True)
        col_sum = np.where(col_sum == 0, 1.0, col_sum)
        chroma = chroma / col_sum

        r = np.arange(12)
        phi = np.zeros((6, 12))
        phi[0] = np.sin(2 * np.pi * r * 7 / 12)
        phi[1] = np.cos(2 * np.pi * r * 7 / 12)
        phi[2] = np.sin(2 * np.pi * r * 4 / 12)
        phi[3] = np.cos(2 * np.pi * r * 4 / 12)
        phi[4] = np.sin(2 * np.pi * r * 3 / 12)
        phi[5] = np.cos(2 * np.pi * r * 3 / 12)

        tonnetz = phi @ chroma
        return [float(x) for x in np.mean(tonnetz, axis=1)]
