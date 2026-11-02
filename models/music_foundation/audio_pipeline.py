"""Audio data loading, segmentation, augmentation, and synthetic generation.

Provides the data infrastructure needed to train MusicFrontierModel on
raw audio: segmentation into fixed-length windows, simple augmentations,
and synthetic audio generation for testing.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AudioPipelineConfig:
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    segment_duration_s: float = 5.0
    overlap_ratio: float = 0.5
    normalize: bool = True
    augment: bool = False
    seed: int = 42


class AudioSegmenter:
    def __init__(self, segment_samples: int, overlap_ratio: float = 0.5) -> None:
        self.segment_samples = segment_samples
        self.hop = max(1, int(segment_samples * (1.0 - overlap_ratio)))

    def segment(self, audio: np.ndarray) -> list[np.ndarray]:
        n = len(audio)
        if n <= self.segment_samples:
            padded = np.zeros(self.segment_samples)
            padded[:n] = audio
            return [padded]
        segments = []
        start = 0
        while start + self.segment_samples <= n:
            segments.append(audio[start:start + self.segment_samples].copy())
            start += self.hop
        if start < n and start + self.segment_samples > n:
            last = np.zeros(self.segment_samples)
            last[:n - start] = audio[start:]
            segments.append(last)
        return segments


class AudioAugmenter:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        if abs(rate - 1.0) < 1e-6:
            return audio.copy()
        n = len(audio)
        new_n = int(n / rate)
        if new_n < 2:
            return audio[:2].copy()
        indices = np.linspace(0, n - 1, new_n)
        return np.interp(indices, np.arange(n), audio)

    def pitch_shift(self, audio: np.ndarray, n_semitones: float) -> np.ndarray:
        rate = 2.0 ** (n_semitones / 12.0)
        stretched = self.time_stretch(audio, rate)
        n = len(audio)
        if len(stretched) >= n:
            return stretched[:n]
        result = np.zeros(n)
        result[:len(stretched)] = stretched
        return result

    def add_noise(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        signal_power = np.mean(audio ** 2) + 1e-12
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = self.rng.randn(len(audio)) * np.sqrt(noise_power)
        return audio + noise

    def random_gain(self, audio: np.ndarray, min_db: float = -6.0,
                    max_db: float = 6.0) -> np.ndarray:
        gain_db = self.rng.uniform(min_db, max_db)
        return audio * (10.0 ** (gain_db / 20.0))

    def augment(self, audio: np.ndarray) -> np.ndarray:
        result = audio.copy()
        if self.rng.rand() < 0.5:
            rate = self.rng.uniform(0.9, 1.1)
            stretched = self.time_stretch(result, rate)
            n = len(result)
            if len(stretched) >= n:
                result = stretched[:n]
            else:
                result = np.zeros(n)
                result[:len(stretched)] = stretched
        if self.rng.rand() < 0.5:
            result = self.add_noise(result, snr_db=self.rng.uniform(15, 30))
        if self.rng.rand() < 0.5:
            result = self.random_gain(result, -3, 3)
        return result


class AudioPreprocessor:
    def __init__(self, config: AudioPipelineConfig | None = None) -> None:
        self.config = config or AudioPipelineConfig()
        seg_samples = int(self.config.segment_duration_s * self.config.sr)
        self.segmenter = AudioSegmenter(seg_samples, self.config.overlap_ratio)
        self.augmenter = AudioAugmenter(self.config.seed) if self.config.augment else None

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak > 1e-8:
            return audio / peak
        return audio

    def process(self, audio: np.ndarray) -> np.ndarray:
        if self.config.normalize:
            audio = self._normalize(audio)
        segments = self.segmenter.segment(audio)
        if self.augmenter is not None:
            segments = [self.augmenter.augment(s) for s in segments]
        return np.array(segments)

    def process_batch(self, audio_list: list[np.ndarray]) -> np.ndarray:
        all_segments = []
        for audio in audio_list:
            segs = self.process(audio)
            all_segments.append(segs)
        return np.concatenate(all_segments, axis=0)


class SyntheticAudioGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def sine_tone(self, freq: float, duration_s: float, sr: int = 22050) -> np.ndarray:
        t = np.arange(int(duration_s * sr)) / sr
        return np.sin(2.0 * np.pi * freq * t)

    def chord(self, freqs: list[float], duration_s: float,
              sr: int = 22050) -> np.ndarray:
        t = np.arange(int(duration_s * sr)) / sr
        signal = np.zeros_like(t)
        for f in freqs:
            signal += np.sin(2.0 * np.pi * f * t)
        if len(freqs) > 0:
            signal /= len(freqs)
        return signal

    def drum_hit(self, duration_s: float = 0.5, sr: int = 22050) -> np.ndarray:
        n = int(duration_s * sr)
        noise = self.rng.randn(n)
        decay = np.exp(-np.linspace(0, 8, n))
        return noise * decay

    def generate_dataset(self, n_samples: int, duration_s: float = 2.0,
                         sr: int = 22050) -> list[np.ndarray]:
        samples = []
        for _ in range(n_samples):
            choice = self.rng.randint(0, 4)
            if choice == 0:
                freq = self.rng.uniform(100, 2000)
                samples.append(self.sine_tone(freq, duration_s, sr))
            elif choice == 1:
                n_notes = self.rng.randint(2, 5)
                freqs = [self.rng.uniform(100, 2000) for _ in range(n_notes)]
                samples.append(self.chord(freqs, duration_s, sr))
            elif choice == 2:
                samples.append(self.drum_hit(duration_s, sr))
            else:
                samples.append(self.rng.randn(int(duration_s * sr)) * 0.3)
        return samples
