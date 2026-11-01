"""Data-to-sound mapping for accessible music experiences."""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParameterMapping:
    """Map a data dimension to an audio parameter."""

    data_min: float
    data_max: float
    audio_min: float
    audio_max: float
    mapping_type: str = "linear"

    def map_value(self, x: float) -> float:
        if self.data_max == self.data_min:
            return (self.audio_min + self.audio_max) / 2.0
        t = (x - self.data_min) / (self.data_max - self.data_min)
        t = np.clip(t, 0, 1)
        if self.mapping_type == "log":
            t = np.log1p(t * (np.e - 1)) / 1.0
        elif self.mapping_type == "exponential":
            t = (np.exp(t) - 1) / (np.e - 1)
        return float(self.audio_min + t * (self.audio_max - self.audio_min))

    def map_array(self, data: np.ndarray) -> np.ndarray:
        return np.array([self.map_value(x) for x in data.ravel()]).reshape(data.shape)


PENTATONIC = [0, 2, 4, 7, 9]
WHOLE_TONE = [0, 2, 4, 6, 8, 10]
CHROMATIC = list(range(12))


@dataclass
class PitchSonifier:
    """Map scalar time series to pitch contour audio."""

    min_freq: float = 200.0
    max_freq: float = 2000.0
    scale: str = "chromatic"

    def _quantize_to_scale(self, freq: float) -> float:
        if self.scale == "chromatic":
            intervals = CHROMATIC
        elif self.scale == "pentatonic":
            intervals = PENTATONIC
        elif self.scale == "whole_tone":
            intervals = WHOLE_TONE
        else:
            return freq
        if freq <= 0:
            return self.min_freq
        midi = 69 + 12 * np.log2(freq / 440.0)
        octave = int(midi) // 12
        pc = round(midi) % 12
        best = min(intervals, key=lambda x: abs(x - pc))
        quantized_midi = octave * 12 + best
        return 440.0 * 2 ** ((quantized_midi - 69) / 12.0)

    def sonify(
        self, data: np.ndarray, sr: int = 16000, note_duration: float = 0.1
    ) -> np.ndarray:
        mapping = ParameterMapping(
            float(np.min(data)), float(np.max(data)),
            self.min_freq, self.max_freq,
        )
        n_samples_per_note = int(sr * note_duration)
        total = n_samples_per_note * len(data)
        signal = np.zeros(total)
        phase = 0.0
        for i, val in enumerate(data):
            freq = mapping.map_value(float(val))
            freq = self._quantize_to_scale(freq)
            t = np.arange(n_samples_per_note) / sr
            start = i * n_samples_per_note
            end = start + n_samples_per_note
            env = np.ones(n_samples_per_note)
            fade = min(n_samples_per_note // 10, 50)
            if fade > 0:
                env[:fade] = np.linspace(0, 1, fade)
                env[-fade:] = np.linspace(1, 0, fade)
            signal[start:end] = env * np.sin(2 * np.pi * freq * t + phase)
            phase += 2 * np.pi * freq * note_duration
        return signal


@dataclass
class RhythmSonifier:
    """Map categorical/binary data to rhythmic click patterns."""

    tempo: float = 120.0

    def sonify(
        self, events: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        beat_duration = 60.0 / self.tempo
        total_duration = len(events) * beat_duration
        n_samples = int(sr * total_duration)
        signal = np.zeros(n_samples)
        click_dur = 0.005
        click_samples = int(sr * click_dur)
        click = np.sin(2 * np.pi * 1000 * np.arange(click_samples) / sr)
        click *= np.linspace(1, 0, click_samples)
        for i, val in enumerate(events):
            if val != 0:
                pos = int(i * beat_duration * sr)
                end = min(pos + click_samples, n_samples)
                n = end - pos
                if n > 0:
                    amplitude = min(abs(float(val)), 1.0)
                    signal[pos:end] += amplitude * click[:n]
        return signal


@dataclass
class MultidimensionalSonifier:
    """Map multi-dimensional data to audio parameters simultaneously."""

    def sonify(
        self,
        data: np.ndarray,
        sr: int = 16000,
        duration: float = 5.0,
        pitch_range: tuple = (200, 2000),
        pan_range: tuple = (-1, 1),
    ) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_points, n_dims = data.shape
        n_samples = int(sr * duration)
        samples_per_point = max(1, n_samples // n_points)
        signal = np.zeros(n_samples)
        for i in range(n_points):
            start = i * samples_per_point
            end = min(start + samples_per_point, n_samples)
            t = np.arange(end - start) / sr
            norm_vals = np.zeros(n_dims)
            for d in range(n_dims):
                col = data[:, d]
                mn, mx = np.min(col), np.max(col)
                if mx > mn:
                    norm_vals[d] = (data[i, d] - mn) / (mx - mn)
                else:
                    norm_vals[d] = 0.5
            freq = pitch_range[0] + norm_vals[0] * (pitch_range[1] - pitch_range[0])
            amplitude = norm_vals[1] if n_dims > 1 else 0.5
            brightness = norm_vals[2] if n_dims > 2 else 0.0
            tone = amplitude * np.sin(2 * np.pi * freq * t)
            if brightness > 0.3:
                tone += 0.3 * brightness * np.sin(4 * np.pi * freq * t)
            signal[start:end] = tone
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal /= peak
        return signal


@dataclass
class SonificationEvaluator:
    """Evaluate quality of sonification mappings."""

    def discriminability(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> float:
        if len(original) != len(reconstructed) or len(original) < 2:
            return 0.0
        corr = np.corrcoef(original.ravel(), reconstructed.ravel())[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(np.clip((corr + 1) / 2, 0, 1))

    def information_density(
        self, data: np.ndarray, audio_duration: float
    ) -> float:
        if audio_duration <= 0 or len(data) == 0:
            return 0.0
        n_unique = len(np.unique(np.round(data.ravel(), 2)))
        bits = np.log2(max(n_unique, 1))
        return float(bits * len(data.ravel()) / audio_duration)

    def temporal_resolution(
        self, n_data_points: int, audio_duration: float
    ) -> float:
        if audio_duration <= 0:
            return 0.0
        return n_data_points / audio_duration
