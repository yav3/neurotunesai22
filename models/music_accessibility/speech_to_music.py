"""Voice-controlled music interaction: pitch tracking, MIDI conversion, gesture detection."""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PitchTracker:
    """Extract fundamental frequency from voice input via autocorrelation."""

    min_freq: float = 80.0
    max_freq: float = 600.0

    def track(
        self, signal: np.ndarray, sr: int, frame_size: int = 1024, hop: int = 512
    ) -> np.ndarray:
        if len(signal) < frame_size:
            signal = np.pad(signal, (0, frame_size - len(signal)))
        n_frames = max(1, (len(signal) - frame_size) // hop + 1)
        f0 = np.zeros(n_frames)
        min_lag = int(sr / self.max_freq)
        max_lag = int(sr / self.min_freq)
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_size) / frame_size))
        for i in range(n_frames):
            start = i * hop
            frame = signal[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            frame = frame * w
            energy = np.sum(frame ** 2)
            if energy < 1e-10:
                continue
            ac = np.correlate(frame, frame, mode="full")
            ac = ac[len(ac) // 2:]
            if ac[0] > 0:
                ac /= ac[0]
            search_region = ac[min_lag:max_lag + 1]
            if len(search_region) < 2:
                continue
            peak_idx = np.argmax(search_region)
            if search_region[peak_idx] > 0.3:
                lag = peak_idx + min_lag
                f0[i] = sr / lag
        return f0

    def voiced_segments(
        self, f0_track: np.ndarray, threshold: float = 0.5
    ) -> list:
        voiced = f0_track > 0
        segments = []
        in_segment = False
        start = 0
        for i in range(len(voiced)):
            if voiced[i] and not in_segment:
                start = i
                in_segment = True
            elif not voiced[i] and in_segment:
                if (i - start) >= 2:
                    segments.append((start, i - 1))
                in_segment = False
        if in_segment and (len(voiced) - start) >= 2:
            segments.append((start, len(voiced) - 1))
        return segments


@dataclass
class VoiceToMIDI:
    """Convert voice pitch track to MIDI-like note events."""

    quantize: bool = True
    scale: Optional[list] = None

    def convert(
        self, f0_track: np.ndarray, sr: int, hop: int
    ) -> list:
        events = []
        current_note = None
        onset_frame = 0
        for i, f0 in enumerate(f0_track):
            if f0 > 0:
                midi = 69 + 12 * np.log2(f0 / 440.0)
                if self.quantize:
                    midi = round(midi)
                if self.scale is not None:
                    midi = round(midi)
                    midi = min(self.scale, key=lambda x: abs(x - midi))
                if current_note is None:
                    current_note = midi
                    onset_frame = i
                elif abs(midi - current_note) > 0.5:
                    onset_sec = onset_frame * hop / sr
                    dur_sec = (i - onset_frame) * hop / sr
                    velocity = min(1.0, dur_sec * 2)
                    events.append({
                        "pitch": int(round(current_note)),
                        "onset": onset_sec,
                        "duration": dur_sec,
                        "velocity": velocity,
                    })
                    current_note = midi
                    onset_frame = i
            else:
                if current_note is not None:
                    onset_sec = onset_frame * hop / sr
                    dur_sec = (i - onset_frame) * hop / sr
                    velocity = min(1.0, dur_sec * 2)
                    events.append({
                        "pitch": int(round(current_note)),
                        "onset": onset_sec,
                        "duration": dur_sec,
                        "velocity": velocity,
                    })
                    current_note = None
        if current_note is not None:
            onset_sec = onset_frame * hop / sr
            dur_sec = (len(f0_track) - onset_frame) * hop / sr
            events.append({
                "pitch": int(round(current_note)),
                "onset": onset_sec,
                "duration": dur_sec,
                "velocity": min(1.0, dur_sec * 2),
            })
        return events


@dataclass
class VocalGestureDetector:
    """Detect musical gestures from voice: onsets, vibrato, glissando."""

    def detect_onset(
        self, signal: np.ndarray, sr: int, frame_size: int = 512
    ) -> list:
        hop = frame_size // 2
        n_frames = max(1, (len(signal) - frame_size) // hop)
        energy = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            frame = signal[start:start + frame_size]
            energy[i] = np.sum(frame ** 2)
        onsets = []
        threshold = np.mean(energy) * 2
        for i in range(1, n_frames):
            if energy[i] > threshold and energy[i] > energy[i - 1] * 1.5:
                time = i * hop / sr
                onsets.append(time)
        return onsets

    def detect_vibrato(
        self, f0_track: np.ndarray, sr: int, hop: int
    ) -> dict:
        voiced = f0_track[f0_track > 0]
        if len(voiced) < 10:
            return {"rate_hz": 0.0, "depth_cents": 0.0, "regularity": 0.0}
        mean_f0 = np.mean(voiced)
        detrended = voiced - mean_f0
        ac = np.correlate(detrended, detrended, mode="full")
        ac = ac[len(ac) // 2:]
        if ac[0] > 0:
            ac /= ac[0]
        frame_rate = sr / hop
        min_period = int(frame_rate / 12)
        max_period = int(frame_rate / 3)
        search = ac[min_period:max_period + 1]
        if len(search) < 2:
            return {"rate_hz": 0.0, "depth_cents": 0.0, "regularity": 0.0}
        peak_idx = np.argmax(search)
        peak_val = search[peak_idx]
        period_frames = peak_idx + min_period
        rate = frame_rate / period_frames if period_frames > 0 else 0.0
        depth_hz = np.std(voiced)
        depth_cents = 1200.0 * np.log2(1 + depth_hz / mean_f0) if mean_f0 > 0 else 0.0
        regularity = float(np.clip(peak_val, 0, 1))
        return {
            "rate_hz": float(rate),
            "depth_cents": float(depth_cents),
            "regularity": regularity,
        }

    def detect_glissando(
        self, f0_track: np.ndarray, sr: int, hop: int, threshold_cents: float = 50.0
    ) -> list:
        voiced = f0_track.copy()
        glissandi = []
        start = None
        direction = None
        for i in range(1, len(voiced)):
            if voiced[i] > 0 and voiced[i - 1] > 0:
                cents = 1200 * np.log2(voiced[i] / voiced[i - 1])
                if abs(cents) > threshold_cents / 10:
                    d = "up" if cents > 0 else "down"
                    if start is None:
                        start = i - 1
                        direction = d
                    elif d != direction:
                        total_cents = 1200 * np.log2(voiced[i - 1] / voiced[start])
                        if abs(total_cents) >= threshold_cents:
                            glissandi.append((
                                start * hop / sr,
                                (i - 1) * hop / sr,
                                direction,
                            ))
                        start = i - 1
                        direction = d
                else:
                    if start is not None:
                        total_cents = 1200 * np.log2(voiced[i - 1] / voiced[start])
                        if abs(total_cents) >= threshold_cents:
                            glissandi.append((
                                start * hop / sr,
                                (i - 1) * hop / sr,
                                direction,
                            ))
                    start = None
                    direction = None
            else:
                if start is not None and voiced[i - 1] > 0 and voiced[start] > 0:
                    total_cents = 1200 * np.log2(voiced[i - 1] / voiced[start])
                    if abs(total_cents) >= threshold_cents:
                        glissandi.append((start * hop / sr, (i - 1) * hop / sr, direction))
                start = None
                direction = None
        return glissandi


@dataclass
class VoiceCommandClassifier:
    """Classify simple vocal commands using rule-based features."""

    commands: tuple = ("up", "down", "louder", "softer", "faster", "slower")

    def extract_features(self, signal: np.ndarray, sr: int) -> np.ndarray:
        if len(signal) < 2:
            return np.zeros(6)
        energy = np.sum(signal ** 2) / len(signal)
        duration = len(signal) / sr
        frame = min(1024, len(signal))
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame) / frame))
        chunk = signal[:frame]
        if len(chunk) < frame:
            chunk = np.pad(chunk, (0, frame - len(chunk)))
        spec = np.abs(np.fft.rfft(chunk * w))
        centroid = np.sum(spec * np.arange(len(spec))) / max(np.sum(spec), 1e-10)
        half = len(signal) // 2
        e1 = np.sum(signal[:half] ** 2)
        e2 = np.sum(signal[half:] ** 2)
        energy_slope = (e2 - e1) / max(e1 + e2, 1e-10)
        zcr = np.sum(np.abs(np.diff(np.sign(signal)))) / (2.0 * len(signal))
        spectral_spread = np.sqrt(
            np.sum(spec * (np.arange(len(spec)) - centroid) ** 2) / max(np.sum(spec), 1e-10)
        )
        return np.array([energy, duration, centroid, energy_slope, zcr, spectral_spread])

    def classify(self, signal: np.ndarray, sr: int) -> tuple[str, float]:
        feats = self.extract_features(signal, sr)
        energy, duration, centroid, slope, zcr, spread = feats
        if centroid > 200 and slope > 0.1:
            return "up", 0.6
        elif centroid > 200 and slope < -0.1:
            return "down", 0.6
        elif energy > 0.1 and slope > 0.1:
            return "louder", 0.5
        elif energy < 0.01:
            return "softer", 0.5
        elif zcr > 0.3:
            return "faster", 0.4
        elif zcr < 0.1:
            return "slower", 0.4
        return "up", 0.3
