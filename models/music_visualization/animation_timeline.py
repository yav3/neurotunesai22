"""Animation timeline for music visualization: keyframes, interpolation, sync to audio.

Provides a general-purpose animation system for music visualizations:
time-aligned keyframes, easing functions, beat-sync, parameter automation,
and multi-track timeline composition.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from enum import Enum


class EasingType(Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"       # quadratic
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    CUBIC_IN = "cubic_in"
    CUBIC_OUT = "cubic_out"
    ELASTIC = "elastic"
    BOUNCE = "bounce"
    EXPONENTIAL = "exponential"


def ease(t: float, easing: EasingType = EasingType.LINEAR) -> float:
    """Map t ∈ [0,1] through easing function → [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    if easing == EasingType.LINEAR:
        return float(t)
    elif easing == EasingType.EASE_IN:
        return float(t * t)
    elif easing == EasingType.EASE_OUT:
        return float(1 - (1 - t) ** 2)
    elif easing == EasingType.EASE_IN_OUT:
        return float(2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2)
    elif easing == EasingType.CUBIC_IN:
        return float(t ** 3)
    elif easing == EasingType.CUBIC_OUT:
        return float(1 - (1 - t) ** 3)
    elif easing == EasingType.ELASTIC:
        if t == 0 or t == 1:
            return float(t)
        p = 0.3
        return float(-(2 ** (10 * (t - 1))) * np.sin((t - 1 - p / 4) * 2 * np.pi / p))
    elif easing == EasingType.BOUNCE:
        if t < 1 / 2.75:
            return float(7.5625 * t * t)
        elif t < 2 / 2.75:
            t -= 1.5 / 2.75
            return float(7.5625 * t * t + 0.75)
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return float(7.5625 * t * t + 0.9375)
        else:
            t -= 2.625 / 2.75
            return float(7.5625 * t * t + 0.984375)
    elif easing == EasingType.EXPONENTIAL:
        return float(0 if t == 0 else 2 ** (10 * (t - 1)))
    return float(t)


def ease_array(t: np.ndarray, easing: EasingType = EasingType.LINEAR) -> np.ndarray:
    return np.array([ease(float(x), easing) for x in t])


@dataclass
class Keyframe:
    time: float
    value: float
    easing: EasingType = EasingType.LINEAR


@dataclass
class AnimationTrack:
    """Single parameter animation track with keyframes."""
    name: str
    keyframes: List[Keyframe] = field(default_factory=list)

    def add(self, time: float, value: float,
            easing: EasingType = EasingType.LINEAR) -> None:
        self.keyframes.append(Keyframe(time, value, easing))
        self.keyframes.sort(key=lambda k: k.time)

    def evaluate(self, t: float) -> float:
        if not self.keyframes:
            return 0.0
        if t <= self.keyframes[0].time:
            return self.keyframes[0].value
        if t >= self.keyframes[-1].time:
            return self.keyframes[-1].value
        for i in range(len(self.keyframes) - 1):
            k0, k1 = self.keyframes[i], self.keyframes[i + 1]
            if k0.time <= t <= k1.time:
                frac = (t - k0.time) / max(k1.time - k0.time, 1e-10)
                eased = ease(frac, k1.easing)
                return k0.value + (k1.value - k0.value) * eased
        return self.keyframes[-1].value

    def evaluate_array(self, times: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate(float(t)) for t in times])

    @property
    def duration(self) -> float:
        if not self.keyframes:
            return 0.0
        return self.keyframes[-1].time - self.keyframes[0].time


@dataclass
class BeatSync:
    """Synchronise animation events to musical beats."""
    bpm: float
    time_signature_num: int = 4
    time_signature_den: int = 4
    offset_s: float = 0.0

    @property
    def beat_duration(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_duration(self) -> float:
        return self.beat_duration * self.time_signature_num

    def beat_times(self, n_bars: int) -> np.ndarray:
        total_beats = n_bars * self.time_signature_num
        return self.offset_s + np.arange(total_beats) * self.beat_duration

    def bar_times(self, n_bars: int) -> np.ndarray:
        return self.offset_s + np.arange(n_bars) * self.bar_duration

    def time_to_beat(self, t: float) -> float:
        return (t - self.offset_s) / self.beat_duration

    def beat_to_time(self, beat: float) -> float:
        return self.offset_s + beat * self.beat_duration

    def snap_to_beat(self, t: float, subdivision: int = 1) -> float:
        sub_dur = self.beat_duration / subdivision
        n = round((t - self.offset_s) / sub_dur)
        return self.offset_s + n * sub_dur


@dataclass
class Timeline:
    """Multi-track animation timeline with beat sync."""
    tracks: Dict[str, AnimationTrack] = field(default_factory=dict)
    beat_sync: Optional[BeatSync] = None
    fps: float = 30.0

    def add_track(self, name: str) -> AnimationTrack:
        track = AnimationTrack(name)
        self.tracks[name] = track
        return track

    def get_or_create(self, name: str) -> AnimationTrack:
        if name not in self.tracks:
            return self.add_track(name)
        return self.tracks[name]

    def evaluate(self, t: float) -> Dict[str, float]:
        return {name: track.evaluate(t) for name, track in self.tracks.items()}

    def render_frames(self, duration: float) -> List[Dict[str, float]]:
        n_frames = int(duration * self.fps)
        times = np.linspace(0, duration, n_frames)
        return [self.evaluate(float(t)) for t in times]

    @property
    def duration(self) -> float:
        if not self.tracks:
            return 0.0
        return max(t.duration for t in self.tracks.values())


def beat_reactive_track(beat_sync: BeatSync, n_bars: int,
                        attack: float = 0.05, decay: float = 0.3,
                        peak: float = 1.0) -> AnimationTrack:
    """Generate a track that pulses on each beat with exponential decay."""
    track = AnimationTrack("beat_pulse")
    for bt in beat_sync.beat_times(n_bars):
        track.add(bt, peak, EasingType.LINEAR)
        track.add(bt + attack, peak, EasingType.LINEAR)
        track.add(bt + attack + decay, 0.0, EasingType.EASE_OUT)
    return track


def onset_reactive_track(onsets: np.ndarray, values: Optional[np.ndarray] = None,
                         decay: float = 0.2) -> AnimationTrack:
    """Generate a track that reacts to detected audio onsets."""
    track = AnimationTrack("onset_reactive")
    if values is None:
        values = np.ones(len(onsets))
    for onset, val in zip(onsets, values):
        track.add(float(onset), float(val), EasingType.LINEAR)
        track.add(float(onset) + decay, 0.0, EasingType.EASE_OUT)
    return track


def spectrum_band_tracks(times: np.ndarray, spectrogram: np.ndarray,
                         band_edges: List[Tuple[int, int]],
                         names: Optional[List[str]] = None) -> Dict[str, AnimationTrack]:
    """Create animation tracks from frequency band energies over time."""
    if names is None:
        names = [f"band_{i}" for i in range(len(band_edges))]
    tracks = {}
    for name, (lo, hi) in zip(names, band_edges):
        band_energy = np.mean(spectrogram[lo:hi, :], axis=0)
        band_energy = band_energy / max(np.max(band_energy), 1e-10)
        track = AnimationTrack(name)
        for t_val, e_val in zip(times, band_energy):
            track.add(float(t_val), float(e_val))
        tracks[name] = track
    return tracks


def lfo_track(frequency: float, duration: float, amplitude: float = 1.0,
              offset: float = 0.0, waveform: str = "sine",
              n_points: int = 256) -> AnimationTrack:
    """Low-frequency oscillator as an animation track."""
    track = AnimationTrack(f"lfo_{waveform}_{frequency}hz")
    t = np.linspace(0, duration, n_points)
    if waveform == "sine":
        values = offset + amplitude * np.sin(2 * np.pi * frequency * t)
    elif waveform == "triangle":
        values = offset + amplitude * (2 * np.abs(2 * (frequency * t % 1) - 1) - 1)
    elif waveform == "square":
        values = offset + amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == "sawtooth":
        values = offset + amplitude * (2 * (frequency * t % 1) - 1)
    else:
        values = np.full(n_points, offset)
    for tv, vv in zip(t, values):
        track.add(float(tv), float(vv))
    return track


def crossfade(track_a: AnimationTrack, track_b: AnimationTrack,
              start: float, duration: float, n_points: int = 64) -> AnimationTrack:
    """Crossfade between two tracks over a time interval."""
    result = AnimationTrack(f"xfade_{track_a.name}_{track_b.name}")
    t = np.linspace(start, start + duration, n_points)
    for tv in t:
        frac = (tv - start) / duration
        va = track_a.evaluate(float(tv))
        vb = track_b.evaluate(float(tv))
        result.add(float(tv), va * (1 - frac) + vb * frac)
    return result
