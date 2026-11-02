"""Event-level haptic feedback mapping: beats, melody, dynamics, rhythm patterns."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class HapticPattern(Enum):
    PULSE = "pulse"
    BUZZ = "buzz"
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    DOUBLE_TAP = "double_tap"
    SUSTAINED = "sustained"


@dataclass
class HapticEvent:
    time_s: float
    intensity: float  # 0..1
    frequency_hz: float  # vibration frequency
    duration_ms: float
    pattern: HapticPattern = HapticPattern.PULSE


def beat_to_haptic(
    beat_times: np.ndarray,
    tempo: float,
    downbeat_boost: float = 0.3,
    beats_per_bar: int = 4,
) -> list[HapticEvent]:
    """Convert beat timestamps to haptic pulses with metrical accent."""
    if len(beat_times) == 0:
        return []
    beat_dur_ms = 60000.0 / max(tempo, 1.0)
    pulse_ms = min(80.0, beat_dur_ms * 0.25)
    events = []
    for i, t in enumerate(beat_times):
        is_downbeat = (i % beats_per_bar) == 0
        intensity = min(1.0, 0.6 + downbeat_boost) if is_downbeat else 0.6
        freq = 200.0 if is_downbeat else 150.0
        pat = HapticPattern.DOUBLE_TAP if is_downbeat else HapticPattern.PULSE
        events.append(HapticEvent(
            time_s=float(t),
            intensity=intensity,
            frequency_hz=freq,
            duration_ms=pulse_ms,
            pattern=pat,
        ))
    return events


def melody_to_haptic(
    pitches: np.ndarray,
    durations: np.ndarray,
    pitch_range: tuple[float, float] = (40.0, 300.0),
    midi_range: tuple[int, int] = (36, 96),
) -> list[HapticEvent]:
    """Map a melody line to haptic events: pitch → vibration frequency, velocity → intensity."""
    if len(pitches) == 0:
        return []
    lo_midi, hi_midi = midi_range
    lo_freq, hi_freq = pitch_range
    span_midi = max(hi_midi - lo_midi, 1)
    events = []
    t = 0.0
    for pitch, dur in zip(pitches, durations):
        normed = np.clip((pitch - lo_midi) / span_midi, 0.0, 1.0)
        freq = lo_freq + normed * (hi_freq - lo_freq)
        events.append(HapticEvent(
            time_s=t,
            intensity=float(normed * 0.5 + 0.3),
            frequency_hz=float(freq),
            duration_ms=float(dur * 1000.0),
            pattern=HapticPattern.SUSTAINED,
        ))
        t += dur
    return events


def dynamics_to_intensity(
    rms_values: np.ndarray,
    floor: float = 0.05,
    ceiling: float = 1.0,
) -> np.ndarray:
    """Normalise an RMS envelope to [floor, ceiling] for haptic intensity."""
    rms = np.asarray(rms_values, dtype=float)
    if len(rms) == 0:
        return np.array([], dtype=float)
    mn, mx = rms.min(), rms.max()
    if mx - mn < 1e-12:
        return np.full_like(rms, (floor + ceiling) / 2)
    normed = (rms - mn) / (mx - mn)
    return floor + normed * (ceiling - floor)


def rhythm_pattern_to_vibration(
    onset_pattern: np.ndarray,
    grid_resolution: int = 16,
    base_freq: float = 150.0,
    accent_freq: float = 250.0,
) -> list[HapticEvent]:
    """Convert a binary onset grid to vibration events.

    onset_pattern: 1D array of length grid_resolution with 1 at onsets, 0 elsewhere.
    Values >1 are treated as accented onsets.
    """
    pat = np.asarray(onset_pattern, dtype=float)
    if len(pat) == 0:
        return []
    step_dur_ms = 1000.0 / grid_resolution
    events = []
    for i, val in enumerate(pat):
        if val > 0:
            accented = val > 1.0
            events.append(HapticEvent(
                time_s=i * step_dur_ms / 1000.0,
                intensity=min(1.0, float(val) / 2.0) if accented else 0.5,
                frequency_hz=accent_freq if accented else base_freq,
                duration_ms=step_dur_ms * 0.6,
                pattern=HapticPattern.PULSE,
            ))
    return events


def continuous_haptic_envelope(
    events: list[HapticEvent],
    duration_s: float,
    sample_rate: int = 250,
) -> np.ndarray:
    """Render a list of HapticEvents into a continuous intensity envelope."""
    n = int(duration_s * sample_rate)
    if n <= 0:
        return np.array([], dtype=float)
    env = np.zeros(n, dtype=float)
    for ev in events:
        start = int(ev.time_s * sample_rate)
        length = max(1, int(ev.duration_ms / 1000.0 * sample_rate))
        end = min(start + length, n)
        if start < 0 or start >= n:
            continue
        if ev.pattern == HapticPattern.RAMP_UP:
            ramp = np.linspace(0, ev.intensity, end - start)
        elif ev.pattern == HapticPattern.RAMP_DOWN:
            ramp = np.linspace(ev.intensity, 0, end - start)
        elif ev.pattern == HapticPattern.DOUBLE_TAP:
            seg = np.zeros(end - start)
            mid = len(seg) // 2
            seg[:mid] = ev.intensity
            seg[mid:min(mid + max(1, mid // 2), len(seg))] = ev.intensity * 0.7
            ramp = seg
        else:
            ramp = np.full(end - start, ev.intensity)
        env[start:end] = np.maximum(env[start:end], ramp)
    return env
