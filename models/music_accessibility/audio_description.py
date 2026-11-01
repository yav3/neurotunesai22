"""Verbal/textual description of musical content for screen readers."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_INTERVAL_NAMES = {
    0: "unison", 1: "minor second", 2: "major second", 3: "minor third",
    4: "major third", 5: "perfect fourth", 6: "tritone", 7: "perfect fifth",
    8: "minor sixth", 9: "major sixth", 10: "minor seventh",
    11: "major seventh", 12: "octave",
}

_DYNAMICS = [
    (0, "silent"), (16, "ppp"), (32, "pp"), (48, "p"),
    (64, "mp"), (80, "mf"), (96, "f"), (112, "ff"), (127, "fff"),
]

_CHORD_TEMPLATES = {
    (0, 4, 7): "major triad",
    (0, 3, 7): "minor triad",
    (0, 3, 6): "diminished triad",
    (0, 4, 8): "augmented triad",
    (0, 4, 7, 11): "major seventh",
    (0, 3, 7, 10): "minor seventh",
    (0, 4, 7, 10): "dominant seventh",
    (0, 3, 6, 9): "diminished seventh",
    (0, 3, 6, 10): "half-diminished seventh",
}

_DURATION_WORDS = {
    4.0: "whole note", 2.0: "half note", 1.5: "dotted quarter",
    1.0: "quarter note", 0.75: "dotted eighth", 0.5: "eighth note",
    0.25: "sixteenth note",
}


class DescriptionLevel(Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


@dataclass
class AudioDescription:
    rhythm: str = ""
    harmony: str = ""
    dynamics: str = ""
    timbre: str = ""
    level: DescriptionLevel = DescriptionLevel.STANDARD

    def full_text(self) -> str:
        parts = [s for s in [self.rhythm, self.harmony, self.dynamics, self.timbre] if s]
        return ". ".join(parts) + ("." if parts else "")


_TEMPO_WORDS: list[tuple[float, str]] = [
    (0, "very slow"), (40, "grave"), (60, "adagio"), (76, "andante"),
    (108, "moderato"), (120, "allegro"), (156, "vivace"), (176, "presto"),
]


def describe_tempo(tempo: float) -> str:
    word = "very slow"
    for threshold, label in _TEMPO_WORDS:
        if tempo >= threshold:
            word = label
    return f"{word} ({tempo:.0f} BPM)"


def describe_rhythm_features(
    tempo: float,
    time_sig: tuple[int, int] = (4, 4),
    swing_ratio: float = 1.0,
) -> str:
    tempo_desc = describe_tempo(tempo)
    meter_desc = f"{time_sig[0]}/{time_sig[1]} time"
    parts = [f"{tempo_desc}, {meter_desc}"]
    if swing_ratio > 1.3:
        parts.append("with a swung feel")
    elif swing_ratio < 0.8:
        parts.append("with a reverse-swing feel")
    return ", ".join(parts)


def describe_harmony_features(
    key: str,
    mode: str = "major",
    chord_progression: list[str] | None = None,
) -> str:
    parts = [f"{key} {mode}"]
    if chord_progression:
        parts.append("progression: " + " – ".join(chord_progression))
    return ", ".join(parts)


def describe_dynamics_features(
    rms_envelope: np.ndarray,
    peak_db: float | None = None,
) -> str:
    env = np.asarray(rms_envelope, dtype=float)
    if len(env) == 0:
        return "no dynamic information"
    mean_rms = float(np.mean(env))
    std_rms = float(np.std(env))
    if mean_rms < 0.05:
        level = "very quiet"
    elif mean_rms < 0.15:
        level = "quiet"
    elif mean_rms < 0.4:
        level = "moderate"
    elif mean_rms < 0.7:
        level = "loud"
    else:
        level = "very loud"
    parts = [f"overall {level}"]
    if std_rms > 0.1:
        parts.append("with wide dynamic range")
    elif std_rms < 0.02:
        parts.append("with little dynamic variation")
    if peak_db is not None:
        parts.append(f"peak at {peak_db:.1f} dB")
    return ", ".join(parts)


def describe_timbre_features(
    spectral_centroid: float,
    brightness: float | None = None,
    roughness: float | None = None,
) -> str:
    if spectral_centroid < 500:
        tone = "dark, warm tone"
    elif spectral_centroid < 2000:
        tone = "balanced tone"
    elif spectral_centroid < 5000:
        tone = "bright tone"
    else:
        tone = "very bright, sharp tone"
    parts = [tone]
    if brightness is not None:
        if brightness > 0.7:
            parts.append("highly bright")
        elif brightness < 0.3:
            parts.append("subdued brightness")
    if roughness is not None:
        if roughness > 0.5:
            parts.append("rough or distorted texture")
        elif roughness < 0.1:
            parts.append("smooth texture")
    return ", ".join(parts)


def generate_full_description(
    features: dict,
    level: DescriptionLevel = DescriptionLevel.STANDARD,
) -> AudioDescription:
    """Build a complete AudioDescription from a dict of musical features."""
    rhythm = ""
    if "tempo" in features:
        rhythm = describe_rhythm_features(
            features["tempo"],
            features.get("time_signature", (4, 4)),
            features.get("swing_ratio", 1.0),
        )
    harmony = ""
    if "key" in features:
        harmony = describe_harmony_features(
            features["key"],
            features.get("mode", "major"),
            features.get("chord_progression"),
        )
    dynamics = ""
    if "rms_envelope" in features:
        dynamics = describe_dynamics_features(
            features["rms_envelope"],
            features.get("peak_db"),
        )
    timbre = ""
    if "spectral_centroid" in features:
        timbre = describe_timbre_features(
            features["spectral_centroid"],
            features.get("brightness"),
            features.get("roughness"),
        )
    if level == DescriptionLevel.BRIEF:
        rhythm = describe_tempo(features["tempo"]) if "tempo" in features else ""
        harmony = f"{features['key']} {features.get('mode', 'major')}" if "key" in features else ""
        timbre = ""
    return AudioDescription(
        rhythm=rhythm,
        harmony=harmony,
        dynamics=dynamics,
        timbre=timbre,
        level=level,
    )


def describe_pitch(midi_pitch: int) -> str:
    pc = midi_pitch % 12
    octave = midi_pitch // 12 - 1
    name = _NOTE_NAMES[pc]
    if midi_pitch == 60:
        return "middle C"
    if octave == 4:
        return f"{name} above middle C" if pc > 0 else "middle C"
    return f"{name}{octave}"


def describe_interval(semitones: int) -> str:
    semitones = abs(semitones)
    if semitones <= 12:
        return _INTERVAL_NAMES.get(semitones, f"{semitones} semitones")
    octaves = semitones // 12
    remainder = semitones % 12
    base = _INTERVAL_NAMES.get(remainder, f"{remainder} semitones")
    if remainder == 0:
        return f"{octaves} octaves" if octaves > 1 else "octave"
    return f"{base} plus {octaves} octave{'s' if octaves > 1 else ''}"


def describe_rhythm(durations: list[float], bpm: float) -> str:
    if not durations:
        return "silence"
    parts = []
    for d in durations:
        closest = min(_DURATION_WORDS.keys(), key=lambda k: abs(k - d))
        if abs(closest - d) < 0.1:
            parts.append(_DURATION_WORDS[closest])
        else:
            ms = d * 60000.0 / max(bpm, 1.0)
            parts.append(f"{ms:.0f}ms")
    total_beats = sum(durations)
    return f"{len(parts)} notes over {total_beats:.1f} beats: {', '.join(parts)}"


def describe_dynamics(velocity: int) -> str:
    v = max(0, min(127, velocity))
    result = "silent"
    for threshold, label in _DYNAMICS:
        if v >= threshold:
            result = label
    return result


def describe_chord(pitches: list[int]) -> str:
    if not pitches:
        return "silence"
    if len(pitches) == 1:
        return describe_pitch(pitches[0])
    pcs = sorted(set(p % 12 for p in pitches))
    root = pcs[0]
    intervals = tuple((pc - root) % 12 for pc in pcs)
    root_name = _NOTE_NAMES[root]
    quality = _CHORD_TEMPLATES.get(intervals)
    if quality:
        return f"{root_name} {quality}"
    return f"{root_name} chord ({', '.join(_NOTE_NAMES[pc] for pc in pcs)})"


def describe_progression(chords: list[list[int]]) -> str:
    if not chords:
        return "no chords"
    descriptions = [describe_chord(c) for c in chords]
    return " → ".join(descriptions)


def generate_audio_guide(
    pitches: list[int],
    durations: list[float],
    velocities: list[int] | None = None,
) -> list[str]:
    if velocities is None:
        velocities = [64] * len(pitches)
    guide = []
    for i, (p, d, v) in enumerate(zip(pitches, durations, velocities)):
        closest_dur = min(_DURATION_WORDS.keys(), key=lambda k: abs(k - d))
        dur_word = _DURATION_WORDS.get(closest_dur, f"{d} beats")
        dyn = describe_dynamics(v)
        pitch_desc = describe_pitch(p)
        parts = [pitch_desc, dur_word, dyn]
        if i > 0:
            interval = abs(p - pitches[i - 1])
            direction = "up" if p > pitches[i - 1] else "down" if p < pitches[i - 1] else "repeated"
            parts.append(f"{direction} {describe_interval(interval)}")
        guide.append(", ".join(parts))
    return guide
