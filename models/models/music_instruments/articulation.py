"""Articulation types and envelope shaping for instrument performance."""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ArticulationType(Enum):
    LEGATO = "legato"
    STACCATO = "staccato"
    MARCATO = "marcato"
    TENUTO = "tenuto"
    PIZZICATO = "pizzicato"
    ARCO = "arco"
    TREMOLO = "tremolo"
    TRILL = "trill"
    GLISSANDO = "glissando"
    HARMONICS = "harmonics"
    MUTED = "muted"
    OPEN = "open"
    BEND = "bend"
    VIBRATO = "vibrato"
    FLUTTER_TONGUE = "flutter_tongue"


@dataclass
class ArticulationProfile:
    attack_ms: float
    sustain_factor: float
    release_ms: float
    amplitude_mod: float = 1.0
    pitch_mod: float = 0.0


ARTICULATION_PROFILES: dict[ArticulationType, ArticulationProfile] = {
    ArticulationType.LEGATO: ArticulationProfile(50.0, 1.0, 80.0),
    ArticulationType.STACCATO: ArticulationProfile(10.0, 0.3, 20.0, 0.9),
    ArticulationType.MARCATO: ArticulationProfile(5.0, 0.7, 40.0, 1.2),
    ArticulationType.TENUTO: ArticulationProfile(30.0, 1.0, 60.0, 1.05),
    ArticulationType.PIZZICATO: ArticulationProfile(2.0, 0.1, 100.0, 0.8),
    ArticulationType.ARCO: ArticulationProfile(40.0, 1.0, 70.0),
    ArticulationType.TREMOLO: ArticulationProfile(15.0, 0.8, 30.0, 0.85),
    ArticulationType.TRILL: ArticulationProfile(10.0, 0.9, 25.0, 0.95, 1.0),
    ArticulationType.GLISSANDO: ArticulationProfile(20.0, 0.9, 30.0, 1.0, 2.0),
    ArticulationType.HARMONICS: ArticulationProfile(60.0, 0.6, 100.0, 0.5),
    ArticulationType.MUTED: ArticulationProfile(15.0, 0.5, 40.0, 0.4),
    ArticulationType.OPEN: ArticulationProfile(20.0, 1.0, 50.0, 1.0),
    ArticulationType.BEND: ArticulationProfile(25.0, 0.9, 40.0, 1.0, 1.5),
    ArticulationType.VIBRATO: ArticulationProfile(30.0, 1.0, 50.0, 0.95, 0.3),
    ArticulationType.FLUTTER_TONGUE: ArticulationProfile(10.0, 0.7, 25.0, 0.8),
}

_STRING_INSTRUMENTS = {"violin", "viola", "cello", "double_bass", "guitar", "harp"}
_BRASS_INSTRUMENTS = {"trumpet_bb", "french_horn", "trombone", "tuba"}
_WOODWIND_INSTRUMENTS = {"flute", "oboe", "clarinet_bb", "bassoon", "piccolo",
                         "alto_saxophone", "tenor_saxophone"}
_VOICE_TYPES = {"soprano", "alto", "tenor", "baritone", "bass_voice"}

_COMPATIBILITY: dict[ArticulationType, set[str]] = {
    ArticulationType.PIZZICATO: _STRING_INSTRUMENTS,
    ArticulationType.ARCO: _STRING_INSTRUMENTS,
    ArticulationType.HARMONICS: _STRING_INSTRUMENTS | _WOODWIND_INSTRUMENTS,
    ArticulationType.MUTED: _STRING_INSTRUMENTS | _BRASS_INSTRUMENTS,
    ArticulationType.FLUTTER_TONGUE: _WOODWIND_INSTRUMENTS | _BRASS_INSTRUMENTS,
    ArticulationType.BEND: _STRING_INSTRUMENTS | _BRASS_INSTRUMENTS | _VOICE_TYPES,
}


def apply_articulation(signal: np.ndarray, sr: int,
                       art_type: ArticulationType) -> np.ndarray:
    if len(signal) == 0:
        return signal.copy()
    prof = ARTICULATION_PROFILES[art_type]
    n = len(signal)
    attack_samples = max(1, int(prof.attack_ms * sr / 1000.0))
    release_samples = max(1, int(prof.release_ms * sr / 1000.0))
    env = np.ones(n)
    a_end = min(attack_samples, n)
    env[:a_end] = np.linspace(0, 1, a_end)
    sustain_end = max(a_end, n - release_samples)
    env[a_end:sustain_end] = prof.sustain_factor
    if sustain_end < n:
        env[sustain_end:] = np.linspace(prof.sustain_factor, 0, n - sustain_end)
    out = signal * env * prof.amplitude_mod
    if art_type == ArticulationType.TREMOLO:
        trem_rate = 8.0
        t = np.arange(n) / sr
        out *= 0.5 + 0.5 * np.sin(2.0 * np.pi * trem_rate * t)
    return out


def detect_articulation(signal: np.ndarray, sr: int,
                        frame_ms: float = 50.0) -> list[tuple[float, ArticulationType, float]]:
    if len(signal) < 2:
        return []
    frame_len = max(1, int(frame_ms * sr / 1000.0))
    n_frames = max(1, len(signal) // frame_len)
    results: list[tuple[float, ArticulationType, float]] = []
    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, len(signal))
        frame = signal[start:end]
        if len(frame) < 2:
            continue
        rms = float(np.sqrt(np.mean(frame ** 2)))
        zcr = float(np.sum(np.diff(np.sign(frame)) != 0)) / len(frame)
        t_sec = start / sr
        if rms < 0.01:
            continue
        onset_slope = abs(frame[min(5, len(frame) - 1)] - frame[0])
        if onset_slope > 0.5 and rms < 0.3:
            results.append((t_sec, ArticulationType.STACCATO, 0.6))
        elif onset_slope > 0.5:
            results.append((t_sec, ArticulationType.MARCATO, 0.5))
        elif zcr > 0.4:
            results.append((t_sec, ArticulationType.TREMOLO, 0.4))
        else:
            results.append((t_sec, ArticulationType.LEGATO, 0.5))
    return results


def articulation_compatibility(instrument: str, art_type: ArticulationType) -> float:
    restricted = _COMPATIBILITY.get(art_type)
    if restricted is None:
        return 0.8
    if instrument in restricted:
        return 1.0
    all_known = _STRING_INSTRUMENTS | _BRASS_INSTRUMENTS | _WOODWIND_INSTRUMENTS | _VOICE_TYPES | {"piano", "marimba"}
    if instrument in all_known:
        return 0.1
    return 0.5
