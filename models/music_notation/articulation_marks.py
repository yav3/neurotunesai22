"""Articulation marks: encode, interpret, and render performance markings.

Map between symbolic articulation notations (staccato, legato, accent, etc.)
and their acoustic realisations: duration ratios, velocity offsets,
attack/release envelopes, and combined articulation chains.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ArticulationType(str, Enum):
    STACCATO = "staccato"
    STACCATISSIMO = "staccatissimo"
    TENUTO = "tenuto"
    ACCENT = "accent"
    MARCATO = "marcato"
    LEGATO = "legato"
    PORTATO = "portato"
    SFORZANDO = "sforzando"
    FERMATA = "fermata"
    TRILL = "trill"
    MORDENT = "mordent"
    TURN = "turn"
    TREMOLO = "tremolo"
    PIZZICATO = "pizzicato"
    ARCO = "arco"
    HARMONICS = "harmonics"


@dataclass
class ArticulationEffect:
    duration_ratio: float
    velocity_offset: float
    attack_shape: str
    release_shape: str


ARTICULATION_DEFAULTS: Dict[ArticulationType, ArticulationEffect] = {
    ArticulationType.STACCATO: ArticulationEffect(0.5, 0, "sharp", "sharp"),
    ArticulationType.STACCATISSIMO: ArticulationEffect(0.25, 5, "sharp", "sharp"),
    ArticulationType.TENUTO: ArticulationEffect(1.0, 0, "normal", "smooth"),
    ArticulationType.ACCENT: ArticulationEffect(0.9, 20, "sharp", "normal"),
    ArticulationType.MARCATO: ArticulationEffect(0.75, 30, "sharp", "sharp"),
    ArticulationType.LEGATO: ArticulationEffect(1.05, -5, "smooth", "smooth"),
    ArticulationType.PORTATO: ArticulationEffect(0.8, 0, "normal", "smooth"),
    ArticulationType.SFORZANDO: ArticulationEffect(0.85, 40, "sharp", "normal"),
    ArticulationType.FERMATA: ArticulationEffect(1.8, 0, "normal", "smooth"),
    ArticulationType.PIZZICATO: ArticulationEffect(0.3, 10, "sharp", "sharp"),
}


@dataclass
class NoteWithArticulation:
    pitch: int
    onset_ms: float
    duration_ms: float
    velocity: int
    articulations: List[ArticulationType]


def apply_articulation(note: NoteWithArticulation) -> Tuple[float, float, int]:
    dur = note.duration_ms
    vel = note.velocity
    onset_shift = 0.0
    for art in note.articulations:
        effect = ARTICULATION_DEFAULTS.get(art)
        if effect is None:
            continue
        dur *= effect.duration_ratio
        vel = int(np.clip(vel + effect.velocity_offset, 1, 127))
    return note.onset_ms + onset_shift, dur, vel


def trill_expansion(pitch: int, interval: int = 2,
                    n_notes: int = 8, note_dur_ms: float = 50.0) -> List[Tuple[int, float]]:
    notes = []
    for i in range(n_notes):
        p = pitch if i % 2 == 0 else pitch + interval
        notes.append((p, i * note_dur_ms))
    return notes


def mordent_expansion(pitch: int, upper: bool = True,
                      interval: int = 2,
                      ornament_dur_ms: float = 60.0) -> List[Tuple[int, float]]:
    neighbor = pitch + interval if upper else pitch - interval
    third = ornament_dur_ms / 3
    return [
        (pitch, 0.0),
        (neighbor, third),
        (pitch, 2 * third),
    ]


def turn_expansion(pitch: int, interval: int = 2,
                   dur_ms: float = 120.0) -> List[Tuple[int, float]]:
    q = dur_ms / 4
    return [
        (pitch + interval, 0.0),
        (pitch, q),
        (pitch - interval, 2 * q),
        (pitch, 3 * q),
    ]


def tremolo_expansion(pitch: int, n_repeats: int = 8,
                      note_dur_ms: float = 40.0) -> List[Tuple[int, float]]:
    return [(pitch, i * note_dur_ms) for i in range(n_repeats)]


def articulation_envelope(art_type: ArticulationType,
                          n_samples: int, sr: int = 44100) -> np.ndarray:
    effect = ARTICULATION_DEFAULTS.get(art_type)
    if effect is None:
        return np.ones(n_samples)
    t = np.linspace(0, 1, n_samples)
    if effect.attack_shape == "sharp":
        attack = np.minimum(t * 20, 1.0)
    elif effect.attack_shape == "smooth":
        attack = 1 - np.exp(-5 * t)
    else:
        attack = np.minimum(t * 10, 1.0)
    if effect.release_shape == "sharp":
        release = np.minimum((1 - t) * 20, 1.0)
    elif effect.release_shape == "smooth":
        release = np.exp(-3 * np.maximum(t - 0.7, 0) / 0.3)
    else:
        release = np.minimum((1 - t) * 10, 1.0)
    return attack * release


def detect_articulation(durations_ms: np.ndarray,
                        velocities: np.ndarray,
                        nominal_dur_ms: float) -> List[ArticulationType]:
    results = []
    for dur, vel in zip(durations_ms, velocities):
        ratio = dur / max(nominal_dur_ms, 1.0)
        if ratio < 0.35:
            results.append(ArticulationType.STACCATISSIMO)
        elif ratio < 0.6:
            results.append(ArticulationType.STACCATO)
        elif ratio > 1.5:
            results.append(ArticulationType.FERMATA)
        elif ratio > 0.95 and vel > 100:
            results.append(ArticulationType.ACCENT)
        elif ratio > 0.95:
            results.append(ArticulationType.TENUTO)
        else:
            results.append(ArticulationType.LEGATO)
    return results


def articulation_density(notes: List[NoteWithArticulation],
                         window_ms: float = 2000.0) -> np.ndarray:
    if not notes:
        return np.array([0.0])
    total_dur = notes[-1].onset_ms - notes[0].onset_ms
    if total_dur <= 0:
        return np.array([float(sum(len(n.articulations) for n in notes))])
    n_windows = max(1, int(total_dur / window_ms))
    density = np.zeros(n_windows)
    for n in notes:
        w = int((n.onset_ms - notes[0].onset_ms) / window_ms)
        w = min(w, n_windows - 1)
        density[w] += len(n.articulations)
    return density


def combined_effect(articulations: List[ArticulationType]) -> ArticulationEffect:
    if not articulations:
        return ArticulationEffect(1.0, 0, "normal", "normal")
    dur_ratio = 1.0
    vel_offset = 0.0
    attack = "normal"
    release = "normal"
    for art in articulations:
        eff = ARTICULATION_DEFAULTS.get(art)
        if eff is None:
            continue
        dur_ratio *= eff.duration_ratio
        vel_offset += eff.velocity_offset
        if eff.attack_shape == "sharp":
            attack = "sharp"
        if eff.release_shape == "sharp":
            release = "sharp"
    return ArticulationEffect(dur_ratio, vel_offset, attack, release)
