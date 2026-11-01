"""Articulation markings: staccato, legato, tenuto, and related note shaping."""
from __future__ import annotations

import enum
from dataclasses import dataclass


class ArticulationType(enum.Enum):
    LEGATO = "legato"
    STACCATO = "staccato"
    STACCATISSIMO = "staccatissimo"
    TENUTO = "tenuto"
    MARCATO = "marcato"
    PORTATO = "portato"
    ACCENT = "accent"
    NORMAL = "normal"


# Fraction of notated duration actually sounded.
ARTICULATION_DURATION_RATIO: dict[ArticulationType, float] = {
    ArticulationType.LEGATO: 1.0,
    ArticulationType.STACCATO: 0.5,
    ArticulationType.STACCATISSIMO: 0.25,
    ArticulationType.TENUTO: 1.0,
    ArticulationType.MARCATO: 0.75,
    ArticulationType.PORTATO: 0.85,
    ArticulationType.ACCENT: 0.85,
    ArticulationType.NORMAL: 0.9,
}

# Velocity multiplier applied on top of the base dynamic.
ARTICULATION_VELOCITY_SCALE: dict[ArticulationType, float] = {
    ArticulationType.LEGATO: 1.0,
    ArticulationType.STACCATO: 0.9,
    ArticulationType.STACCATISSIMO: 0.85,
    ArticulationType.TENUTO: 1.05,
    ArticulationType.MARCATO: 1.3,
    ArticulationType.PORTATO: 1.0,
    ArticulationType.ACCENT: 1.2,
    ArticulationType.NORMAL: 1.0,
}


@dataclass
class ArticulatedNote:
    """A note with articulation applied."""
    pitch_midi: int
    onset_beats: float
    notated_duration_beats: float
    sounded_duration_beats: float
    velocity: int
    articulation: ArticulationType


def apply_articulation(
    pitch_midi: int,
    onset_beats: float,
    duration_beats: float,
    base_velocity: int,
    articulation: ArticulationType,
) -> ArticulatedNote:
    """Apply an articulation marking to a note."""
    ratio = ARTICULATION_DURATION_RATIO[articulation]
    scale = ARTICULATION_VELOCITY_SCALE[articulation]
    sounded = duration_beats * ratio
    velocity = max(1, min(127, int(round(base_velocity * scale))))
    return ArticulatedNote(
        pitch_midi=pitch_midi,
        onset_beats=onset_beats,
        notated_duration_beats=duration_beats,
        sounded_duration_beats=sounded,
        velocity=velocity,
        articulation=articulation,
    )


def legato_overlap(
    notes: list[ArticulatedNote],
    overlap_beats: float = 0.05,
) -> list[ArticulatedNote]:
    """Extend legato notes to slightly overlap the next note onset."""
    result = []
    for i, note in enumerate(notes):
        if note.articulation == ArticulationType.LEGATO and i + 1 < len(notes):
            next_onset = notes[i + 1].onset_beats
            gap = next_onset - note.onset_beats
            extended = gap + overlap_beats
            result.append(ArticulatedNote(
                pitch_midi=note.pitch_midi,
                onset_beats=note.onset_beats,
                notated_duration_beats=note.notated_duration_beats,
                sounded_duration_beats=extended,
                velocity=note.velocity,
                articulation=note.articulation,
            ))
        else:
            result.append(note)
    return result


def detect_articulation_from_ratio(
    sounded: float,
    notated: float,
) -> ArticulationType:
    """Infer articulation type from the ratio of sounded to notated duration."""
    if notated <= 0:
        return ArticulationType.NORMAL
    ratio = sounded / notated
    if ratio <= 0.3:
        return ArticulationType.STACCATISSIMO
    if ratio <= 0.6:
        return ArticulationType.STACCATO
    if ratio <= 0.88:
        return ArticulationType.PORTATO
    if ratio >= 1.0:
        return ArticulationType.LEGATO
    return ArticulationType.NORMAL


def batch_articulate(
    pitches: list[int],
    onsets: list[float],
    durations: list[float],
    velocities: list[int],
    articulation: ArticulationType,
) -> list[ArticulatedNote]:
    """Apply the same articulation to a sequence of notes."""
    return [
        apply_articulation(p, o, d, v, articulation)
        for p, o, d, v in zip(pitches, onsets, durations, velocities)
    ]
