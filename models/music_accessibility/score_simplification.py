"""Score simplification for different ability levels."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AbilityLevel(Enum):
    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"


@dataclass
class SimplificationProfile:
    """Configuration for score simplification."""
    max_simultaneous_notes: int = 4
    max_interval_semitones: int = 12
    allow_accidentals: bool = True
    allow_key_changes: bool = True
    max_rhythmic_subdivision: int = 4  # 1=whole, 2=half, 4=quarter, 8=eighth, 16=sixteenth
    allow_syncopation: bool = True
    max_tempo: int = 160


LEVEL_PROFILES: dict[AbilityLevel, SimplificationProfile] = {
    AbilityLevel.BEGINNER: SimplificationProfile(
        max_simultaneous_notes=1, max_interval_semitones=5,
        allow_accidentals=False, allow_key_changes=False,
        max_rhythmic_subdivision=4, allow_syncopation=False, max_tempo=100,
    ),
    AbilityLevel.ELEMENTARY: SimplificationProfile(
        max_simultaneous_notes=2, max_interval_semitones=7,
        allow_accidentals=True, allow_key_changes=False,
        max_rhythmic_subdivision=8, allow_syncopation=False, max_tempo=120,
    ),
    AbilityLevel.INTERMEDIATE: SimplificationProfile(
        max_simultaneous_notes=3, max_interval_semitones=12,
        allow_accidentals=True, allow_key_changes=True,
        max_rhythmic_subdivision=8, allow_syncopation=True, max_tempo=140,
    ),
    AbilityLevel.ADVANCED: SimplificationProfile(
        max_simultaneous_notes=4, max_interval_semitones=24,
        allow_accidentals=True, allow_key_changes=True,
        max_rhythmic_subdivision=16, allow_syncopation=True, max_tempo=180,
    ),
    AbilityLevel.PROFESSIONAL: SimplificationProfile(
        max_simultaneous_notes=10, max_interval_semitones=48,
        allow_accidentals=True, allow_key_changes=True,
        max_rhythmic_subdivision=32, allow_syncopation=True, max_tempo=300,
    ),
}


@dataclass
class NoteEvent:
    """A single note event."""
    pitch: int  # MIDI
    onset_beat: float
    duration_beats: float
    velocity: int = 80


def difficulty_score(events: list[NoteEvent], profile: SimplificationProfile) -> float:
    """Score difficulty of a passage against a profile (0=easy, 1=hardest)."""
    if not events:
        return 0.0
    penalties = 0.0
    total_checks = 0

    # Check simultaneous notes
    onsets: dict[float, list[NoteEvent]] = {}
    for e in events:
        onsets.setdefault(e.onset_beat, []).append(e)
    for group in onsets.values():
        total_checks += 1
        if len(group) > profile.max_simultaneous_notes:
            penalties += 1.0

    # Check intervals
    sorted_events = sorted(events, key=lambda e: e.onset_beat)
    for i in range(1, len(sorted_events)):
        total_checks += 1
        interval = abs(sorted_events[i].pitch - sorted_events[i - 1].pitch)
        if interval > profile.max_interval_semitones:
            penalties += 1.0

    # Check rhythmic subdivision
    for e in events:
        total_checks += 1
        beat_frac = e.duration_beats
        subdivision = round(1.0 / beat_frac) if beat_frac > 0 else 1
        if subdivision > profile.max_rhythmic_subdivision:
            penalties += 1.0

    if total_checks == 0:
        return 0.0
    return min(1.0, penalties / total_checks)


def simplify_melody(
    events: list[NoteEvent],
    target_level: AbilityLevel,
) -> list[NoteEvent]:
    """Simplify a melody to fit the target ability level."""
    profile = LEVEL_PROFILES[target_level]
    result: list[NoteEvent] = []

    for i, e in enumerate(events):
        new_pitch = e.pitch
        new_dur = e.duration_beats

        # Clamp interval from previous note
        if result:
            interval = new_pitch - result[-1].pitch
            if abs(interval) > profile.max_interval_semitones:
                direction = 1 if interval > 0 else -1
                new_pitch = result[-1].pitch + direction * profile.max_interval_semitones

        # Remove accidentals if not allowed
        if not profile.allow_accidentals:
            pc = new_pitch % 12
            if pc in (1, 3, 6, 8, 10):  # black keys
                new_pitch += 1  # snap up to white key

        # Simplify rhythmic subdivision
        min_dur = 1.0 / profile.max_rhythmic_subdivision
        if new_dur < min_dur:
            new_dur = min_dur

        result.append(NoteEvent(
            pitch=new_pitch,
            onset_beat=e.onset_beat,
            duration_beats=new_dur,
            velocity=e.velocity,
        ))

    return result


def reduce_polyphony(
    events: list[NoteEvent],
    max_voices: int,
) -> list[NoteEvent]:
    """Reduce polyphony by keeping the top N voices at each onset."""
    onsets: dict[float, list[NoteEvent]] = {}
    for e in events:
        onsets.setdefault(e.onset_beat, []).append(e)

    result: list[NoteEvent] = []
    for onset in sorted(onsets.keys()):
        group = sorted(onsets[onset], key=lambda e: e.pitch, reverse=True)
        result.extend(group[:max_voices])
    return result


def suggest_level(events: list[NoteEvent]) -> AbilityLevel:
    """Suggest the appropriate ability level for a passage."""
    for level in AbilityLevel:
        profile = LEVEL_PROFILES[level]
        score = difficulty_score(events, profile)
        if score < 0.2:
            return level
    return AbilityLevel.PROFESSIONAL


def tempo_adjustment(original_bpm: float, target_level: AbilityLevel) -> float:
    """Adjusted tempo for the target level."""
    profile = LEVEL_PROFILES[target_level]
    return min(original_bpm, float(profile.max_tempo))
