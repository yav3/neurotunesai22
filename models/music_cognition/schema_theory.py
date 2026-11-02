"""Gjerdingen (2007) galant schemata — pattern recognition in tonal music.

Encodes the most common melodic-harmonic schemata of 18th-century
European music (Romanesca, Prinner, Fonte, Monte, etc.) and provides
pattern matching against a melody/bass pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np


# ── Schema definitions ───────────────────────────────────────────────────

@dataclass(frozen=True)
class SchemaTemplate:
    name: str
    melody_degrees: Tuple[int, ...]     # scale degrees (1-indexed, mod 7)
    bass_degrees: Tuple[int, ...]
    typical_length_beats: int
    harmonic_labels: Tuple[str, ...]    # Roman numerals
    description: str


ROMANESCA = SchemaTemplate(
    name="Romanesca",
    melody_degrees=(8, 7, 6, 5),
    bass_degrees=(1, 7, 6, 5),
    typical_length_beats=4,
    harmonic_labels=("I", "V6", "vi", "V"),
    description="Descending 8-7-6-5 soprano over 1-7-6-5 bass",
)

PRINNER = SchemaTemplate(
    name="Prinner",
    melody_degrees=(6, 5, 4, 3),
    bass_degrees=(4, 3, 2, 1),
    typical_length_beats=4,
    harmonic_labels=("IV", "I6", "ii6", "I"),
    description="Riposte to opening gambit; descending 6-5-4-3 over 4-3-2-1",
)

FONTE = SchemaTemplate(
    name="Fonte",
    melody_degrees=(4, 3, 3, 2),
    bass_degrees=(6, 5, 5, 4),
    typical_length_beats=4,
    harmonic_labels=("ii", "V/V", "I6/4", "V"),
    description="Sequencing schema, often at start of second half",
)

MONTE = SchemaTemplate(
    name="Monte",
    melody_degrees=(2, 3, 4, 5),
    bass_degrees=(7, 1, 2, 3),
    typical_length_beats=4,
    harmonic_labels=("V", "I", "V/V", "V"),
    description="Rising sequence, opposite of Fonte",
)

DO_RE_MI = SchemaTemplate(
    name="Do-Re-Mi",
    melody_degrees=(1, 2, 3),
    bass_degrees=(1, 7, 1),
    typical_length_beats=3,
    harmonic_labels=("I", "V", "I"),
    description="Opening gambit: stepwise ascent 1-2-3",
)

SOL_FA_MI = SchemaTemplate(
    name="Sol-Fa-Mi",
    melody_degrees=(5, 4, 3),
    bass_degrees=(1, 7, 1),
    typical_length_beats=3,
    harmonic_labels=("I", "viio6", "I"),
    description="Cadential: 5-4-3 descent to mediant",
)

QUIESCENZA = SchemaTemplate(
    name="Quiescenza",
    melody_degrees=(1, 7, 2, 1),
    bass_degrees=(1, 1, 1, 1),
    typical_length_beats=4,
    harmonic_labels=("I", "V4/2", "V6/5", "I"),
    description="Pedal-point schema with neighbour tones",
)

CADENZA_DOPPIA = SchemaTemplate(
    name="Cadenza Doppia",
    melody_degrees=(2, 1, 7, 1),
    bass_degrees=(4, 3, 5, 1),
    typical_length_beats=4,
    harmonic_labels=("ii", "I6", "V", "I"),
    description="Double cadence: pre-dominant → tonic6 → dominant → tonic",
)

INDUGIO = SchemaTemplate(
    name="Indugio",
    melody_degrees=(4, 4, 3),
    bass_degrees=(2, 2, 3),
    typical_length_beats=3,
    harmonic_labels=("ii", "V/V", "I6"),
    description="Lingering on predominant before resolution",
)

FENAROLI = SchemaTemplate(
    name="Fenaroli",
    melody_degrees=(7, 1, 2, 3),
    bass_degrees=(4, 3, 2, 1),
    typical_length_beats=4,
    harmonic_labels=("viio6", "I", "V4/3", "I"),
    description="Rising soprano 7-1-2-3 over descending bass 4-3-2-1",
)


SCHEMA_CATALOG: Dict[str, SchemaTemplate] = {
    "romanesca": ROMANESCA, "prinner": PRINNER, "fonte": FONTE,
    "monte": MONTE, "do_re_mi": DO_RE_MI, "sol_fa_mi": SOL_FA_MI,
    "quiescenza": QUIESCENZA, "cadenza_doppia": CADENZA_DOPPIA,
    "indugio": INDUGIO, "fenaroli": FENAROLI,
}


# ── Pitch → scale degree ────────────────────────────────────────────────

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)


def pitch_to_degree(midi_pitch: int, key_root: int = 0) -> int:
    """Convert MIDI pitch to 1-indexed scale degree (nearest major-scale degree)."""
    pc = (midi_pitch - key_root) % 12
    best_deg = 1
    best_dist = 12
    for i, s in enumerate(MAJOR_SCALE):
        d = min(abs(pc - s), 12 - abs(pc - s))
        if d < best_dist:
            best_dist = d
            best_deg = i + 1
    return best_deg


def pitches_to_degrees(pitches: Sequence[int], key_root: int = 0) -> List[int]:
    return [pitch_to_degree(p, key_root) for p in pitches]


# ── Pattern matching ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class SchemaMatch:
    schema_name: str
    start_index: int
    length: int
    melody_match_score: float
    bass_match_score: float
    combined_score: float


def _degree_similarity(seq: Sequence[int], template: Tuple[int, ...]) -> float:
    """Proportion of matching scale degrees (allowing octave equivalence)."""
    n = min(len(seq), len(template))
    if n == 0:
        return 0.0
    matches = sum(1 for a, b in zip(seq[:n], template[:n]) if a % 7 == b % 7 or a == b)
    return matches / n


def match_schema(
    melody_degrees: Sequence[int],
    bass_degrees: Sequence[int],
    schema: SchemaTemplate,
    start: int = 0,
) -> Optional[SchemaMatch]:
    """Check if schema matches at a given start position."""
    length = len(schema.melody_degrees)
    if start + length > len(melody_degrees) or start + length > len(bass_degrees):
        return None

    mel_seg = melody_degrees[start : start + length]
    bass_seg = bass_degrees[start : start + length]

    mel_score = _degree_similarity(mel_seg, schema.melody_degrees)
    bass_score = _degree_similarity(bass_seg, schema.bass_degrees)
    combined = 0.5 * mel_score + 0.5 * bass_score

    if combined < 0.5:
        return None

    return SchemaMatch(
        schema_name=schema.name,
        start_index=start,
        length=length,
        melody_match_score=mel_score,
        bass_match_score=bass_score,
        combined_score=combined,
    )


def find_schemata(
    melody_pitches: Sequence[int],
    bass_pitches: Sequence[int],
    key_root: int = 0,
    threshold: float = 0.6,
) -> List[SchemaMatch]:
    """Scan melody+bass for all galant schemata occurrences."""
    mel_deg = pitches_to_degrees(melody_pitches, key_root)
    bass_deg = pitches_to_degrees(bass_pitches, key_root)
    matches: List[SchemaMatch] = []

    for schema in SCHEMA_CATALOG.values():
        length = len(schema.melody_degrees)
        for start in range(max(len(mel_deg), len(bass_deg)) - length + 1):
            m = match_schema(mel_deg, bass_deg, schema, start)
            if m is not None and m.combined_score >= threshold:
                matches.append(m)

    matches.sort(key=lambda m: m.combined_score, reverse=True)
    return matches


# ── Schema generation ────────────────────────────────────────────────────

def generate_schema_melody(
    schema: SchemaTemplate,
    key_root: int = 0,
    octave: int = 5,
) -> List[int]:
    """Generate MIDI pitches for a schema's melody in a given key."""
    base = key_root + 12 * octave
    return [base + MAJOR_SCALE[(d - 1) % 7] + 12 * ((d - 1) // 7) for d in schema.melody_degrees]


def generate_schema_bass(
    schema: SchemaTemplate,
    key_root: int = 0,
    octave: int = 3,
) -> List[int]:
    """Generate MIDI pitches for a schema's bass in a given key."""
    base = key_root + 12 * octave
    return [base + MAJOR_SCALE[(d - 1) % 7] + 12 * ((d - 1) // 7) for d in schema.bass_degrees]


# ── Schema complexity ────────────────────────────────────────────────────

def schema_complexity(schema: SchemaTemplate) -> float:
    """Estimate cognitive complexity: melodic range × harmonic variety."""
    mel_range = max(schema.melody_degrees) - min(schema.melody_degrees)
    bass_range = max(schema.bass_degrees) - min(schema.bass_degrees)
    harm_variety = len(set(schema.harmonic_labels))
    return float(mel_range + bass_range) * harm_variety / len(schema.melody_degrees)


def schema_catalog_summary() -> Dict[str, float]:
    """Return complexity scores for all catalogued schemata."""
    return {name: schema_complexity(s) for name, s in SCHEMA_CATALOG.items()}
