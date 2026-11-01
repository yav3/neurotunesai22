"""Modulation analysis: key distance, pivot chords, chromatic mediants,
common-tone modulation scoring, and enharmonic reinterpretation.

Keys are represented as (root_pc, mode) where root_pc ∈ 0..11 and
mode ∈ {"major", "minor"}.  root_pc follows: C=0, C#=1, … B=11.
"""
from __future__ import annotations

from dataclasses import dataclass

# Circle of fifths position: C=0, G=1, D=2, … F=-1/11, Bb=-2/10, etc.
_COF_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
_COF_POS = {pc: i for i, pc in enumerate(_COF_ORDER)}

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


def _scale_pcs(root: int, mode: str) -> frozenset[int]:
    template = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    return frozenset((root + s) % 12 for s in template)


# ── Key distance ─────────────────────────────────────────────────────

def circle_of_fifths_distance(key_a: tuple[int, str], key_b: tuple[int, str]) -> int:
    """Minimum steps around the circle of fifths between two keys.

    Distance is computed on root pitch-class only; mode adds +1 if the
    modes differ (parallel relationship costs 1, relative relationship
    costs 0 if roots match the relative pair).
    """
    pa = _COF_POS[key_a[0]]
    pb = _COF_POS[key_b[0]]
    raw = abs(pa - pb)
    cof_dist = min(raw, 12 - raw)
    mode_penalty = 0 if key_a[1] == key_b[1] else 1
    return cof_dist + mode_penalty


def relative_key(root: int, mode: str) -> tuple[int, str]:
    """Return the relative major/minor of a key."""
    if mode == "major":
        return ((root + 9) % 12, "minor")  # relative minor is down a m3
    return ((root + 3) % 12, "major")  # relative major is up a m3


def parallel_key(root: int, mode: str) -> tuple[int, str]:
    """Return the parallel major/minor (same root, opposite mode)."""
    return (root, "minor" if mode == "major" else "major")


def key_distance(key_a: tuple[int, str], key_b: tuple[int, str]) -> int:
    """Combined distance: circle-of-fifths steps + mode penalty.

    Relative keys have distance 0; parallel keys have distance 1;
    closely related keys 1–2; remote keys up to 7.
    """
    # Relative key shortcut
    if relative_key(*key_a) == key_b:
        return 0
    return circle_of_fifths_distance(key_a, key_b)


# ── Pivot chords ─────────────────────────────────────────────────────

TRIAD_QUALITY = {
    (4, 3): "major",
    (3, 4): "minor",
    (3, 3): "diminished",
    (4, 4): "augmented",
}


def _diatonic_triads(root: int, mode: str) -> list[tuple[frozenset[int], str]]:
    """Build the seven diatonic triads of a key.

    Returns list of (pitch-class set, Roman-numeral label).
    """
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    roman = (["I", "ii", "iii", "IV", "V", "vi", "vii°"]
             if mode == "major"
             else ["i", "ii°", "III", "iv", "v", "VI", "VII"])
    triads = []
    for deg in range(7):
        pcs = frozenset(
            (root + scale[deg] + scale[(deg + j) % 7] - scale[deg]) % 12
            if j == 0
            else (root + scale[(deg + j) % 7] + (12 if (deg + j) >= 7 else 0)) % 12
            for j in range(3)
        )
        # Simpler: just build by scale index
        notes = [(root + scale[(deg + k) % 7]) % 12 for k in range(3)]
        pcs = frozenset(notes)
        triads.append((pcs, roman[deg]))
    return triads


def pivot_chords(
    key_a: tuple[int, str],
    key_b: tuple[int, str],
) -> list[tuple[frozenset[int], str, str]]:
    """Find chords common to both keys.

    Returns list of (pitch-class set, label_in_key_a, label_in_key_b).
    """
    triads_a = _diatonic_triads(key_a[0], key_a[1])
    triads_b = _diatonic_triads(key_b[0], key_b[1])
    result = []
    for pcs_a, label_a in triads_a:
        for pcs_b, label_b in triads_b:
            if pcs_a == pcs_b:
                result.append((pcs_a, label_a, label_b))
    return result


# ── Common-tone modulation ───────────────────────────────────────────

def common_tone_count(key_a: tuple[int, str], key_b: tuple[int, str]) -> int:
    """Number of pitch classes shared between two key scales."""
    sa = _scale_pcs(key_a[0], key_a[1])
    sb = _scale_pcs(key_b[0], key_b[1])
    return len(sa & sb)


def common_tone_modulation_score(key_a: tuple[int, str], key_b: tuple[int, str]) -> float:
    """Score 0–1 for how smooth a common-tone modulation would be.

    7/7 shared tones → 1.0 (enharmonic reinterpretation or same key);
    0/7 → 0.0 (maximally remote).
    """
    return common_tone_count(key_a, key_b) / 7.0


# ── Chromatic mediants ───────────────────────────────────────────────

@dataclass(frozen=True)
class ChromaticMediant:
    root_interval: int
    shared_tones: int
    kind: str  # "diatonic", "chromatic", "doubly_chromatic"


def chromatic_mediant_relationship(
    key_a: tuple[int, str],
    key_b: tuple[int, str],
) -> ChromaticMediant | None:
    """Classify the chromatic-mediant relationship (if any) between two
    major keys.

    Mediant = root distance of 3 or 4 semitones (M3/m3 up or down).
    """
    if key_a[1] != "major" or key_b[1] != "major":
        return None
    diff = (key_b[0] - key_a[0]) % 12
    if diff not in {3, 4, 8, 9}:
        return None
    shared = common_tone_count(key_a, key_b)
    if shared >= 5:
        kind = "diatonic"
    elif shared >= 3:
        kind = "chromatic"
    else:
        kind = "doubly_chromatic"
    return ChromaticMediant(root_interval=diff, shared_tones=shared, kind=kind)


# ── Enharmonic reinterpretation ──────────────────────────────────────

@dataclass(frozen=True)
class EnharmonicReinterpretation:
    pcs: frozenset[int]
    interpretation_a: str
    interpretation_b: str


def detect_enharmonic_reinterpretation(
    chord_pcs: frozenset[int],
    key_a: tuple[int, str],
    key_b: tuple[int, str],
) -> EnharmonicReinterpretation | None:
    """Detect whether *chord_pcs* can be read differently in two keys.

    Classic case: German augmented sixth {Ab, C, Eb, F#} in key_a
    reinterpreted as a dominant seventh {Ab, C, Eb, Gb} in key_b.
    """
    triads_a = _diatonic_triads(key_a[0], key_a[1])
    triads_b = _diatonic_triads(key_b[0], key_b[1])

    label_a = None
    for pcs, label in triads_a:
        if pcs == chord_pcs:
            label_a = label
            break

    label_b = None
    for pcs, label in triads_b:
        if pcs == chord_pcs:
            label_b = label
            break

    if label_a and label_b and label_a != label_b:
        return EnharmonicReinterpretation(chord_pcs, label_a, label_b)

    # Check augmented-sixth / dominant-seventh equivalence
    if len(chord_pcs) == 4:
        pcs_list = sorted(chord_pcs)
        intervals = [(pcs_list[(i + 1) % 4] - pcs_list[i]) % 12 for i in range(4)]
        if 6 in intervals:
            return EnharmonicReinterpretation(
                chord_pcs, "Aug6", "V7 (enharmonic)")

    return None


# ── Modulation plan ──────────────────────────────────────────────────

@dataclass
class ModulationPlan:
    source: tuple[int, str]
    target: tuple[int, str]
    distance: int
    pivot_count: int
    common_tones: int
    chromatic_mediant: ChromaticMediant | None
    smoothness: float


def analyze_modulation(
    source: tuple[int, str],
    target: tuple[int, str],
) -> ModulationPlan:
    """Comprehensive modulation analysis between two keys."""
    dist = key_distance(source, target)
    pivots = pivot_chords(source, target)
    ct = common_tone_count(source, target)
    cm = chromatic_mediant_relationship(source, target)
    smoothness = common_tone_modulation_score(source, target)
    return ModulationPlan(
        source=source,
        target=target,
        distance=dist,
        pivot_count=len(pivots),
        common_tones=ct,
        chromatic_mediant=cm,
        smoothness=smoothness,
    )
