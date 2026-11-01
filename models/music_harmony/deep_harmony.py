"""Deep harmonic analysis: Schenkerian reduction, tonal tension, chromatic harmony.

Implements analytical frameworks beyond Roman-numeral labeling:
- Schenkerian analysis: prolongation, Ursatz detection, structural levels
  (Schenker, 1935; Forte & Gilbert, 1982)
- Tonal Pitch Space model (Lerdahl, 2001) — quantified tonal tension
- Psychoacoustic roughness / sensory dissonance (Sethares, 1993; Plomp & Levelt, 1965)
- Chromatic harmony: omnibus, enharmonic pivot, equal division (Cohn, 2012)
- Cadential weight and closure strength (Temperley, 2004)
- Harmonic expectation via information content (Pearce, 2005)

Pure numpy. Forward-pass / analysis only.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from enum import Enum, auto


# ── pitch utilities ────────────────────────────────────────────────────

def midi_to_pc(midi: int) -> int:
    return midi % 12

def midi_to_freq(midi: int, a4: float = 440.0) -> float:
    return a4 * 2.0 ** ((midi - 69) / 12.0)

PC_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


# ══════════════════════════════════════════════════════════════════════
# § 1  Psychoacoustic roughness — Plomp & Levelt (1965), Sethares (1993)
# ══════════════════════════════════════════════════════════════════════

def plomp_levelt_curve(freq_diff: float, f_mean: float) -> float:
    """Single-pair dissonance (Plomp & Levelt, 1965).

    Parameterised by Sethares (1993): d(x) = e^{-ax} - e^{-bx}
    where x = freq_diff / (s * f_mean + 0.001), s ≈ 0.24.
    """
    s = 0.24
    x = np.abs(freq_diff) / (s * f_mean + 0.001)
    a, b = 3.5, 5.75
    return np.exp(-a * x) - np.exp(-b * x)


def dyad_roughness(f1: float, f2: float, a1: float = 1.0, a2: float = 1.0) -> float:
    """Roughness between two partials with amplitudes (Sethares, 1993)."""
    return a1 * a2 * max(0.0, plomp_levelt_curve(abs(f2 - f1), 0.5 * (f1 + f2)))


def spectrum_roughness(
    freqs: np.ndarray, amps: np.ndarray
) -> float:
    """Total roughness of a complex spectrum — sum of all dyad interactions.

    Sethares (1993) eq. 3: R = Σ_{i<j} a_i·a_j · d(f_j - f_i).
    """
    n = len(freqs)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += dyad_roughness(freqs[i], freqs[j], amps[i], amps[j])
    return total


def chord_roughness_harmonic(
    midis: Sequence[int], n_partials: int = 6, rolloff: float = 0.7
) -> float:
    """Roughness of a chord assuming harmonic timbres with exponential rolloff.

    Each note contributes partials at integer multiples with amplitude rolloff^k.
    """
    freqs, amps = [], []
    for m in midis:
        f0 = midi_to_freq(m)
        for k in range(1, n_partials + 1):
            freqs.append(f0 * k)
            amps.append(rolloff ** (k - 1))
    return spectrum_roughness(np.array(freqs), np.array(amps))


def sensory_consonance_curve(
    root_midi: int = 60, n_points: int = 1200, max_interval_cents: int = 1200,
    n_partials: int = 6, rolloff: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sensory consonance/dissonance curve for interval above root (Sethares, 1993).

    Returns (cents_array, roughness_array). Minima correspond to
    consonant intervals (octave, fifth, fourth, etc.).
    """
    cents = np.linspace(0, max_interval_cents, n_points)
    f0 = midi_to_freq(root_midi)
    roughnesses = np.zeros(n_points)
    for i, c in enumerate(cents):
        f1 = f0 * 2.0 ** (c / 1200.0)
        freqs = []
        amps = []
        for k in range(1, n_partials + 1):
            freqs.extend([f0 * k, f1 * k])
            amps.extend([rolloff ** (k - 1)] * 2)
        roughnesses[i] = spectrum_roughness(np.array(freqs), np.array(amps))
    return cents, roughnesses


# ══════════════════════════════════════════════════════════════════════
# § 2  Tonal Pitch Space — Lerdahl (2001)
# ══════════════════════════════════════════════════════════════════════

class TPSLevel(Enum):
    """Levels of the Tonal Pitch Space basic space (Lerdahl, 2001, Ch. 2)."""
    CHROMATIC = auto()
    DIATONIC = auto()
    TRIADIC = auto()
    FIFTH = auto()
    ROOT = auto()


# Major-key basic space: which pitch classes are active at each level.
# Row = pc (0–11), column = level (root, fifth, triadic, diatonic, chromatic).
MAJOR_BASIC_SPACE = np.zeros((12, 5), dtype=int)
MAJOR_BASIC_SPACE[:, 4] = 1  # chromatic: all 12
for pc in [0, 2, 4, 5, 7, 9, 11]:
    MAJOR_BASIC_SPACE[pc, 3] = 1  # diatonic
for pc in [0, 4, 7]:
    MAJOR_BASIC_SPACE[pc, 2] = 1  # triadic (I = C-E-G)
MAJOR_BASIC_SPACE[7, 1] = 1  # fifth
MAJOR_BASIC_SPACE[0, 0] = 1  # root

MINOR_BASIC_SPACE = np.zeros((12, 5), dtype=int)
MINOR_BASIC_SPACE[:, 4] = 1
for pc in [0, 2, 3, 5, 7, 8, 10]:
    MINOR_BASIC_SPACE[pc, 3] = 1  # natural minor diatonic
for pc in [0, 3, 7]:
    MINOR_BASIC_SPACE[pc, 2] = 1  # triadic (i = C-Eb-G)
MINOR_BASIC_SPACE[7, 1] = 1
MINOR_BASIC_SPACE[0, 0] = 1


def _rotate_basic_space(space: np.ndarray, semitones: int) -> np.ndarray:
    return np.roll(space, semitones, axis=0)


def tps_chord_distance(
    chord1_pcs: Sequence[int], root1: int, key1: int, mode1: str,
    chord2_pcs: Sequence[int], root2: int, key2: int, mode2: str,
) -> float:
    """Lerdahl (2001) chord distance δ(x, y).

    δ = i + j + k where:
      i = distance on the circle of fifths between keys
      j = distance on the circle of fifths between chord roots (within key context)
      k = number of non-common tones between the two basic-space representations

    Lower-level simplification: i from key roots on CoF, j from chord
    roots on CoF, k from pitch-class set symmetric difference.
    """
    cof = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    cof_pos = {pc: idx for idx, pc in enumerate(cof)}

    i = min(abs(cof_pos[key1 % 12] - cof_pos[key2 % 12]),
            12 - abs(cof_pos[key1 % 12] - cof_pos[key2 % 12]))

    j = min(abs(cof_pos[root1 % 12] - cof_pos[root2 % 12]),
            12 - abs(cof_pos[root1 % 12] - cof_pos[root2 % 12]))

    s1 = set(pc % 12 for pc in chord1_pcs)
    s2 = set(pc % 12 for pc in chord2_pcs)
    k = len(s1.symmetric_difference(s2))

    return float(i + j + k)


def tps_sequential_tension(
    chords: List[Tuple[List[int], int]],
    key: int = 0, mode: str = 'major',
) -> np.ndarray:
    """Sequential tension curve: δ between successive chords (Lerdahl, 2001, Ch. 4).

    Each chord is (pitch_classes, root).  Returns array of length len(chords)-1.
    """
    tensions = []
    for idx in range(len(chords) - 1):
        pcs1, r1 = chords[idx]
        pcs2, r2 = chords[idx + 1]
        d = tps_chord_distance(pcs1, r1, key, mode, pcs2, r2, key, mode)
        tensions.append(d)
    return np.array(tensions)


def tps_hierarchical_tension(
    chords: List[Tuple[List[int], int]],
    domination: List[int],
    key: int = 0, mode: str = 'major',
) -> np.ndarray:
    """Hierarchical tension: distance from each chord to its dominating event.

    `domination[i]` is the index of the event that dominates chord i in the
    prolongational tree (Lerdahl, 2001, Ch. 4).  Root event dominates itself.
    """
    n = len(chords)
    tensions = np.zeros(n)
    for i in range(n):
        parent = domination[i]
        if parent == i:
            tensions[i] = 0.0
        else:
            pcs_i, r_i = chords[i]
            pcs_p, r_p = chords[parent]
            tensions[i] = tps_chord_distance(pcs_i, r_i, key, mode,
                                             pcs_p, r_p, key, mode)
    return tensions


@dataclass
class TPSAnalysis:
    sequential_tension: np.ndarray
    hierarchical_tension: np.ndarray
    attraction_values: np.ndarray
    total_tension: np.ndarray


# ══════════════════════════════════════════════════════════════════════
# § 3  Tonal attraction — Lerdahl (2001, Ch. 5)
# ══════════════════════════════════════════════════════════════════════

STABILITY_WEIGHTS = {0: 4.0, 7: 3.5, 4: 3.0, 3: 3.0, 5: 2.5, 9: 2.5,
                     2: 2.0, 10: 2.0, 11: 2.0, 1: 1.0, 6: 1.0, 8: 1.0}


def _anchoring_strength(pc_from: int, pc_to: int, key: int = 0) -> float:
    """Lerdahl (2001) melodic attraction α(p1 → p2).

    α = s2 / (n^2) where s2 = stability of destination, n = semitone distance.
    """
    dist = min(abs(pc_from - pc_to), 12 - abs(pc_from - pc_to))
    if dist == 0:
        return 0.0
    s_to = STABILITY_WEIGHTS.get((pc_to - key) % 12, 1.0)
    return s_to / (dist ** 2)


def melodic_attraction(
    melody_pcs: Sequence[int], key: int = 0
) -> np.ndarray:
    """Compute Lerdahl melodic-attraction values for a pitch-class sequence."""
    n = len(melody_pcs)
    attractions = np.zeros(max(0, n - 1))
    for i in range(n - 1):
        attractions[i] = _anchoring_strength(melody_pcs[i], melody_pcs[i + 1], key)
    return attractions


def harmonic_attraction(
    chord1_pcs: Sequence[int], chord2_pcs: Sequence[int], key: int = 0
) -> float:
    """Harmonic attraction between two chords (Lerdahl, 2001, §5.3).

    Sum of voice-leading attractions (nearest-pair pairing by semitone distance).
    """
    total = 0.0
    for p1 in chord1_pcs:
        best = 0.0
        for p2 in chord2_pcs:
            a = _anchoring_strength(p1 % 12, p2 % 12, key)
            if a > best:
                best = a
        total += best
    return total


# ══════════════════════════════════════════════════════════════════════
# § 4  Schenkerian analysis — prolongation & reduction
# ══════════════════════════════════════════════════════════════════════

class ProlongationType(Enum):
    STRONG_PROLONGATION = auto()   # same harmony repeated
    WEAK_PROLONGATION = auto()     # same root, different inversion/voicing
    PROGRESSION = auto()           # different harmony — tensing or relaxing
    NEIGHBOR = auto()              # departure and return to same harmony
    PASSING = auto()               # stepwise motion connecting two structural tones


@dataclass
class ReductionNode:
    """A node in a Schenkerian prolongation tree."""
    event_idx: int
    pitch_class: int
    midi: int
    metric_weight: float
    is_structural: bool
    prolongation: Optional[ProlongationType] = None
    parent_idx: Optional[int] = None
    children: List[int] = field(default_factory=list)
    level: int = 0  # 0 = foreground, higher = deeper structural level


def _metric_weight(position: float, beats_per_bar: int = 4) -> float:
    """Assign metric weight based on position within a bar (Lerdahl & Jackendoff, 1983)."""
    beat = position % beats_per_bar
    if beat == 0:
        return 4.0
    if beats_per_bar == 4 and beat == 2:
        return 3.0
    if beat == int(beat):
        return 2.0
    return 1.0


def classify_prolongation(
    pc_from: int, pc_mid: int, pc_to: int,
) -> ProlongationType:
    """Classify the prolongation type of a middle event between two structural tones."""
    if pc_mid == pc_from == pc_to:
        return ProlongationType.STRONG_PROLONGATION
    if pc_mid == pc_from or pc_mid == pc_to:
        return ProlongationType.WEAK_PROLONGATION
    if pc_from == pc_to and pc_mid != pc_from:
        return ProlongationType.NEIGHBOR
    d1 = min(abs(pc_mid - pc_from), 12 - abs(pc_mid - pc_from))
    d2 = min(abs(pc_to - pc_mid), 12 - abs(pc_to - pc_mid))
    if d1 <= 2 and d2 <= 2:
        return ProlongationType.PASSING
    return ProlongationType.PROGRESSION


def detect_ursatz(
    melody_pcs: Sequence[int],
    bass_pcs: Sequence[int],
    key: int = 0,
    mode: str = 'major',
) -> Dict:
    """Detect Ursatz (fundamental structure) — Schenker's Urlinie + Bassbrechung.

    Urlinie: stepwise descent to ^1 (scale degree 1) from ^3, ^5, or ^8.
    Bassbrechung: I–V–I arpeggiation in the bass.

    Returns dict with urlinie_degrees, bass_arpeggiation, and ursatz_found.
    """
    if mode == 'major':
        scale = [0, 2, 4, 5, 7, 9, 11]
    else:
        scale = [0, 2, 3, 5, 7, 8, 10]

    def pc_to_degree(pc: int) -> Optional[int]:
        rel = (pc - key) % 12
        if rel in scale:
            return scale.index(rel) + 1
        return None

    melody_degrees = [pc_to_degree(pc) for pc in melody_pcs]
    bass_degrees = [pc_to_degree(pc) for pc in bass_pcs]

    urlinie_found = None
    for start_deg in [8, 5, 3]:
        actual_start = start_deg if start_deg != 8 else 1
        if start_deg == 8:
            target_seq = [8, 7, 6, 5, 4, 3, 2, 1]
        elif start_deg == 5:
            target_seq = [5, 4, 3, 2, 1]
        else:
            target_seq = [3, 2, 1]

        ptr = 0
        for d in melody_degrees:
            if d is not None and ptr < len(target_seq):
                if d == target_seq[ptr] or (target_seq[ptr] == 8 and d == 1):
                    ptr += 1
        if ptr >= len(target_seq):
            urlinie_found = target_seq
            break

    bass_ivvi = False
    state = 0
    for d in bass_degrees:
        if d is None:
            continue
        if state == 0 and d == 1:
            state = 1
        elif state == 1 and d == 5:
            state = 2
        elif state == 2 and d == 1:
            bass_ivvi = True
            break

    return {
        'urlinie': urlinie_found,
        'bassbrechung': bass_ivvi,
        'ursatz_found': urlinie_found is not None and bass_ivvi,
        'melody_degrees': melody_degrees,
        'bass_degrees': bass_degrees,
    }


def schenkerian_reduce(
    midis: np.ndarray,
    metric_positions: np.ndarray,
    beats_per_bar: int = 4,
    levels: int = 3,
) -> List[List[ReductionNode]]:
    """Iterative time-span reduction inspired by Lerdahl & Jackendoff (1983) GTTM.

    At each level, the least structurally important events are removed.
    Importance = metric_weight + stability_weight + registral extremity bonus.

    Returns list of levels, each a list of ReductionNode.
    """
    n = len(midis)
    nodes = []
    for i in range(n):
        mw = _metric_weight(metric_positions[i], beats_per_bar)
        pc = midis[i] % 12
        stab = STABILITY_WEIGHTS.get(pc, 1.0)
        nodes.append(ReductionNode(
            event_idx=i, pitch_class=pc, midi=int(midis[i]),
            metric_weight=mw, is_structural=True, level=0,
        ))

    all_levels = [list(nodes)]
    current = list(nodes)

    for lv in range(1, levels):
        scores = []
        for nd in current:
            reg_bonus = 0.5 * abs(nd.midi - 60) / 24.0
            score = nd.metric_weight + STABILITY_WEIGHTS.get(nd.pitch_class, 1.0) + reg_bonus
            scores.append(score)

        threshold = np.percentile(scores, 30 * lv)
        reduced = []
        for nd, sc in zip(current, scores):
            if sc >= threshold:
                new_node = ReductionNode(
                    event_idx=nd.event_idx, pitch_class=nd.pitch_class,
                    midi=nd.midi, metric_weight=nd.metric_weight,
                    is_structural=True, level=lv,
                )
                reduced.append(new_node)

        if len(reduced) < 2:
            reduced = current[:2]

        all_levels.append(reduced)
        current = reduced

    return all_levels


# ══════════════════════════════════════════════════════════════════════
# § 5  Cadential analysis — strength and closure
# ══════════════════════════════════════════════════════════════════════

class CadenceStrength(Enum):
    PAC = auto()       # perfect authentic
    IAC = auto()       # imperfect authentic
    HC = auto()        # half
    DC = auto()        # deceptive
    PLAGAL = auto()
    PHRYGIAN = auto()
    NONE = auto()


@dataclass
class CadenceAnalysis:
    cadence_type: CadenceStrength
    strength: float         # 0..1 closure strength
    final_root: int
    penult_root: int
    key: int
    soprano_on_tonic: bool


def _root_from_pcs(pcs: Sequence[int]) -> int:
    """Heuristic root-finding: lowest pitch-class in the chord (simplified)."""
    return min(pcs) % 12


def analyze_cadence(
    penult_pcs: Sequence[int],
    final_pcs: Sequence[int],
    key: int = 0,
    mode: str = 'major',
    soprano_final: Optional[int] = None,
) -> CadenceAnalysis:
    """Classify and score a cadential gesture (Temperley, 2004; Caplin, 1998).

    Strength reflects metric placement, voice leading, and cadence type.
    """
    r_pen = _root_from_pcs(penult_pcs)
    r_fin = _root_from_pcs(final_pcs)
    deg_pen = (r_pen - key) % 12
    deg_fin = (r_fin - key) % 12

    soprano_on_1 = False
    if soprano_final is not None:
        soprano_on_1 = (soprano_final % 12) == (key % 12)

    ct = CadenceStrength.NONE
    strength = 0.0

    if deg_pen == 7 and deg_fin == 0:
        if soprano_on_1:
            ct = CadenceStrength.PAC
            strength = 1.0
        else:
            ct = CadenceStrength.IAC
            strength = 0.75
    elif deg_pen == 7:
        ct = CadenceStrength.DC
        strength = 0.3
    elif deg_fin == 7:
        ct = CadenceStrength.HC
        strength = 0.5
    elif deg_pen == 5 and deg_fin == 0:
        ct = CadenceStrength.PLAGAL
        strength = 0.6
    elif mode == 'minor' and deg_pen in (1, 3) and deg_fin == 7:
        ct = CadenceStrength.PHRYGIAN
        strength = 0.55

    return CadenceAnalysis(
        cadence_type=ct, strength=strength,
        final_root=r_fin, penult_root=r_pen,
        key=key, soprano_on_tonic=soprano_on_1,
    )


# ══════════════════════════════════════════════════════════════════════
# § 6  Chromatic harmony — mediants, omnibus, equal division
# ══════════════════════════════════════════════════════════════════════

def chromatic_mediant_type(
    root1: int, quality1: str, root2: int, quality2: str,
) -> str:
    """Classify chromatic mediant relationship (Cohn, 2012).

    Returns one of: 'diatonic_mediant', 'chromatic_mediant', 'doubly_chromatic',
    'not_mediant'.  Mediant = root interval of 3 or 4 semitones.
    """
    interval = (root2 - root1) % 12
    if interval not in (3, 4, 8, 9):
        return 'not_mediant'

    same_quality = (quality1 == quality2)
    if same_quality:
        return 'chromatic_mediant'

    q_set = {quality1, quality2}
    if q_set == {'major', 'minor'}:
        shared_pcs_1 = _triad_pcs(root1, quality1)
        shared_pcs_2 = _triad_pcs(root2, quality2)
        common = len(shared_pcs_1 & shared_pcs_2)
        if common == 0:
            return 'doubly_chromatic'
        return 'diatonic_mediant' if common >= 2 else 'chromatic_mediant'
    return 'chromatic_mediant'


def _triad_pcs(root: int, quality: str) -> set:
    if quality == 'major':
        return {root % 12, (root + 4) % 12, (root + 7) % 12}
    elif quality == 'minor':
        return {root % 12, (root + 3) % 12, (root + 7) % 12}
    elif quality == 'diminished':
        return {root % 12, (root + 3) % 12, (root + 6) % 12}
    elif quality == 'augmented':
        return {root % 12, (root + 4) % 12, (root + 8) % 12}
    return {root % 12}


def omnibus_progression(start_root: int = 0, n_steps: int = 8) -> List[Tuple[int, str]]:
    """Generate an omnibus (chromatic wedge) progression.

    Outer voices move in contrary chromatic motion while inner voices hold.
    Returns list of (root_pc, quality).
    """
    chords = []
    bass = start_root
    soprano_offset = 7
    for step in range(n_steps):
        if step % 2 == 0:
            bass_pc = (start_root + step // 2) % 12
        else:
            bass_pc = (start_root + (step + 1) // 2) % 12
        top_pc = (start_root + soprano_offset - step // 2) % 12
        middle = (start_root + 4) % 12
        interval_b_m = (middle - bass_pc) % 12
        quality = 'major' if interval_b_m in (3, 4) else 'minor' if interval_b_m in (3, 9) else 'diminished'
        chords.append((bass_pc, quality))
    return chords


def equal_division(interval: int, root: int = 0) -> List[int]:
    """Roots that equally divide the octave by the given interval.

    e.g. interval=4 → augmented triad roots, interval=3 → diminished 7th roots.
    """
    if 12 % interval != 0:
        pts = []
        pc = root % 12
        for _ in range(12):
            pts.append(pc)
            pc = (pc + interval) % 12
            if pc == root % 12:
                break
        return pts
    return [(root + k * interval) % 12 for k in range(12 // interval)]


def enharmonic_pivot_options(
    chord_pcs: Sequence[int], quality: str,
) -> List[Dict]:
    """Find enharmonic reinterpretation options for modulation.

    A diminished 7th chord can pivot to 4 keys; an augmented triad to 3;
    a German augmented 6th reinterprets as a dominant 7th.
    """
    pcs = set(pc % 12 for pc in chord_pcs)
    options = []

    if quality in ('diminished', 'dim7') and len(pcs) >= 3:
        for root_candidate in pcs:
            remaining = pcs - {root_candidate}
            options.append({
                'reinterpreted_root': root_candidate,
                'function': f'viio7 of {PC_NAMES[root_candidate]}',
                'target_key': (root_candidate + 1) % 12,
            })

    if quality == 'augmented' and len(pcs) == 3:
        for root_candidate in pcs:
            options.append({
                'reinterpreted_root': root_candidate,
                'function': f'V+ of {PC_NAMES[(root_candidate + 5) % 12]}',
                'target_key': (root_candidate + 5) % 12,
            })

    if quality in ('ger6', 'german_augmented_sixth'):
        options.append({
            'reinterpreted_root': min(pcs),
            'function': 'Ger+6 ↔ V7 enharmonic',
            'target_key': (min(pcs) + 5) % 12,
        })

    return options


# ══════════════════════════════════════════════════════════════════════
# § 7  Harmonic information content / expectation (Pearce, 2005)
# ══════════════════════════════════════════════════════════════════════

def chord_transition_matrix(
    progression: Sequence[int], n_states: int = 12,
) -> np.ndarray:
    """Learn first-order Markov transition matrix from root sequence.

    Returns (n_states, n_states) row-stochastic matrix.
    """
    counts = np.zeros((n_states, n_states))
    for i in range(len(progression) - 1):
        counts[progression[i] % n_states, progression[i + 1] % n_states] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def information_content(
    progression: Sequence[int],
    transition_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-event information content IC = -log2 P(x_t | x_{t-1}).

    High IC = surprising / low-probability transition.
    If no transition matrix provided, trains on the input sequence.
    """
    if transition_matrix is None:
        transition_matrix = chord_transition_matrix(progression)
    n = len(progression)
    ic = np.zeros(max(0, n - 1))
    for i in range(n - 1):
        s = progression[i] % 12
        t = progression[i + 1] % 12
        p = transition_matrix[s, t]
        ic[i] = -np.log2(p + 1e-12)
    return ic


def harmonic_entropy_rate(progression: Sequence[int]) -> float:
    """Entropy rate of the harmonic sequence (bits per chord).

    H_rate = - Σ_{i,j} π_i P(j|i) log2 P(j|i), where π is the stationary distribution.
    """
    T = chord_transition_matrix(progression)
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.abs(eigvecs[:, idx])
    pi = pi / (pi.sum() + 1e-12)

    H = 0.0
    for i in range(12):
        for j in range(12):
            if T[i, j] > 1e-12:
                H -= pi[i] * T[i, j] * np.log2(T[i, j])
    return float(H)


def harmonic_surprise_curve(
    progression: Sequence[int],
    context_window: int = 8,
) -> np.ndarray:
    """Sliding-window harmonic surprise using adaptive transition model.

    At each position, the model is trained on the preceding context_window events.
    """
    n = len(progression)
    surprises = np.zeros(max(0, n - 1))
    for i in range(1, n):
        start = max(0, i - context_window)
        context = progression[start:i + 1]
        T = chord_transition_matrix(context)
        s = progression[i - 1] % 12
        t = progression[i] % 12
        p = T[s, t]
        surprises[i - 1] = -np.log2(p + 1e-12)
    return surprises


# ══════════════════════════════════════════════════════════════════════
# § 8  Augmented sixth chords
# ══════════════════════════════════════════════════════════════════════

def classify_augmented_sixth(chord_pcs: Sequence[int], key: int = 0) -> Optional[str]:
    """Classify as Italian, French, or German augmented 6th (or None).

    All share b6-1-#4 (relative to key); Fr adds 2, Ger adds b3.
    """
    rel = sorted(set((pc - key) % 12 for pc in chord_pcs))
    b6 = 8
    one = 0
    s4 = 6

    if b6 not in rel or one not in rel or s4 not in rel:
        return None

    extras = set(rel) - {b6, one, s4}
    if len(extras) == 0:
        return 'Italian'
    if extras == {2}:
        return 'French'
    if extras == {3}:
        return 'German'
    return None


# ══════════════════════════════════════════════════════════════════════
# § 9  Aggregate analysis
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DeepHarmonicAnalysis:
    roughness_per_chord: np.ndarray
    tps_sequential: np.ndarray
    information_content: np.ndarray
    entropy_rate: float
    cadences: List[CadenceAnalysis]
    ursatz: Dict


def analyze_harmony_deep(
    chord_sequence: List[Tuple[List[int], int]],
    key: int = 0, mode: str = 'major',
    melody_pcs: Optional[Sequence[int]] = None,
    bass_pcs: Optional[Sequence[int]] = None,
) -> DeepHarmonicAnalysis:
    """Run the full deep harmonic analysis pipeline.

    chord_sequence: list of (pitch_class_list, root) tuples.
    """
    roughness = np.array([
        chord_roughness_harmonic(pcs) for pcs, _ in chord_sequence
    ])

    tps_seq = tps_sequential_tension(chord_sequence, key, mode)

    roots = [r for _, r in chord_sequence]
    ic = information_content(roots)
    h_rate = harmonic_entropy_rate(roots)

    cadences = []
    for i in range(len(chord_sequence) - 1):
        ca = analyze_cadence(
            chord_sequence[i][0], chord_sequence[i + 1][0],
            key, mode,
        )
        if ca.cadence_type != CadenceStrength.NONE:
            cadences.append(ca)

    ursatz = {}
    if melody_pcs is not None and bass_pcs is not None:
        ursatz = detect_ursatz(melody_pcs, bass_pcs, key, mode)

    return DeepHarmonicAnalysis(
        roughness_per_chord=roughness,
        tps_sequential=tps_seq,
        information_content=ic,
        entropy_rate=h_rate,
        cadences=cadences,
        ursatz=ursatz,
    )
