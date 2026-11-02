"""Deep music theory — neo-Riemannian, set theory, voice leading, tuning systems."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, FrozenSet, Set

# ---------------------------------------------------------------------------
# §1  Neo-Riemannian operations (Cohn, 1998)
# ---------------------------------------------------------------------------

def _triad(root: int, quality: str) -> Tuple[int, int, int]:
    r = root % 12
    if quality == "major":
        return (r, (r + 4) % 12, (r + 7) % 12)
    return (r, (r + 3) % 12, (r + 7) % 12)


def parallel(triad: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """P: major↔minor by moving the third."""
    r, t, f = triad
    interval = (t - r) % 12
    if interval == 4:  # major → minor
        return (r, (r + 3) % 12, f)
    return (r, (r + 4) % 12, f)


def relative(triad: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """R: relative major↔minor."""
    r, t, f = triad
    interval = (t - r) % 12
    if interval == 4:  # major → relative minor
        return ((r + 9) % 12, r, (r + 4) % 12)
    return ((r + 3) % 12, (r + 7) % 12, r)


def leading_tone(triad: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """L: leading-tone exchange."""
    r, t, f = triad
    interval = (t - r) % 12
    if interval == 4:  # major → L
        return ((r + 11) % 12, r, (r + 4) % 12)
    return ((r + 3) % 12, (r + 7) % 12, (r + 8) % 12)


def neo_riemannian_distance(triad_a: Tuple[int, int, int],
                            triad_b: Tuple[int, int, int],
                            max_depth: int = 6) -> int:
    """BFS shortest PLR-path length between two triads."""
    sa, sb = frozenset(triad_a), frozenset(triad_b)
    if sa == sb:
        return 0
    ops = [parallel, relative, leading_tone]
    visited: Set[FrozenSet[int]] = {sa}
    frontier = [triad_a]
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for tri in frontier:
            for op in ops:
                result = op(tri)
                fs = frozenset(result)
                if fs == sb:
                    return depth
                if fs not in visited:
                    visited.add(fs)
                    next_frontier.append(result)
        frontier = next_frontier
    return max_depth + 1

# ---------------------------------------------------------------------------
# §2  Pitch-class set theory (Forte)
# ---------------------------------------------------------------------------

def to_pc_set(pitches: np.ndarray) -> FrozenSet[int]:
    return frozenset(int(p) % 12 for p in pitches)


def interval_class_vector(pc_set: FrozenSet[int]) -> np.ndarray:
    """6-element interval-class vector."""
    icv = np.zeros(6, dtype=int)
    pcs = sorted(pc_set)
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            ic = min((pcs[j] - pcs[i]) % 12, (pcs[i] - pcs[j]) % 12)
            if 1 <= ic <= 6:
                icv[ic - 1] += 1
    return icv


def transpose_set(pc_set: FrozenSet[int], n: int) -> FrozenSet[int]:
    return frozenset((p + n) % 12 for p in pc_set)


def invert_set(pc_set: FrozenSet[int]) -> FrozenSet[int]:
    return frozenset((12 - p) % 12 for p in pc_set)


def normal_order(pc_set: FrozenSet[int]) -> Tuple[int, ...]:
    """Normal order (most compact ascending arrangement)."""
    pcs = sorted(pc_set)
    n = len(pcs)
    if n == 0:
        return ()
    best = None
    for rotation in range(n):
        rotated = [pcs[(rotation + i) % n] for i in range(n)]
        # make ascending from first element
        ordered = [(rotated[i] - rotated[0]) % 12 for i in range(n)]
        if best is None or ordered < best:
            best = ordered
            best_start = rotated[0]
    return tuple((best_start + x) % 12 for x in best)


def prime_form(pc_set: FrozenSet[int]) -> Tuple[int, ...]:
    """Prime form (most compact, starting from 0)."""
    no = normal_order(pc_set)
    inv_no = normal_order(invert_set(pc_set))
    # normalise both to start at 0
    n1 = tuple((p - no[0]) % 12 for p in no) if no else ()
    n2 = tuple((p - inv_no[0]) % 12 for p in inv_no) if inv_no else ()
    return min(n1, n2) if n1 and n2 else n1 or n2


def z_related(set_a: FrozenSet[int], set_b: FrozenSet[int]) -> bool:
    """Z-relation: same ICV but different prime forms."""
    if len(set_a) != len(set_b):
        return False
    return (np.array_equal(interval_class_vector(set_a),
                           interval_class_vector(set_b)) and
            prime_form(set_a) != prime_form(set_b))

# ---------------------------------------------------------------------------
# §3  Diatonic set theory
# ---------------------------------------------------------------------------

def maximal_evenness(cardinality: int, chromatic: int = 12) -> np.ndarray:
    """Maximally even set (Clough & Douthett, 1991)."""
    return np.array([int(np.floor(i * chromatic / cardinality))
                     for i in range(cardinality)]) % chromatic


def is_well_formed(scale: np.ndarray) -> bool:
    """A scale is well-formed if generated by a single interval mod chromatic."""
    intervals = np.diff(np.sort(scale % 12))
    unique_intervals = np.unique(intervals)
    return len(unique_intervals) <= 2


def myhill_property(scale: np.ndarray) -> bool:
    """Myhill's property: exactly two interval sizes for each generic interval."""
    pcs = sorted(set(int(p) % 12 for p in scale))
    n = len(pcs)
    for g in range(1, n):
        specific = set()
        for i in range(n):
            s = (pcs[(i + g) % n] - pcs[i]) % 12
            specific.add(s)
        if len(specific) != 2:
            return False
    return True

# ---------------------------------------------------------------------------
# §4  Generalised voice leading (Tymoczko, 2006)
# ---------------------------------------------------------------------------

def voice_leading_distance(chord_a: np.ndarray, chord_b: np.ndarray) -> float:
    """Minimal voice-leading distance in semitones (L1 in pitch-class space)."""
    a = np.sort(np.array(chord_a) % 12.0)
    b = np.sort(np.array(chord_b) % 12.0)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    # try all cyclic permutations, pick minimal
    best = float('inf')
    for shift in range(n):
        b_perm = np.roll(b, shift)
        diffs = np.abs(a - b_perm)
        diffs = np.minimum(diffs, 12 - diffs)
        d = float(np.sum(diffs))
        best = min(best, d)
    return best


def smooth_voice_leading(chord_a: np.ndarray, chord_b: np.ndarray) -> bool:
    """Whether voice leading is smooth (max motion ≤ 2 semitones per voice)."""
    a = np.sort(np.array(chord_a) % 12.0)
    b = np.sort(np.array(chord_b) % 12.0)
    n = min(len(a), len(b))
    for shift in range(n):
        b_perm = np.roll(b[:n], shift)
        diffs = np.abs(a[:n] - b_perm)
        diffs = np.minimum(diffs, 12 - diffs)
        if diffs.max() <= 2:
            return True
    return False

# ---------------------------------------------------------------------------
# §5  Transformational theory — GIS (Lewin, 1987)
# ---------------------------------------------------------------------------

def gis_interval(a: int, b: int, modulus: int = 12) -> int:
    """Generalised interval in Z_n."""
    return (b - a) % modulus


def gis_transpose(element: int, interval: int, modulus: int = 12) -> int:
    return (element + interval) % modulus


def gis_inversion(element: int, axis: int = 0, modulus: int = 12) -> int:
    return (2 * axis - element) % modulus


def interval_function(set_a: FrozenSet[int], set_b: FrozenSet[int],
                      modulus: int = 12) -> Dict[int, int]:
    """IFUNC: counts of interval i from set_a to set_b."""
    ifunc = {i: 0 for i in range(modulus)}
    for a in set_a:
        for b in set_b:
            ifunc[(b - a) % modulus] += 1
    return ifunc

# ---------------------------------------------------------------------------
# §6  Roman numeral analysis
# ---------------------------------------------------------------------------

TRIAD_TYPES = {
    (0, 4, 7): "M", (0, 3, 7): "m", (0, 3, 6): "dim", (0, 4, 8): "aug",
}
ROMAN = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]


def roman_numeral(pitches: np.ndarray, key_root: int = 0) -> str:
    """Assign a Roman numeral label given pitches and key."""
    pcs = sorted(set((int(p) - key_root) % 12 for p in pitches))
    if len(pcs) < 3:
        return "?"
    root_pc = pcs[0]
    intervals = tuple((p - root_pc) % 12 for p in pcs[:3])
    quality = TRIAD_TYPES.get(intervals, "?")
    degree = ROMAN[root_pc % 12]
    if quality == "m":
        degree = degree.lower()
    elif quality == "dim":
        degree = degree.lower() + "°"
    elif quality == "aug":
        degree = degree + "+"
    return degree

# ---------------------------------------------------------------------------
# §7  Harmonic function (Riemann T/S/D)
# ---------------------------------------------------------------------------

FUNCTION_MAP = {0: "T", 2: "T", 4: "T",
                5: "S", 9: "S",
                7: "D", 11: "D"}


def harmonic_function(root_pc: int, key_root: int = 0) -> str:
    """Riemann T/S/D classification."""
    degree = (root_pc - key_root) % 12
    return FUNCTION_MAP.get(degree, "?")


def functional_distance(func_a: str, func_b: str) -> int:
    """Distance on T→S→D→T cycle."""
    order = ["T", "S", "D"]
    if func_a not in order or func_b not in order:
        return 3
    return (order.index(func_b) - order.index(func_a)) % 3

# ---------------------------------------------------------------------------
# §8  Scale theory (Balzano, 1980; Rothenberg)
# ---------------------------------------------------------------------------

def generated_scale(generator: int, cardinality: int,
                    chromatic: int = 12) -> np.ndarray:
    """Scale generated by repeated application of generator mod chromatic."""
    return np.sort(np.array([(generator * i) % chromatic
                             for i in range(cardinality)]))


def propriety(scale: np.ndarray) -> str:
    """Rothenberg propriety: strictly proper, proper, or improper."""
    pcs = sorted(set(int(p) % 12 for p in scale))
    n = len(pcs)
    intervals_by_generic: Dict[int, Set[int]] = {}
    for g in range(1, n):
        intervals_by_generic[g] = set()
        for i in range(n):
            s = (pcs[(i + g) % n] - pcs[i]) % 12
            intervals_by_generic[g].add(s)
    # check overlap
    is_proper = True
    is_strictly_proper = True
    for g1 in intervals_by_generic:
        for g2 in intervals_by_generic:
            if g1 < g2:
                if max(intervals_by_generic[g1]) > min(intervals_by_generic[g2]):
                    is_proper = False
                    is_strictly_proper = False
                elif max(intervals_by_generic[g1]) == min(intervals_by_generic[g2]):
                    is_strictly_proper = False
    if is_strictly_proper:
        return "strictly proper"
    if is_proper:
        return "proper"
    return "improper"

# ---------------------------------------------------------------------------
# §9  Tuning systems
# ---------------------------------------------------------------------------

def equal_temperament(n_divisions: int = 12) -> np.ndarray:
    """Equal temperament: n divisions of the octave in cents."""
    return np.arange(n_divisions) * (1200.0 / n_divisions)


def pythagorean_tuning(n_notes: int = 12) -> np.ndarray:
    """Pythagorean tuning via stacked perfect fifths (3:2)."""
    ratios = np.array([(3.0 / 2.0) ** i for i in range(n_notes)])
    # reduce to one octave
    while np.any(ratios >= 2.0):
        ratios = np.where(ratios >= 2.0, ratios / 2.0, ratios)
    return np.sort(1200.0 * np.log2(ratios))


def meantone_tuning(comma_fraction: float = 0.25, n_notes: int = 12) -> np.ndarray:
    """Quarter-comma meantone (default) or general meantone."""
    syntonic_comma = 1200.0 * np.log2(81.0 / 80.0)  # ~21.5 cents
    fifth = 1200.0 * np.log2(3.0 / 2.0) - comma_fraction * syntonic_comma
    notes = np.array([(fifth * i) % 1200.0 for i in range(n_notes)])
    return np.sort(notes)


def cent_deviation(tuning: np.ndarray, reference: Optional[np.ndarray] = None
                   ) -> np.ndarray:
    """Deviation in cents from equal temperament (or given reference)."""
    if reference is None:
        reference = equal_temperament(len(tuning))
    n = min(len(tuning), len(reference))
    return tuning[:n] - reference[:n]


def beating_rate(freq_a: float, freq_b: float) -> float:
    """Beat frequency between two tones."""
    return abs(freq_a - freq_b)

# ---------------------------------------------------------------------------
# §10  Pipeline
# ---------------------------------------------------------------------------

@dataclass
class DeepTheoryAnalysis:
    pc_set: Tuple[int, ...] = ()
    prime_form_val: Tuple[int, ...] = ()
    icv: np.ndarray = field(default_factory=lambda: np.zeros(6))
    roman_numerals: List[str] = field(default_factory=list)
    harmonic_functions: List[str] = field(default_factory=list)
    neo_riemannian_path_lengths: List[int] = field(default_factory=list)
    voice_leading_distances: List[float] = field(default_factory=list)
    scale_propriety: str = ""
    tuning_deviation_cents: np.ndarray = field(default_factory=lambda: np.array([]))
    myhill: bool = False
    well_formed: bool = False


def analyze_theory_deep(chords: List[np.ndarray],
                        key_root: int = 0,
                        scale: Optional[np.ndarray] = None
                        ) -> DeepTheoryAnalysis:
    """Full deep theory analysis on a chord sequence."""
    if not chords:
        return DeepTheoryAnalysis()
    # overall pc set
    all_pcs = to_pc_set(np.concatenate(chords))
    pf = prime_form(all_pcs)
    icv = interval_class_vector(all_pcs)

    romans = [roman_numeral(c, key_root) for c in chords]
    funcs = []
    for c in chords:
        root = int(np.sort(c % 12)[0])
        funcs.append(harmonic_function(root, key_root))

    # neo-Riemannian distances between consecutive chords (as triads)
    nr_dists = []
    for i in range(len(chords) - 1):
        a = tuple(int(p) % 12 for p in sorted(chords[i])[:3])
        b = tuple(int(p) % 12 for p in sorted(chords[i + 1])[:3])
        nr_dists.append(neo_riemannian_distance(a, b))

    vl_dists = []
    for i in range(len(chords) - 1):
        vl_dists.append(voice_leading_distance(chords[i], chords[i + 1]))

    if scale is None:
        scale = np.array([0, 2, 4, 5, 7, 9, 11])  # C major
    prop = propriety(scale)
    wf = is_well_formed(scale)
    my = myhill_property(scale)

    et = equal_temperament()
    pyth = pythagorean_tuning()
    dev = cent_deviation(pyth, et)

    return DeepTheoryAnalysis(
        pc_set=tuple(sorted(all_pcs)),
        prime_form_val=pf,
        icv=icv,
        roman_numerals=romans,
        harmonic_functions=funcs,
        neo_riemannian_path_lengths=nr_dists,
        voice_leading_distances=vl_dists,
        scale_propriety=prop,
        tuning_deviation_cents=dev,
        myhill=my,
        well_formed=wf,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # C major → A minor → F major → G major
    chords = [
        np.array([0, 4, 7]),
        np.array([9, 0, 4]),
        np.array([5, 9, 0]),
        np.array([7, 11, 2]),
    ]
    result = analyze_theory_deep(chords, key_root=0)
    assert len(result.roman_numerals) == 4
    assert result.roman_numerals[0] == "I"
    assert result.scale_propriety == "strictly proper"
    assert result.myhill is True
    print(f"PC set: {result.pc_set}, prime form: {result.prime_form_val}")
    print(f"Romans: {result.roman_numerals}")
    print(f"Functions: {result.harmonic_functions}")
    print(f"NR distances: {result.neo_riemannian_path_lengths}")
    print(f"VL distances: {result.voice_leading_distances}")
    print(f"Scale: {result.scale_propriety}, Myhill={result.myhill}")
    print("deep_theory OK")
