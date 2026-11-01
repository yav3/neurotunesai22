"""Deep rhythm analysis: spectral tempo, multi-agent beat tracking, groove.

Beyond basic beat/meter detection — implements:
- Spectral rhythm analysis: rhythm periodicity via ACF, rhythm spectrum (Lartillot, 2014)
- Multi-agent beat tracking (Dixon, 2001; Goto & Muraoka, 1999)
- Bjorklund's algorithm for Euclidean rhythms (Toussaint, 2005)
- Rhythmic similarity via edit distance on onset grids (Toussaint, 2013)
- nPVI and rhythmic typology (Patel & Daniele, 2003)
- Groove quantification: timing feel vector, ghost-note density, velocity profile
- Aksak and additive meters (London, 2012)
- Rhythmic entrainment: period/phase coupling (Large & Jones, 1999)

Pure numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ══════════════════════════════════════════════════════════════════════
# § 1  Bjorklund's algorithm — Euclidean rhythms (Toussaint, 2005)
# ══════════════════════════════════════════════════════════════════════

def bjorklund(pulses: int, steps: int) -> np.ndarray:
    """Bjorklund's algorithm: distribute k pulses as evenly as possible in n steps.

    Equivalent to computing E(k,n) Euclidean rhythm (Toussaint, 2005).
    The algorithm is the Euclidean GCD rhythm construction applied to binary sequences.

    Returns binary array of length `steps`.
    """
    if pulses >= steps:
        return np.ones(steps, dtype=int)
    if pulses <= 0:
        return np.zeros(steps, dtype=int)

    groups: list = [[1]] * pulses + [[0]] * (steps - pulses)
    while True:
        remainder_start = pulses
        n_groups = len(groups)
        n_remainder = n_groups - remainder_start
        if n_remainder <= 1:
            break
        new_groups = []
        for i in range(min(remainder_start, n_remainder)):
            new_groups.append(groups[i] + groups[remainder_start + i])
        leftover_start = min(remainder_start, n_remainder)
        if leftover_start < remainder_start:
            new_groups.extend(groups[leftover_start:remainder_start])
        elif leftover_start < n_remainder:
            for i in range(leftover_start + remainder_start, n_groups):
                new_groups.append(groups[i])
        groups = new_groups
        pulses = min(remainder_start, n_remainder)
    return np.array([bit for group in groups for bit in group], dtype=int)


def euclidean_necklace(pulses: int, steps: int) -> List[np.ndarray]:
    """All rotations (necklace class) of E(k,n)."""
    base = bjorklund(pulses, steps)
    return [np.roll(base, i) for i in range(steps)]


def euclidean_rhythm_family(steps: int) -> Dict[int, np.ndarray]:
    """All distinct Euclidean rhythms for a given step count."""
    return {k: bjorklund(k, steps) for k in range(1, steps)}


# ══════════════════════════════════════════════════════════════════════
# § 2  Spectral rhythm analysis — rhythm periodicity
# ══════════════════════════════════════════════════════════════════════

def onset_acf(onsets: np.ndarray, max_lag: int = 0) -> np.ndarray:
    """Normalised autocorrelation of an onset signal.

    Reveals periodicities at beat and sub-beat levels.
    If max_lag is 0, uses half the signal length.
    """
    n = len(onsets)
    if max_lag <= 0:
        max_lag = n // 2
    x = onsets - np.mean(onsets)
    var = np.dot(x, x)
    if var < 1e-12:
        return np.zeros(max_lag)
    acf = np.correlate(x, x, mode='full')
    acf = acf[n - 1:n - 1 + max_lag]
    return acf / (var + 1e-12)


def rhythm_spectrum(
    onset_signal: np.ndarray, sr: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rhythm periodicity spectrum via FFT of the onset-strength envelope.

    Returns (bpm_axis, magnitude_spectrum).
    sr = sampling rate of the onset signal in Hz.
    """
    n = len(onset_signal)
    window = np.hanning(n)
    X = np.fft.rfft(onset_signal * window)
    magnitude = np.abs(X)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    bpm = freqs * 60.0
    return bpm, magnitude


def rhythm_periodicity_peaks(
    onset_signal: np.ndarray, sr: float = 100.0,
    min_bpm: float = 40.0, max_bpm: float = 240.0,
    n_peaks: int = 3,
) -> List[Tuple[float, float]]:
    """Find dominant rhythmic periodicities in BPM.

    Returns list of (bpm, strength) tuples sorted by strength.
    """
    bpm, mag = rhythm_spectrum(onset_signal, sr)
    mask = (bpm >= min_bpm) & (bpm <= max_bpm)
    bpm_masked = bpm[mask]
    mag_masked = mag[mask]
    if len(mag_masked) < 3:
        return []

    peaks = []
    for i in range(1, len(mag_masked) - 1):
        if mag_masked[i] > mag_masked[i - 1] and mag_masked[i] > mag_masked[i + 1]:
            peaks.append((bpm_masked[i], mag_masked[i]))
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:n_peaks]


# ══════════════════════════════════════════════════════════════════════
# § 3  Multi-agent beat tracking (Dixon, 2001)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BeatAgent:
    """A single beat-tracking agent (Dixon, 2001)."""
    period: float        # predicted IOI
    phase: float         # time of last predicted beat
    score: float = 0.0   # accumulated evidence
    history: List[float] = field(default_factory=list)
    tolerance: float = 0.04  # fractional tolerance window

    def predict_next(self) -> float:
        return self.phase + self.period

    def evaluate(self, onset_time: float) -> float:
        predicted = self.predict_next()
        error = abs(onset_time - predicted)
        rel_error = error / (self.period + 1e-6)
        if rel_error < self.tolerance:
            return 1.0 - rel_error / self.tolerance
        return 0.0


def _create_initial_agents(
    onset_times: np.ndarray, max_agents: int = 50,
    min_ioi: float = 0.2, max_ioi: float = 2.0,
) -> List[BeatAgent]:
    """Create initial agent population from onset IOIs."""
    agents = []
    n = len(onset_times)
    for i in range(min(n - 1, max_agents)):
        for j in range(i + 1, min(n, i + 6)):
            ioi = onset_times[j] - onset_times[i]
            if min_ioi <= ioi <= max_ioi:
                agents.append(BeatAgent(
                    period=ioi, phase=onset_times[i], tolerance=0.04,
                ))
    agents.sort(key=lambda a: a.period)
    if len(agents) > max_agents:
        step = len(agents) // max_agents
        agents = agents[::step][:max_agents]
    return agents


def multi_agent_beat_track(
    onset_times: np.ndarray,
    min_bpm: float = 50.0, max_bpm: float = 220.0,
    tolerance: float = 0.04,
) -> Tuple[np.ndarray, float]:
    """Multi-agent beat tracking (Dixon, 2001).

    Multiple agents with different period/phase hypotheses compete.
    The winning agent's beat sequence is returned along with its BPM estimate.
    """
    min_ioi = 60.0 / max_bpm
    max_ioi = 60.0 / min_bpm
    agents = _create_initial_agents(onset_times, 40, min_ioi, max_ioi)

    if not agents:
        return np.array([]), 0.0

    for t in onset_times:
        for agent in agents:
            match = agent.evaluate(t)
            if match > 0:
                agent.score += match
                alpha = 0.1
                error = t - agent.predict_next()
                agent.period += alpha * error
                agent.period = np.clip(agent.period, min_ioi, max_ioi)
                agent.phase = t
                agent.history.append(t)

    best = max(agents, key=lambda a: a.score)
    bpm = 60.0 / best.period if best.period > 0 else 0.0

    beats = []
    t = onset_times[0] if len(onset_times) > 0 else 0.0
    end = onset_times[-1] if len(onset_times) > 0 else 0.0
    while t <= end:
        beats.append(t)
        t += best.period
    return np.array(beats), bpm


# ══════════════════════════════════════════════════════════════════════
# § 4  Rhythmic similarity metrics (Toussaint, 2013)
# ══════════════════════════════════════════════════════════════════════

def hamming_distance(r1: np.ndarray, r2: np.ndarray) -> int:
    """Hamming distance between two binary onset patterns of equal length."""
    assert len(r1) == len(r2), "Patterns must have equal length"
    return int(np.sum(r1 != r2))


def swap_distance(r1: np.ndarray, r2: np.ndarray) -> int:
    """Swap distance (minimum swaps to transform r1 into r2).

    Counts minimum onset-position swaps needed. Equal to half the
    earth-mover distance on the circular onset positions.
    """
    onsets1 = np.where(r1 > 0)[0]
    onsets2 = np.where(r2 > 0)[0]
    if len(onsets1) != len(onsets2):
        return abs(len(onsets1) - len(onsets2)) * len(r1)
    if len(onsets1) == 0:
        return 0
    n = len(r1)
    best = n * len(onsets1)
    for shift in range(n):
        shifted = (onsets2 + shift) % n
        shifted.sort()
        cost = int(np.sum(np.abs(onsets1 - shifted)))
        if cost < best:
            best = cost
    return best


def chronotonic_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    """Chronotonic distance: area between IOI duration chains (Toussaint, 2013).

    Compares the inter-onset interval sequences as step functions.
    """
    ioi1 = _binary_to_ioi(r1)
    ioi2 = _binary_to_ioi(r2)
    if len(ioi1) == 0 or len(ioi2) == 0:
        return float(len(r1))
    n = max(len(ioi1), len(ioi2))
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    s1[:len(ioi1)] = ioi1
    s2[:len(ioi2)] = ioi2
    if len(ioi1) < n:
        s1[len(ioi1):] = ioi1[-1]
    if len(ioi2) < n:
        s2[len(ioi2):] = ioi2[-1]
    return float(np.sum(np.abs(s1 - s2)))


def _binary_to_ioi(pattern: np.ndarray) -> np.ndarray:
    onsets = np.where(pattern > 0)[0]
    if len(onsets) < 2:
        return np.array([])
    return np.diff(onsets).astype(float)


def directed_swap_distance(r1: np.ndarray, r2: np.ndarray) -> int:
    """Directed swap distance: number of adjacent swaps in one direction.

    Useful for rhythmic "pull" analysis — how far ahead or behind onsets are shifted.
    """
    onsets1 = np.where(r1 > 0)[0]
    onsets2 = np.where(r2 > 0)[0]
    if len(onsets1) != len(onsets2):
        return -1
    return int(np.sum(onsets2 - onsets1))


# ══════════════════════════════════════════════════════════════════════
# § 5  Groove quantification — timing feel vector
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GrooveProfile:
    swing_ratio: float
    ghost_note_density: float
    velocity_variance: float
    timing_std_ms: float
    anticipation_ratio: float
    backbeat_emphasis: float
    feel_vector: np.ndarray


def quantify_groove(
    onset_times: np.ndarray,
    velocities: np.ndarray,
    grid_period: float,
    subdivisions: int = 4,
) -> GrooveProfile:
    """Comprehensive groove quantification.

    onset_times: actual onset times in seconds
    velocities: MIDI velocities (0–127) per onset
    grid_period: beat period in seconds
    subdivisions: grid subdivisions per beat (4 = 16th notes)
    """
    grid_step = grid_period / subdivisions
    n = len(onset_times)

    deviations = np.zeros(n)
    grid_positions = np.zeros(n, dtype=int)
    for i in range(n):
        nearest_grid = round(onset_times[i] / grid_step)
        grid_positions[i] = nearest_grid
        deviations[i] = onset_times[i] - nearest_grid * grid_step

    timing_std = float(np.std(deviations) * 1000)

    even_positions = deviations[grid_positions % 2 == 0]
    odd_positions = deviations[grid_positions % 2 == 1]
    if len(even_positions) > 0 and len(odd_positions) > 0:
        swing = float(np.mean(odd_positions) - np.mean(even_positions))
        swing_ratio = (grid_step + swing) / grid_step if grid_step > 0 else 1.0
    else:
        swing_ratio = 1.0

    ghost_threshold = np.percentile(velocities, 25) if len(velocities) > 4 else 40
    ghost_density = float(np.mean(velocities < ghost_threshold))

    vel_var = float(np.std(velocities) / (np.mean(velocities) + 1e-6))

    early = np.sum(deviations < -0.005)
    total_offgrid = np.sum(np.abs(deviations) > 0.005)
    anticipation = float(early / (total_offgrid + 1e-6))

    beat_vels = velocities[grid_positions % subdivisions == 0]
    offbeat_idx = subdivisions // 2
    offbeat_vels = velocities[grid_positions % subdivisions == offbeat_idx]
    if len(beat_vels) > 0 and len(offbeat_vels) > 0:
        backbeat = float(np.mean(offbeat_vels) / (np.mean(beat_vels) + 1e-6))
    else:
        backbeat = 0.0

    feel_vector = np.array([swing_ratio, ghost_density, vel_var,
                            timing_std, anticipation, backbeat])

    return GrooveProfile(
        swing_ratio=swing_ratio,
        ghost_note_density=ghost_density,
        velocity_variance=vel_var,
        timing_std_ms=timing_std,
        anticipation_ratio=anticipation,
        backbeat_emphasis=backbeat,
        feel_vector=feel_vector,
    )


def groove_distance(g1: GrooveProfile, g2: GrooveProfile) -> float:
    """Euclidean distance between groove feel vectors."""
    return float(np.linalg.norm(g1.feel_vector - g2.feel_vector))


# ══════════════════════════════════════════════════════════════════════
# § 6  Additive / aksak meters (London, 2012)
# ══════════════════════════════════════════════════════════════════════

AKSAK_METERS = {
    '5/8_23': (2, 3),
    '5/8_32': (3, 2),
    '7/8_223': (2, 2, 3),
    '7/8_232': (2, 3, 2),
    '7/8_322': (3, 2, 2),
    '9/8_2223': (2, 2, 2, 3),
    '9/8_2232': (2, 2, 3, 2),
    '9/8_2322': (2, 3, 2, 2),
    '9/8_3222': (3, 2, 2, 2),
    '9/8_333': (3, 3, 3),
    '11/8_33221': (3, 3, 2, 2, 1),
}


def aksak_grid(grouping: Tuple[int, ...], n_cycles: int = 2) -> np.ndarray:
    """Generate an aksak metric grid from a grouping pattern.

    grouping=(2,3) for 5/8 with 2+3 subdivision.
    Returns binary onset grid with 1s at group boundaries.
    """
    cycle_len = sum(grouping)
    total = cycle_len * n_cycles
    grid = np.zeros(total, dtype=int)
    pos = 0
    for _ in range(n_cycles):
        for g in grouping:
            grid[pos] = 1
            pos += g
    return grid


def detect_additive_meter(
    onset_signal: np.ndarray, candidates: Optional[List[Tuple[int, ...]]] = None,
) -> Tuple[Optional[Tuple[int, ...]], float]:
    """Detect the best-fitting additive meter pattern.

    Correlates onset signal against candidate aksak grids.
    Returns (best_grouping, correlation_score).
    """
    if candidates is None:
        candidates = list(AKSAK_METERS.values())

    best_grouping = None
    best_score = -1.0
    n = len(onset_signal)

    for grouping in candidates:
        cycle_len = sum(grouping)
        if cycle_len > n:
            continue
        grid = aksak_grid(grouping, n_cycles=(n // cycle_len) + 1)[:n]
        corr = np.corrcoef(onset_signal[:len(grid)], grid[:len(onset_signal)])[0, 1]
        if np.isfinite(corr) and corr > best_score:
            best_score = corr
            best_grouping = grouping

    return best_grouping, float(best_score)


# ══════════════════════════════════════════════════════════════════════
# § 7  Rhythmic entrainment — oscillator model (Large & Jones, 1999)
# ══════════════════════════════════════════════════════════════════════

def entrainment_oscillator(
    onset_times: np.ndarray,
    initial_period: float = 0.5,
    coupling: float = 0.3,
    adaptation_rate: float = 0.1,
    n_steps: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive oscillator model for rhythmic entrainment (Large & Jones, 1999).

    An internal oscillator with period P and phase φ adjusts to external onsets.
    At each onset, phase error drives corrections to both P and φ.

    Returns (predicted_beats, period_history).
    """
    if len(onset_times) < 2:
        return onset_times.copy(), np.array([initial_period])

    period = initial_period
    phase = onset_times[0]
    periods = [period]
    beats = [phase]

    for t in onset_times[1:]:
        predicted = phase + period
        error = t - predicted

        phase = predicted + coupling * error
        period = period + adaptation_rate * error
        period = max(period, 0.05)

        periods.append(period)
        beats.append(phase)

    return np.array(beats), np.array(periods)


def phase_coupling_strength(
    onset_times: np.ndarray, beat_period: float,
) -> float:
    """Measure how tightly onsets lock to a periodic grid (entrainment strength).

    Returns Rayleigh's R ∈ [0,1] — circular mean resultant length of
    onset phases relative to the beat period. 1 = perfect lock, 0 = uniform.
    """
    if len(onset_times) < 2 or beat_period <= 0:
        return 0.0
    phases = 2 * np.pi * (onset_times / beat_period) % (2 * np.pi)
    c = np.mean(np.cos(phases))
    s = np.mean(np.sin(phases))
    return float(np.sqrt(c ** 2 + s ** 2))


# ══════════════════════════════════════════════════════════════════════
# § 8  nPVI and rhythmic typology (Patel & Daniele, 2003)
# ══════════════════════════════════════════════════════════════════════

def compute_npvi(durations: np.ndarray) -> float:
    """Normalised Pairwise Variability Index (Patel & Daniele, 2003).

    nPVI = 100 × (1/(m-1)) Σ |d_k - d_{k+1}| / ((d_k + d_{k+1})/2)

    Higher nPVI → more durational contrast (e.g., English vs. French speech;
    analogously, swing vs. straight rhythms).
    """
    m = len(durations)
    if m < 2:
        return 0.0
    total = 0.0
    for k in range(m - 1):
        mean_pair = (durations[k] + durations[k + 1]) / 2.0
        if mean_pair > 1e-12:
            total += abs(durations[k] - durations[k + 1]) / mean_pair
    return 100.0 * total / (m - 1)


def rhythmic_typology_classify(npvi_value: float) -> str:
    """Classify rhythmic character from nPVI (Patel & Daniele, 2003).

    Based on cross-linguistic studies:
      < 35: syllable-timed (French, Spanish) → "straight"
      35–55: mixed
      > 55: stress-timed (English, German) → "swung/varied"
    """
    if npvi_value < 35:
        return 'straight'
    elif npvi_value < 55:
        return 'moderate'
    else:
        return 'varied'


def local_npvi(durations: np.ndarray, window: int = 8) -> np.ndarray:
    """Sliding-window nPVI for tracking rhythmic complexity over time."""
    n = len(durations)
    if n < window:
        return np.array([compute_npvi(durations)])
    result = []
    for i in range(n - window + 1):
        result.append(compute_npvi(durations[i:i + window]))
    return np.array(result)


# ══════════════════════════════════════════════════════════════════════
# § 9  Rhythmic complexity — Lempel-Ziv & entropy
# ══════════════════════════════════════════════════════════════════════

def lempel_ziv_complexity(pattern: np.ndarray) -> float:
    """Lempel-Ziv complexity of a binary rhythm pattern.

    Counts the number of distinct substrings in the LZ76 factorisation.
    Normalised by the theoretical maximum for a random binary sequence.
    """
    n = len(pattern)
    if n == 0:
        return 0.0
    s = ''.join(str(int(x)) for x in pattern)
    complexity = 1
    i = 0
    k = 1
    while i + k <= n:
        substr = s[i:i + k]
        if substr in s[:i]:
            k += 1
        else:
            complexity += 1
            i += k
            k = 1
    max_complexity = n / (np.log2(n) + 1e-12)
    return complexity / max_complexity


def rhythm_entropy(pattern: np.ndarray, order: int = 2) -> float:
    """n-gram entropy of a binary rhythm pattern.

    Higher entropy → less predictable rhythm.
    """
    n = len(pattern)
    if n < order + 1:
        return 0.0
    ngrams = {}
    for i in range(n - order):
        key = tuple(int(x) for x in pattern[i:i + order + 1])
        ngrams[key] = ngrams.get(key, 0) + 1
    total = sum(ngrams.values())
    H = 0.0
    for count in ngrams.values():
        p = count / total
        H -= p * np.log2(p + 1e-12)
    return float(H)


# ══════════════════════════════════════════════════════════════════════
# § 10  Aggregate deep rhythm analysis
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DeepRhythmAnalysis:
    estimated_bpm: float
    beat_times: np.ndarray
    periodicity_peaks: List[Tuple[float, float]]
    groove_profile: Optional[GrooveProfile]
    npvi: float
    typology: str
    lz_complexity: float
    entropy: float
    entrainment_strength: float
    additive_meter: Optional[Tuple[int, ...]]


def analyze_rhythm_deep(
    onset_times: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    onset_signal: Optional[np.ndarray] = None,
    sr: float = 100.0,
) -> DeepRhythmAnalysis:
    """Full deep rhythm analysis pipeline."""
    beats, bpm = multi_agent_beat_track(onset_times)

    peaks = []
    if onset_signal is not None:
        peaks = rhythm_periodicity_peaks(onset_signal, sr)

    groove = None
    if velocities is not None and bpm > 0:
        beat_period = 60.0 / bpm
        groove = quantify_groove(onset_times, velocities, beat_period)

    iois = np.diff(onset_times) if len(onset_times) > 1 else np.array([])
    npvi_val = compute_npvi(iois) if len(iois) > 0 else 0.0
    typology = rhythmic_typology_classify(npvi_val)

    quantized = np.zeros(int(max(onset_times) * 10) + 1) if len(onset_times) > 0 else np.array([])
    for t in onset_times:
        idx = int(t * 10)
        if 0 <= idx < len(quantized):
            quantized[idx] = 1
    lz = lempel_ziv_complexity(quantized) if len(quantized) > 0 else 0.0
    entropy = rhythm_entropy(quantized) if len(quantized) > 0 else 0.0

    beat_period = 60.0 / bpm if bpm > 0 else 0.5
    entrain = phase_coupling_strength(onset_times, beat_period)

    additive = None
    if onset_signal is not None:
        additive, _ = detect_additive_meter(onset_signal)

    return DeepRhythmAnalysis(
        estimated_bpm=bpm, beat_times=beats,
        periodicity_peaks=peaks, groove_profile=groove,
        npvi=npvi_val, typology=typology,
        lz_complexity=lz, entropy=entropy,
        entrainment_strength=entrain,
        additive_meter=additive,
    )
