"""Deep algorithmic composition: evolutionary, fractal, stochastic, and
constraint-based generative methods.

References:
  - Genetic algorithms for music: Biles (1994) GenJam
  - L-system music: Prusinkiewicz & Lindenmayer (1990)
  - Fractal music: Voss & Clarke (1978) 1/f noise
  - Constraint satisfaction: Pachet & Roy (2001)
  - Stochastic processes: Xenakis (1992) Formalized Music
  - Variable-order Markov: Conklin & Witten (1995)
  - Cellular automata music: Wolfram (1984), Miranda (2001)
  - Schillinger system rhythmic interference patterns

Pure numpy. No external ML frameworks.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Set

# ---------------------------------------------------------------------------
# §1  Variable-order Markov chain
# ---------------------------------------------------------------------------

@dataclass
class MarkovChain:
    """Variable-order Markov chain (Conklin & Witten, 1995)."""
    max_order: int = 4
    transitions: Dict[Tuple, Dict[int, int]] = field(default_factory=dict)
    alphabet: Set[int] = field(default_factory=set)

    def train(self, sequence: List[int]) -> None:
        for s in sequence:
            self.alphabet.add(s)
        for order in range(1, self.max_order + 1):
            for i in range(order, len(sequence)):
                context = tuple(sequence[i - order:i])
                symbol = sequence[i]
                if context not in self.transitions:
                    self.transitions[context] = {}
                self.transitions[context][symbol] = \
                    self.transitions[context].get(symbol, 0) + 1

    def generate(
        self, seed: List[int], length: int, rng: Optional[np.random.RandomState] = None
    ) -> List[int]:
        rng = rng or np.random.RandomState(42)
        result = list(seed)
        for _ in range(length):
            symbol = self._predict(result, rng)
            result.append(symbol)
        return result

    def _predict(self, history: List[int], rng: np.random.RandomState) -> int:
        for order in range(min(self.max_order, len(history)), 0, -1):
            ctx = tuple(history[-order:])
            if ctx in self.transitions:
                dist = self.transitions[ctx]
                symbols = list(dist.keys())
                counts = np.array([dist[s] for s in symbols], dtype=np.float64)
                probs = counts / counts.sum()
                return int(rng.choice(symbols, p=probs))
        return int(rng.choice(list(self.alphabet)))

    def cross_entropy(self, sequence: List[int]) -> float:
        """Per-symbol cross-entropy H(model, data)."""
        log_prob = 0.0
        count = 0
        for i in range(1, len(sequence)):
            for order in range(min(self.max_order, i), 0, -1):
                ctx = tuple(sequence[i - order:i])
                if ctx in self.transitions:
                    dist = self.transitions[ctx]
                    total = sum(dist.values())
                    p = dist.get(sequence[i], 0) / total
                    if p > 0:
                        log_prob -= np.log2(p)
                    else:
                        log_prob -= np.log2(1.0 / len(self.alphabet))
                    count += 1
                    break
            else:
                log_prob -= np.log2(1.0 / max(len(self.alphabet), 1))
                count += 1
        return log_prob / max(count, 1)


# ---------------------------------------------------------------------------
# §2  L-system (Lindenmayer system) for melodic generation
# ---------------------------------------------------------------------------

@dataclass
class LSystem:
    axiom: str
    rules: Dict[str, str]
    angle: float = 30.0

    def iterate(self, n: int) -> str:
        result = self.axiom
        for _ in range(n):
            result = ''.join(self.rules.get(c, c) for c in result)
        return result

    def to_melody(
        self, n_iterations: int = 4, base_pitch: int = 60
    ) -> List[int]:
        """Interpret L-system string as melody via turtle graphics.

        F → step up, f → step down, + → raise pitch, - → lower pitch,
        [ → push state, ] → pop state
        """
        s = self.iterate(n_iterations)
        pitch = base_pitch
        stack: List[int] = []
        melody: List[int] = []
        step = int(self.angle / 10) or 1

        for c in s:
            if c == 'F':
                pitch += step
                melody.append(np.clip(pitch, 21, 108))
            elif c == 'f':
                pitch -= step
                melody.append(np.clip(pitch, 21, 108))
            elif c == '+':
                pitch += step
            elif c == '-':
                pitch -= step
            elif c == '[':
                stack.append(pitch)
            elif c == ']' and stack:
                pitch = stack.pop()
        return melody

def dragon_curve_melody(n: int = 6, base_pitch: int = 60) -> List[int]:
    """Dragon curve L-system → melody."""
    ls = LSystem(axiom='FX', rules={'X': 'X+YF+', 'Y': '-FX-Y'}, angle=20.0)
    return ls.to_melody(n, base_pitch)

def sierpinski_melody(n: int = 5, base_pitch: int = 60) -> List[int]:
    """Sierpinski triangle L-system → melody."""
    ls = LSystem(axiom='F-G-G', rules={'F': 'F-G+F+G-F', 'G': 'GG'}, angle=30.0)
    return ls.to_melody(n, base_pitch)


# ---------------------------------------------------------------------------
# §3  Fractal/1/f noise generation (Voss & Clarke, 1978)
# ---------------------------------------------------------------------------

def pink_noise(n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Generate 1/f (pink) noise via spectral shaping."""
    rng = rng or np.random.RandomState(42)
    white = rng.randn(n)
    S = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    S /= np.sqrt(freqs)
    return np.fft.irfft(S, n=n)

def brown_noise(n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Generate 1/f² (Brownian) noise."""
    rng = rng or np.random.RandomState(42)
    white = rng.randn(n)
    S = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    S /= freqs
    return np.fft.irfft(S, n=n)

def fractal_melody(
    n_notes: int = 64,
    pitch_range: Tuple[int, int] = (48, 84),
    exponent: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    """Generate a melody from 1/f^α noise.

    exponent=0: white (random walk), =1: pink, =2: Brownian.
    """
    rng = rng or np.random.RandomState(42)
    white = rng.randn(n_notes)
    S = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_notes)
    freqs[0] = 1.0
    S /= freqs ** (exponent / 2.0)
    signal = np.fft.irfft(S, n=n_notes)
    lo, hi = pitch_range
    normalized = (signal - signal.min()) / (signal.max() - signal.min() + 1e-12)
    return [int(round(lo + normalized[i] * (hi - lo))) for i in range(n_notes)]


# ---------------------------------------------------------------------------
# §4  Cellular automaton music (Wolfram, 1984; Miranda, 2001)
# ---------------------------------------------------------------------------

def elementary_ca(
    rule: int = 110, width: int = 64, steps: int = 64,
    init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run an elementary (1-D, 2-state, 3-neighbour) cellular automaton.

    Returns grid of shape (steps, width).
    """
    grid = np.zeros((steps, width), dtype=np.int8)
    if init is not None:
        grid[0, :len(init)] = init[:width]
    else:
        grid[0, width // 2] = 1

    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int8)
    for t in range(1, steps):
        for i in range(width):
            left = grid[t - 1, (i - 1) % width]
            centre = grid[t - 1, i]
            right = grid[t - 1, (i + 1) % width]
            idx = (left << 2) | (centre << 1) | right
            grid[t, i] = rule_bits[idx]
    return grid

def ca_to_melody(
    grid: np.ndarray,
    scale: Optional[List[int]] = None,
    base_pitch: int = 48,
) -> List[int]:
    """Map CA rows to melody: count active cells → scale degree."""
    if scale is None:
        scale = [0, 2, 4, 5, 7, 9, 11]
    melody = []
    for row in grid:
        active = int(np.sum(row))
        degree = active % len(scale)
        octave = active // len(scale)
        pitch = base_pitch + scale[degree] + 12 * min(octave, 3)
        melody.append(np.clip(pitch, 21, 108))
    return melody

def ca_to_rhythm(grid: np.ndarray, row: int = 0) -> List[int]:
    """Extract binary rhythm pattern from one CA row."""
    return grid[row].tolist()


# ---------------------------------------------------------------------------
# §5  Genetic algorithm for melody (Biles, 1994)
# ---------------------------------------------------------------------------

@dataclass
class GeneticMelodyConfig:
    population_size: int = 32
    n_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    melody_length: int = 16
    pitch_range: Tuple[int, int] = (48, 84)
    scale: List[int] = field(default_factory=lambda: [0, 2, 4, 5, 7, 9, 11])

def genetic_melody(
    fitness_fn: Callable[[List[int]], float],
    config: Optional[GeneticMelodyConfig] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[List[int], float, List[float]]:
    """Evolve a melody via genetic algorithm.

    Args:
        fitness_fn: melody → scalar fitness (higher is better)
        config: GA parameters
    Returns:
        (best_melody, best_fitness, generation_bests)
    """
    config = config or GeneticMelodyConfig()
    rng = rng or np.random.RandomState(42)
    lo, hi = config.pitch_range

    def random_melody() -> List[int]:
        return [int(rng.randint(lo, hi + 1)) for _ in range(config.melody_length)]

    def mutate(m: List[int]) -> List[int]:
        result = list(m)
        for i in range(len(result)):
            if rng.random() < config.mutation_rate:
                result[i] = int(np.clip(result[i] + rng.randint(-5, 6), lo, hi))
        return result

    def crossover(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
        if rng.random() > config.crossover_rate:
            return list(a), list(b)
        pt = rng.randint(1, len(a))
        return a[:pt] + b[pt:], b[:pt] + a[pt:]

    pop = [random_melody() for _ in range(config.population_size)]
    fits = np.array([fitness_fn(m) for m in pop])
    gen_bests: List[float] = []

    for _ in range(config.n_generations):
        gen_bests.append(float(np.max(fits)))
        probs = fits - fits.min() + 1e-6
        probs /= probs.sum()

        new_pop: List[List[int]] = []
        best_idx = int(np.argmax(fits))
        new_pop.append(list(pop[best_idx]))

        while len(new_pop) < config.population_size:
            i, j = rng.choice(len(pop), size=2, replace=False, p=probs)
            c1, c2 = crossover(pop[i], pop[j])
            new_pop.append(mutate(c1))
            if len(new_pop) < config.population_size:
                new_pop.append(mutate(c2))

        pop = new_pop[:config.population_size]
        fits = np.array([fitness_fn(m) for m in pop])

    best_idx = int(np.argmax(fits))
    return pop[best_idx], float(fits[best_idx]), gen_bests


# ---------------------------------------------------------------------------
# §6  Schillinger interference patterns
# ---------------------------------------------------------------------------

def interference_pattern(
    a: int, b: int, length: int = 64
) -> np.ndarray:
    """Schillinger rhythmic interference: superposition of two pulse trains.

    Two generators with periods a and b create a resultant pattern
    of length = lcm(a,b) repeated to fill `length`.
    """
    lcm = int(np.lcm(a, b))
    pattern = np.zeros(lcm, dtype=np.int8)
    for i in range(lcm):
        pulse_a = 1 if i % a == 0 else 0
        pulse_b = 1 if i % b == 0 else 0
        pattern[i] = max(pulse_a, pulse_b)
    reps = length // lcm + 1
    return np.tile(pattern, reps)[:length]

def resultant_rhythm(periods: List[int], length: int = 64) -> np.ndarray:
    """Multi-generator Schillinger resultant."""
    result = np.zeros(length, dtype=np.int8)
    for p in periods:
        for i in range(0, length, p):
            result[i] = 1
    return result

def complementary_rhythm(pattern: np.ndarray) -> np.ndarray:
    """Schillinger complementary: invert onsets and rests."""
    return 1 - pattern


# ---------------------------------------------------------------------------
# §7  Stochastic composition (Xenakis, 1992)
# ---------------------------------------------------------------------------

def poisson_events(rate: float, duration: float, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Generate Poisson-distributed event times (Xenakis stochastic music)."""
    rng = rng or np.random.RandomState(42)
    intervals = rng.exponential(1.0 / max(rate, 0.01), size=int(rate * duration * 3))
    times = np.cumsum(intervals)
    return times[times < duration]

def gaussian_cloud(
    n_events: int,
    pitch_mean: float = 60.0,
    pitch_std: float = 12.0,
    time_mean: float = 2.0,
    time_std: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian pitch/time cloud (Xenakis Pithoprakta model).

    Returns (pitches, times).
    """
    rng = rng or np.random.RandomState(42)
    pitches = np.clip(rng.normal(pitch_mean, pitch_std, n_events), 21, 108).astype(int)
    times = np.abs(rng.normal(time_mean, time_std, n_events))
    return pitches, np.sort(times)

def logistic_map_sequence(
    r: float = 3.99, x0: float = 0.5, n: int = 64
) -> np.ndarray:
    """Logistic map x_{n+1} = r·x_n·(1 - x_n) for chaotic pitch generation."""
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


# ---------------------------------------------------------------------------
# §8  Constraint satisfaction composer
# ---------------------------------------------------------------------------

@dataclass
class ConstraintSpec:
    """A constraint on a generated sequence."""
    name: str
    check: Callable[[List[int]], bool]
    weight: float = 1.0

def check_stepwise_motion(melody: List[int], max_leap: int = 7) -> bool:
    """At least 70% of intervals ≤ max_leap semitones."""
    if len(melody) < 2:
        return True
    intervals = [abs(melody[i + 1] - melody[i]) for i in range(len(melody) - 1)]
    return sum(1 for iv in intervals if iv <= max_leap) / len(intervals) >= 0.7

def check_range(melody: List[int], max_range: int = 24) -> bool:
    """Total range ≤ max_range semitones."""
    return max(melody) - min(melody) <= max_range

def check_scale_adherence(
    melody: List[int], scale: Optional[List[int]] = None
) -> bool:
    """≥ 80% of notes belong to the given scale (pitch classes)."""
    if scale is None:
        scale = [0, 2, 4, 5, 7, 9, 11]
    pcs = [m % 12 for m in melody]
    return sum(1 for pc in pcs if pc in scale) / max(len(pcs), 1) >= 0.8

def constraint_generate(
    constraints: List[ConstraintSpec],
    melody_length: int = 16,
    pitch_range: Tuple[int, int] = (48, 84),
    max_attempts: int = 5000,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[Optional[List[int]], float]:
    """Generate a melody satisfying weighted constraints via rejection sampling.

    Returns (melody, satisfaction_score). None if no melody found.
    """
    rng = rng or np.random.RandomState(42)
    lo, hi = pitch_range
    best_melody: Optional[List[int]] = None
    best_score = -1.0

    for _ in range(max_attempts):
        melody = [int(rng.randint(lo, hi + 1)) for _ in range(melody_length)]
        score = sum(c.weight * float(c.check(melody)) for c in constraints)
        total_weight = sum(c.weight for c in constraints)
        norm_score = score / max(total_weight, 1e-12)
        if norm_score > best_score:
            best_score = norm_score
            best_melody = melody
        if norm_score >= 1.0:
            return melody, norm_score

    return best_melody, best_score


# ---------------------------------------------------------------------------
# §9  Fitness functions for evolutionary composition
# ---------------------------------------------------------------------------

def fitness_tonal_coherence(melody: List[int], scale: Optional[List[int]] = None) -> float:
    """Fitness: proportion of notes in scale + stepwise motion bonus."""
    if scale is None:
        scale = [0, 2, 4, 5, 7, 9, 11]
    pcs = [m % 12 for m in melody]
    in_scale = sum(1 for pc in pcs if pc in scale) / max(len(pcs), 1)
    if len(melody) < 2:
        return in_scale
    intervals = [abs(melody[i + 1] - melody[i]) for i in range(len(melody) - 1)]
    step_frac = sum(1 for iv in intervals if iv <= 4) / len(intervals)
    return 0.6 * in_scale + 0.4 * step_frac

def fitness_rhythmic_interest(durations: List[float]) -> float:
    """Fitness: penalise too-uniform rhythms, reward variety."""
    if len(durations) < 2:
        return 0.0
    unique_ratio = len(set(durations)) / len(durations)
    cv = float(np.std(durations) / (np.mean(durations) + 1e-12))
    return 0.5 * unique_ratio + 0.5 * min(cv, 1.0)

def fitness_contour_shape(melody: List[int], target: str = 'arch') -> float:
    """Fitness: match a target melodic contour (arch, descent, ascent, wave)."""
    n = len(melody)
    if n < 3:
        return 0.0
    x = np.linspace(0, 1, n)
    normalized = (np.array(melody) - np.min(melody)) / (np.ptp(melody) + 1e-12)
    targets = {
        'arch': 4.0 * x * (1 - x),
        'descent': 1.0 - x,
        'ascent': x,
        'wave': 0.5 * (1 + np.sin(2 * np.pi * x)),
    }
    template = targets.get(target, targets['arch'])
    return float(1.0 - np.mean((normalized - template) ** 2))


# ---------------------------------------------------------------------------
# §10  Aggregate deep composition analysis
# ---------------------------------------------------------------------------

@dataclass
class DeepCompositionAnalysis:
    markov_entropy: float = 0.0
    fractal_dimension: float = 0.0
    tonal_fitness: float = 0.0
    contour_fitness: float = 0.0
    constraint_score: float = 0.0
    ca_density: float = 0.0

def analyze_composition_deep(
    melody: List[int],
    scale: Optional[List[int]] = None,
) -> DeepCompositionAnalysis:
    """Analyse a melody across deep compositional dimensions."""
    if scale is None:
        scale = [0, 2, 4, 5, 7, 9, 11]

    result = DeepCompositionAnalysis()

    mc = MarkovChain(max_order=3)
    mc.train(melody)
    result.markov_entropy = mc.cross_entropy(melody)

    result.tonal_fitness = fitness_tonal_coherence(melody, scale)
    result.contour_fitness = fitness_contour_shape(melody, 'arch')

    constraints = [
        ConstraintSpec('stepwise', check_stepwise_motion),
        ConstraintSpec('range', check_range),
        ConstraintSpec('scale', lambda m: check_scale_adherence(m, scale)),
    ]
    total_w = sum(c.weight for c in constraints)
    result.constraint_score = sum(
        c.weight * float(c.check(melody)) for c in constraints
    ) / max(total_w, 1e-12)

    if len(melody) > 1:
        intervals = np.abs(np.diff(melody))
        box_sizes = np.unique(np.geomspace(1, max(len(intervals) // 2, 2), 8).astype(int))
        box_sizes = box_sizes[box_sizes > 0]
        if len(box_sizes) >= 2:
            counts = []
            for bs in box_sizes:
                n_boxes = len(intervals) // bs
                if n_boxes == 0:
                    continue
                covered = sum(
                    1 for i in range(n_boxes) if np.any(intervals[i * bs:(i + 1) * bs] > 0)
                )
                counts.append(max(covered, 1))
            if len(counts) >= 2:
                log_s = np.log(1.0 / box_sizes[:len(counts)])
                log_n = np.log(np.array(counts, dtype=np.float64))
                if np.std(log_s) > 0:
                    result.fractal_dimension = float(
                        np.polyfit(log_s, log_n, 1)[0]
                    )

    return result
