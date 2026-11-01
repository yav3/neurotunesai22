"""Algorithmic composition: L-systems, cellular automata, stochastic processes.

Mathematical generative systems that produce musical sequences from
formal rules rather than statistical learning.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum


# ---------------------------------------------------------------------------
# L-system melody generation
# ---------------------------------------------------------------------------

@dataclass
class LSystemRule:
    """Production rule for an L-system."""
    predecessor: str
    successor: str
    probability: float = 1.0


@dataclass
class LSystem:
    """Lindenmayer system for generating symbolic sequences."""
    axiom: str
    rules: List[LSystemRule]
    _rule_map: Dict[str, List[LSystemRule]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for rule in self.rules:
            self._rule_map.setdefault(rule.predecessor, []).append(rule)

    def generate(self, iterations: int, rng: Optional[np.random.RandomState] = None) -> str:
        if rng is None:
            rng = np.random.RandomState(0)
        current = self.axiom
        for _ in range(iterations):
            result = []
            for ch in current:
                candidates = self._rule_map.get(ch, [])
                if not candidates:
                    result.append(ch)
                    continue
                probs = np.array([r.probability for r in candidates])
                probs /= probs.sum()
                chosen = candidates[rng.choice(len(candidates), p=probs)]
                result.append(chosen.successor)
            current = "".join(result)
        return current


def lsystem_to_pitches(sequence: str, pitch_map: Optional[Dict[str, int]] = None,
                       base_pitch: int = 60) -> List[int]:
    """Convert L-system output to MIDI pitches via a symbol-to-interval map."""
    if pitch_map is None:
        pitch_map = {
            "A": 0, "B": 2, "C": 4, "D": 5, "E": 7, "F": 9, "G": 11,
            "+": 1, "-": -1, "[": 0, "]": 0,
        }
    pitches = []
    current = base_pitch
    stack = []
    for ch in sequence:
        if ch == "[":
            stack.append(current)
        elif ch == "]" and stack:
            current = stack.pop()
        elif ch in pitch_map:
            current += pitch_map[ch]
            current = max(21, min(108, current))  # piano range
            pitches.append(current)
    return pitches


# ---------------------------------------------------------------------------
# Cellular automaton rhythm / melody
# ---------------------------------------------------------------------------

class CARule(Enum):
    """Elementary cellular automaton rules (Wolfram numbering)."""
    RULE_30 = 30
    RULE_90 = 90
    RULE_110 = 110
    RULE_150 = 150


def elementary_ca(width: int, steps: int, rule: int = 110,
                  initial: Optional[np.ndarray] = None) -> np.ndarray:
    """Run an elementary cellular automaton.

    Returns (steps, width) binary grid.  Each row is one generation.
    """
    grid = np.zeros((steps, width), dtype=np.int8)
    if initial is not None:
        grid[0, :len(initial)] = initial[:width]
    else:
        grid[0, width // 2] = 1
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int8)
    for t in range(1, steps):
        for x in range(width):
            left = grid[t-1, (x-1) % width]
            center = grid[t-1, x]
            right = grid[t-1, (x+1) % width]
            idx = (left << 2) | (center << 1) | right
            grid[t, x] = rule_bits[idx]
    return grid


def ca_to_rhythm(grid: np.ndarray) -> List[np.ndarray]:
    """Convert CA grid rows to rhythm patterns (1 = onset, 0 = rest)."""
    return [grid[t].copy() for t in range(grid.shape[0])]


def ca_to_pitches(grid: np.ndarray, scale: np.ndarray,
                  base_pitch: int = 48) -> np.ndarray:
    """Map CA columns to scale degrees, producing a pitch sequence per row.

    For each row, active cells select from the scale; the pitch sequence
    is the set of scale degrees where cell == 1.
    """
    n_steps, width = grid.shape
    n_scale = len(scale)
    sequences = []
    for t in range(n_steps):
        row_pitches = []
        for x in range(width):
            if grid[t, x]:
                degree = x % n_scale
                octave = x // n_scale
                row_pitches.append(base_pitch + scale[degree] + 12 * octave)
        sequences.append(np.array(row_pitches, dtype=int))
    return sequences


# ---------------------------------------------------------------------------
# Stochastic processes
# ---------------------------------------------------------------------------

def brownian_melody(n_notes: int, scale: np.ndarray, base_pitch: int = 60,
                    step_std: float = 1.5,
                    rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Brownian-motion melody constrained to a scale.

    At each step, pitch moves by a Gaussian increment quantized
    to the nearest scale degree.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    pitches = np.zeros(n_notes, dtype=int)
    pitches[0] = base_pitch
    full_scale = []
    for octave in range(-2, 8):
        for s in scale:
            full_scale.append(base_pitch + s + 12 * octave)
    full_scale = np.array(sorted(set(full_scale)))
    full_scale = full_scale[(full_scale >= 21) & (full_scale <= 108)]
    current_idx = np.argmin(np.abs(full_scale - base_pitch))
    for i in range(1, n_notes):
        step = rng.normal(0, step_std)
        current_idx = int(np.clip(current_idx + round(step),
                                  0, len(full_scale) - 1))
        pitches[i] = full_scale[current_idx]
    return pitches


def ornstein_uhlenbeck_melody(n_notes: int, scale: np.ndarray,
                              base_pitch: int = 60, theta: float = 0.3,
                              sigma: float = 2.0, mu: float = 0.0,
                              rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Mean-reverting (Ornstein-Uhlenbeck) melody on a scale.

    dx = theta * (mu - x) * dt + sigma * dW
    Produces melodies that wander but return to a central pitch.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    full_scale = []
    for octave in range(-2, 8):
        for s in scale:
            full_scale.append(base_pitch + s + 12 * octave)
    full_scale = np.array(sorted(set(full_scale)))
    full_scale = full_scale[(full_scale >= 21) & (full_scale <= 108)]
    x = 0.0
    pitches = np.zeros(n_notes, dtype=int)
    center_idx = np.argmin(np.abs(full_scale - base_pitch))
    for i in range(n_notes):
        idx = int(np.clip(center_idx + round(x), 0, len(full_scale) - 1))
        pitches[i] = full_scale[idx]
        dw = rng.normal(0, 1)
        x += theta * (mu - x) + sigma * dw
    return pitches


def markov_rhythm(transition_matrix: np.ndarray, n_steps: int,
                  initial_state: int = 0,
                  rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Generate rhythm pattern via Markov chain.

    States represent different rhythmic densities or patterns.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    n_states = transition_matrix.shape[0]
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    P = transition_matrix / row_sums
    states = np.zeros(n_steps, dtype=int)
    states[0] = initial_state % n_states
    for i in range(1, n_steps):
        states[i] = rng.choice(n_states, p=P[states[i-1]])
    return states


# ---------------------------------------------------------------------------
# Fractal music
# ---------------------------------------------------------------------------

def cantor_rhythm(levels: int, beat_duration: float = 1.0) -> List[Tuple[float, float]]:
    """Cantor set rhythm: recursive removal of middle third.

    Returns list of (onset_time, duration) pairs.
    """
    if levels == 0:
        return [(0.0, beat_duration)]
    sub = cantor_rhythm(levels - 1, beat_duration / 3)
    result = []
    for onset, dur in sub:
        result.append((onset, dur))
        result.append((onset + 2 * beat_duration / 3, dur))
    return result


def sierpinski_rhythm(levels: int, n_beats: int = 16) -> np.ndarray:
    """Sierpinski triangle mapped to rhythm: Pascal's triangle mod 2."""
    pattern = np.zeros((levels, n_beats), dtype=int)
    for row in range(levels):
        val = 1
        for col in range(min(row + 1, n_beats)):
            pattern[row, col] = val % 2
            if col < row:
                val = val * (row - col) // (col + 1)
    return pattern


def fibonacci_durations(n_notes: int, base_duration: float = 0.25) -> np.ndarray:
    """Fibonacci-proportioned durations.

    Duration ratios follow the Fibonacci sequence: 1, 1, 2, 3, 5, 8, ...
    normalized and cycled to produce n_notes durations.
    """
    fibs = [1, 1]
    while len(fibs) < n_notes:
        fibs.append(fibs[-1] + fibs[-2])
    fibs = np.array(fibs[:n_notes], dtype=float)
    fibs /= fibs.sum()
    return fibs * n_notes * base_duration


def golden_ratio_pitches(n_notes: int, scale: np.ndarray,
                         base_pitch: int = 60) -> np.ndarray:
    """Pitch sequence where intervals approximate the golden ratio.

    Uses Phi = (1 + sqrt(5)) / 2 to space scale degrees.
    """
    phi = (1 + np.sqrt(5)) / 2
    full_scale = []
    for octave in range(-1, 6):
        for s in scale:
            full_scale.append(base_pitch + s + 12 * octave)
    full_scale = np.array(sorted(set(full_scale)))
    full_scale = full_scale[(full_scale >= 21) & (full_scale <= 108)]
    pitches = np.zeros(n_notes, dtype=int)
    idx = len(full_scale) // 2
    for i in range(n_notes):
        pitches[i] = full_scale[int(idx) % len(full_scale)]
        idx += phi
        if idx >= len(full_scale):
            idx -= len(full_scale)
    return pitches
