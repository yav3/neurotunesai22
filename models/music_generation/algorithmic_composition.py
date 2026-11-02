"""Algorithmic composition: L-systems, cellular automata, constraint satisfaction, stochastic."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable


@dataclass
class LSystemRule:
    predecessor: str
    successor: str
    probability: float = 1.0


@dataclass
class LSystem:
    axiom: str
    rules: List[LSystemRule]
    iterations: int = 4

    def generate(self) -> str:
        current = self.axiom
        for _ in range(self.iterations):
            new = []
            for ch in current:
                applied = False
                candidates = [r for r in self.rules if r.predecessor == ch]
                if candidates:
                    if len(candidates) == 1:
                        new.append(candidates[0].successor)
                    else:
                        probs = np.array([r.probability for r in candidates])
                        probs /= probs.sum()
                        idx = np.random.choice(len(candidates), p=probs)
                        new.append(candidates[idx].successor)
                    applied = True
                if not applied:
                    new.append(ch)
            current = "".join(new)
        return current


def lsystem_to_melody(lstring: str, base_pitch: int = 60,
                       mapping: Optional[Dict[str, int]] = None) -> np.ndarray:
    if mapping is None:
        mapping = {"F": 0, "+": 2, "-": -2, "[": 1, "]": -1, "A": 0, "B": 4}
    pitches = [base_pitch]
    current = base_pitch
    for ch in lstring:
        delta = mapping.get(ch, 0)
        current += delta
        current = np.clip(current, 36, 96)
        pitches.append(int(current))
    return np.array(pitches)


@dataclass
class CellularAutomaton:
    rule_number: int
    width: int = 64
    generations: int = 32

    def __post_init__(self):
        self._rule_table = np.array([(self.rule_number >> i) & 1 for i in range(8)])

    def run(self, initial: Optional[np.ndarray] = None) -> np.ndarray:
        if initial is None:
            state = np.zeros(self.width, dtype=int)
            state[self.width // 2] = 1
        else:
            state = initial.copy()
        history = np.zeros((self.generations, self.width), dtype=int)
        history[0] = state
        for g in range(1, self.generations):
            new_state = np.zeros(self.width, dtype=int)
            for i in range(self.width):
                left = state[(i - 1) % self.width]
                center = state[i]
                right = state[(i + 1) % self.width]
                idx = (left << 2) | (center << 1) | right
                new_state[i] = self._rule_table[idx]
            state = new_state
            history[g] = state
        return history


def ca_to_rhythm(history: np.ndarray, density: float = 0.5) -> np.ndarray:
    threshold = np.quantile(history.flatten(), 1.0 - density) if density < 1.0 else 0
    return (history >= threshold).astype(int)


def ca_to_pitches(history: np.ndarray, scale: np.ndarray,
                   base_octave: int = 60) -> np.ndarray:
    n_gen, width = history.shape
    pitches = np.zeros((n_gen, width), dtype=int)
    for g in range(n_gen):
        for i in range(width):
            if history[g, i]:
                col_sum = np.sum(history[:g + 1, i])
                pitch_idx = col_sum % len(scale)
                pitches[g, i] = base_octave + scale[pitch_idx]
    return pitches


@dataclass
class MarkovChainComposer:
    order: int = 1
    _transitions: Dict = field(default_factory=dict)

    def train(self, sequence: np.ndarray) -> None:
        for i in range(len(sequence) - self.order):
            state = tuple(sequence[i:i + self.order])
            next_val = sequence[i + self.order]
            if state not in self._transitions:
                self._transitions[state] = {}
            self._transitions[state][next_val] = self._transitions[state].get(next_val, 0) + 1

    def generate(self, length: int, seed: Optional[Tuple] = None) -> np.ndarray:
        if not self._transitions:
            return np.zeros(length)
        if seed is None:
            seed = list(self._transitions.keys())[0]
        output = list(seed)
        for _ in range(length - self.order):
            state = tuple(output[-self.order:])
            if state in self._transitions:
                nexts = self._transitions[state]
                vals = list(nexts.keys())
                counts = np.array(list(nexts.values()), dtype=float)
                probs = counts / counts.sum()
                output.append(np.random.choice(vals, p=probs))
            else:
                all_states = list(self._transitions.keys())
                state = all_states[np.random.randint(len(all_states))]
                output.append(state[-1])
        return np.array(output[:length])


@dataclass
class ConstraintSolver:
    pitch_range: Tuple[int, int] = (48, 84)
    allowed_intervals: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7, 12])
    max_leap: int = 12
    scale_degrees: Optional[np.ndarray] = None

    def solve(self, length: int, fixed: Optional[Dict[int, int]] = None) -> np.ndarray:
        melody = np.zeros(length, dtype=int)
        low, high = self.pitch_range
        if fixed:
            for idx, pitch in fixed.items():
                melody[idx] = pitch
        if melody[0] == 0:
            if self.scale_degrees is not None:
                base = (low + high) // 2
                octave = base - base % 12
                candidates = [octave + d for d in self.scale_degrees
                              if low <= octave + d <= high]
                melody[0] = candidates[len(candidates) // 2] if candidates else base
            else:
                melody[0] = (low + high) // 2
        for i in range(1, length):
            if melody[i] != 0:
                continue
            prev = melody[i - 1]
            candidates = []
            for iv in self.allowed_intervals:
                for direction in [1, -1]:
                    p = prev + direction * iv
                    if low <= p <= high and abs(p - prev) <= self.max_leap:
                        if self.scale_degrees is not None:
                            pc = p % 12
                            if pc in self.scale_degrees:
                                candidates.append(p)
                        else:
                            candidates.append(p)
            if candidates:
                melody[i] = candidates[np.random.randint(len(candidates))]
            else:
                melody[i] = prev
        return melody


def stochastic_process(length: int, drift: float = 0.0,
                       volatility: float = 1.0, mean_revert: float = 0.0,
                       start: float = 60.0) -> np.ndarray:
    values = np.zeros(length)
    values[0] = start
    for i in range(1, length):
        noise = np.random.randn() * volatility
        reversion = mean_revert * (start - values[i - 1])
        values[i] = values[i - 1] + drift + reversion + noise
    return values


def stochastic_to_pitches(values: np.ndarray, scale: np.ndarray,
                           base: int = 60) -> np.ndarray:
    quantized = np.round(values).astype(int)
    pitches = np.zeros(len(quantized), dtype=int)
    for i, v in enumerate(quantized):
        octave = (v - base) // 12
        pc = scale[int(abs(v - base)) % len(scale)]
        pitches[i] = base + octave * 12 + pc
    return np.clip(pitches, 36, 96)


def euclidean_rhythm(pulses: int, steps: int) -> np.ndarray:
    if steps <= 0:
        return np.array([], dtype=int)
    if pulses <= 0:
        return np.zeros(steps, dtype=int)
    pulses = min(pulses, steps)
    pattern = np.zeros(steps, dtype=int)
    bucket = 0
    for i in range(steps):
        bucket += pulses
        if bucket >= steps:
            bucket -= steps
            pattern[i] = 1
    return pattern


def serial_row(seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    row = rng.permutation(12)
    return row


def serial_transformations(row: np.ndarray) -> Dict[str, np.ndarray]:
    P0 = row.copy()
    R0 = row[::-1].copy()
    I0 = (12 - row) % 12
    RI0 = I0[::-1].copy()
    return {"P0": P0, "R0": R0, "I0": I0, "RI0": RI0}


def all_transpositions(row: np.ndarray) -> Dict[str, np.ndarray]:
    forms = serial_transformations(row)
    all_forms = {}
    for name, form in forms.items():
        prefix = name[:-1]
        for t in range(12):
            all_forms[f"{prefix}{t}"] = (form + t) % 12
    return all_forms


def fractal_melody(depth: int = 4, base_intervals: Optional[np.ndarray] = None,
                    base_pitch: int = 60) -> np.ndarray:
    if base_intervals is None:
        base_intervals = np.array([0, 2, -1, 3])
    sequence = base_intervals.copy().astype(float)
    for _ in range(depth - 1):
        new_seq = []
        for iv in sequence:
            scaled = base_intervals * (iv / (np.max(np.abs(base_intervals)) + 1e-12))
            new_seq.extend(scaled)
        sequence = np.array(new_seq)
    pitches = np.cumsum(sequence) + base_pitch
    return np.clip(pitches, 36, 96).astype(int)
