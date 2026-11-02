"""Cellular automata music generation: Wolfram rules, Game of Life, L-systems.

Maps deterministic and stochastic cellular automata evolutions to musical
parameters. Includes 1D elementary CA (Wolfram), 2D Game of Life, and
L-system string rewriting with turtle-to-music interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Rule1D:
    """Elementary 1D cellular automaton (Wolfram rules 0-255)."""

    def __init__(self, rule_number: int = 110) -> None:
        self.rule_number = rule_number % 256
        self._table = np.array(
            [(rule_number >> i) & 1 for i in range(8)], dtype=np.int8
        )

    def _neighborhood_index(self, left: int, center: int, right: int) -> int:
        return (left << 2) | (center << 1) | right

    def step(self, state: np.ndarray) -> np.ndarray:
        n = len(state)
        new = np.zeros(n, dtype=np.int8)
        for i in range(n):
            l = state[(i - 1) % n]
            c = state[i]
            r = state[(i + 1) % n]
            new[i] = self._table[self._neighborhood_index(l, c, r)]
        return new

    def evolve(self, initial: np.ndarray, steps: int) -> np.ndarray:
        grid = np.zeros((steps + 1, len(initial)), dtype=np.int8)
        grid[0] = initial
        for t in range(steps):
            grid[t + 1] = self.step(grid[t])
        return grid

    def to_melody(self, grid: np.ndarray, scale: list[int] | None = None) -> list[int]:
        if scale is None:
            scale = [60, 62, 64, 65, 67, 69, 71, 72]
        pitches = []
        for row in grid:
            active = np.where(row > 0)[0]
            if len(active) > 0:
                idx = int(np.mean(active)) % len(scale)
                pitches.append(scale[idx])
            else:
                pitches.append(0)
        return pitches

    def to_rhythm(self, grid: np.ndarray) -> list[int]:
        return [int(np.sum(row) > 0) for row in grid]


class GameOfLifeMusic:
    """Conway's Game of Life mapped to polyphonic music."""

    def __init__(self, width: int = 16, height: int = 16) -> None:
        self.width = width
        self.height = height

    def random_grid(self, density: float = 0.3,
                    rng: np.random.RandomState | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.RandomState()
        return (rng.rand(self.height, self.width) < density).astype(np.int8)

    def step(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        new = np.zeros_like(grid)
        for y in range(h):
            for x in range(w):
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = (y + dy) % h, (x + dx) % w
                        neighbors += grid[ny, nx]
                if grid[y, x] == 1:
                    new[y, x] = 1 if neighbors in (2, 3) else 0
                else:
                    new[y, x] = 1 if neighbors == 3 else 0
        return new

    def evolve(self, initial: np.ndarray, steps: int) -> list[np.ndarray]:
        history = [initial.copy()]
        current = initial.copy()
        for _ in range(steps):
            current = self.step(current)
            history.append(current.copy())
        return history

    def map_to_pitches(self, grid: np.ndarray, scale: list[int] | None = None
                       ) -> list[int]:
        if scale is None:
            scale = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74]
        pitches = []
        h, _ = grid.shape
        for y in range(h):
            active = np.where(grid[y] > 0)[0]
            for x in active:
                pitches.append(scale[x % len(scale)])
        return pitches

    def density_curve(self, history: list[np.ndarray]) -> np.ndarray:
        return np.array([np.sum(g) / g.size for g in history])


@dataclass
class LSystemRule:
    predecessor: str
    successor: str


class LindenmayerMusic:
    """L-system string rewriting with turtle-to-music interpretation."""

    def __init__(self, axiom: str, rules: list[LSystemRule]) -> None:
        self.axiom = axiom
        self._rules = {r.predecessor: r.successor for r in rules}

    def expand(self, iterations: int) -> str:
        current = self.axiom
        for _ in range(iterations):
            result = []
            for ch in current:
                result.append(self._rules.get(ch, ch))
            current = "".join(result)
        return current

    def turtle_to_melody(self, string: str, base_pitch: int = 60,
                         step_interval: int = 2, angle_interval: int = 5,
                         duration: float = 0.25) -> list[tuple[int, float]]:
        pitch = base_pitch
        stack: list[int] = []
        melody: list[tuple[int, float]] = []
        for ch in string:
            if ch == "F":
                melody.append((pitch, duration))
                pitch += step_interval
            elif ch == "f":
                pitch += step_interval
            elif ch == "+":
                pitch += angle_interval
            elif ch == "-":
                pitch -= angle_interval
            elif ch == "[":
                stack.append(pitch)
            elif ch == "]":
                if stack:
                    pitch = stack.pop()
        return melody

    @staticmethod
    def sierpinski() -> LindenmayerMusic:
        return LindenmayerMusic("A", [
            LSystemRule("A", "B-A-B"),
            LSystemRule("B", "A+B+A"),
        ])

    @staticmethod
    def koch() -> LindenmayerMusic:
        return LindenmayerMusic("F", [
            LSystemRule("F", "F+F-F-F+F"),
        ])

    @staticmethod
    def dragon_curve() -> LindenmayerMusic:
        return LindenmayerMusic("FX", [
            LSystemRule("X", "X+YF+"),
            LSystemRule("Y", "-FX-Y"),
        ])


class StochasticCA:
    """Probabilistic 1D cellular automaton with temperature control."""

    def __init__(self, width: int = 32, n_states: int = 2) -> None:
        self.width = width
        self.n_states = n_states

    def step(self, state: np.ndarray, temperature: float = 0.1,
             rng: np.random.RandomState | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.RandomState()
        n = len(state)
        new = np.zeros(n, dtype=np.int8)
        for i in range(n):
            l = state[(i - 1) % n]
            c = state[i]
            r = state[(i + 1) % n]
            neighbor_sum = l + c + r
            probs = np.zeros(self.n_states)
            for s in range(self.n_states):
                energy = abs(s * 3 - neighbor_sum)
                probs[s] = np.exp(-energy / max(temperature, 1e-6))
            probs /= probs.sum()
            new[i] = rng.choice(self.n_states, p=probs)
        return new

    def evolve(self, initial: np.ndarray, steps: int,
               temperature: float = 0.1,
               rng: np.random.RandomState | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.RandomState()
        grid = np.zeros((steps + 1, len(initial)), dtype=np.int8)
        grid[0] = initial
        for t in range(steps):
            grid[t + 1] = self.step(grid[t], temperature, rng)
        return grid
