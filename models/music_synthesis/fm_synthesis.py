"""FM synthesis: Chowning algorithm, DX7-style operator graphs, patches."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ADSREnvelope:
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.2

    def generate(self, duration_s: float, sr: int = 44100) -> np.ndarray:
        n = int(sr * duration_s)
        if n == 0:
            return np.array([], dtype=np.float64)
        env = np.zeros(n, dtype=np.float64)
        a_samp = int(sr * self.attack)
        d_samp = int(sr * self.decay)
        r_samp = int(sr * self.release)
        sustain_end = max(0, n - r_samp)

        for i in range(n):
            if i < a_samp:
                env[i] = i / max(a_samp, 1)
            elif i < a_samp + d_samp:
                frac = (i - a_samp) / max(d_samp, 1)
                env[i] = 1.0 - frac * (1.0 - self.sustain)
            elif i < sustain_end:
                env[i] = self.sustain
            else:
                frac = (i - sustain_end) / max(r_samp, 1)
                env[i] = self.sustain * (1.0 - min(frac, 1.0))
        return env


@dataclass
class FMOperator:
    ratio: float = 1.0
    index: float = 1.0
    amplitude: float = 1.0
    envelope: ADSREnvelope = field(default_factory=ADSREnvelope)
    feedback: float = 0.0

    def render(
        self,
        base_freq: float,
        duration_s: float,
        sr: int = 44100,
        modulator_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = int(sr * duration_s)
        t = np.arange(n, dtype=np.float64) / sr
        freq = base_freq * self.ratio
        env = self.envelope.generate(duration_s, sr)
        if len(env) < n:
            env = np.pad(env, (0, n - len(env)))
        elif len(env) > n:
            env = env[:n]

        phase = 2.0 * np.pi * freq * t

        if modulator_signal is not None:
            mod = modulator_signal
            if len(mod) < n:
                mod = np.pad(mod, (0, n - len(mod)))
            elif len(mod) > n:
                mod = mod[:n]
            phase = phase + self.index * mod

        if self.feedback > 0:
            out = np.zeros(n, dtype=np.float64)
            prev = 0.0
            for i in range(n):
                out[i] = np.sin(phase[i] + self.feedback * prev)
                prev = out[i]
            return (self.amplitude * env * out).astype(np.float64)

        return (self.amplitude * env * np.sin(phase)).astype(np.float64)


@dataclass
class FMAlgorithm:
    n_operators: int = 6
    connections: dict[int, list[int]] = field(default_factory=dict)
    carriers: list[int] = field(default_factory=lambda: [0])

    @classmethod
    def simple_two_op(cls) -> FMAlgorithm:
        return cls(n_operators=2, connections={0: [1]}, carriers=[0])

    @classmethod
    def dx7_algorithm1(cls) -> FMAlgorithm:
        return cls(
            n_operators=6,
            connections={0: [1], 1: [2], 3: [4], 4: [5]},
            carriers=[0, 3],
        )

    @classmethod
    def dx7_algorithm5(cls) -> FMAlgorithm:
        return cls(
            n_operators=6,
            connections={0: [1], 1: [2], 2: [3], 3: [4], 4: [5]},
            carriers=[0],
        )

    @classmethod
    def parallel(cls, n: int = 4) -> FMAlgorithm:
        return cls(n_operators=n, connections={}, carriers=list(range(n)))


@dataclass
class FMPatch:
    name: str = "default"
    algorithm: FMAlgorithm = field(default_factory=FMAlgorithm.simple_two_op)
    operators: list[FMOperator] = field(default_factory=list)

    def __post_init__(self):
        while len(self.operators) < self.algorithm.n_operators:
            self.operators.append(FMOperator())

    @classmethod
    def electric_piano(cls) -> FMPatch:
        algo = FMAlgorithm.simple_two_op()
        ops = [
            FMOperator(ratio=1.0, index=0.0, amplitude=1.0,
                       envelope=ADSREnvelope(0.005, 0.8, 0.0, 0.3)),
            FMOperator(ratio=14.0, index=2.5, amplitude=1.0,
                       envelope=ADSREnvelope(0.001, 0.5, 0.0, 0.2)),
        ]
        return cls(name="electric_piano", algorithm=algo, operators=ops)

    @classmethod
    def brass(cls) -> FMPatch:
        algo = FMAlgorithm.simple_two_op()
        ops = [
            FMOperator(ratio=1.0, index=0.0, amplitude=1.0,
                       envelope=ADSREnvelope(0.05, 0.2, 0.8, 0.15)),
            FMOperator(ratio=1.0, index=5.0, amplitude=1.0,
                       envelope=ADSREnvelope(0.08, 0.3, 0.6, 0.15)),
        ]
        return cls(name="brass", algorithm=algo, operators=ops)

    @classmethod
    def bell(cls) -> FMPatch:
        algo = FMAlgorithm.simple_two_op()
        ops = [
            FMOperator(ratio=1.0, index=0.0, amplitude=1.0,
                       envelope=ADSREnvelope(0.001, 2.0, 0.0, 0.5)),
            FMOperator(ratio=3.5, index=8.0, amplitude=1.0,
                       envelope=ADSREnvelope(0.001, 4.0, 0.0, 1.0)),
        ]
        return cls(name="bell", algorithm=algo, operators=ops)


class FMSynthesizer:
    def __init__(self, patch: Optional[FMPatch] = None, sr: int = 44100):
        self.patch = patch or FMPatch()
        self.sr = sr

    def render_note(self, freq: float, duration_s: float, velocity: float = 1.0) -> np.ndarray:
        algo = self.patch.algorithm
        ops = self.patch.operators
        n = int(self.sr * duration_s)
        signals: dict[int, np.ndarray] = {}

        order = self._topological_order(algo)

        for op_idx in order:
            op = ops[op_idx] if op_idx < len(ops) else FMOperator()
            modulators = algo.connections.get(op_idx, [])
            mod_sum = np.zeros(n, dtype=np.float64)
            for m_idx in modulators:
                if m_idx in signals:
                    mod_sum += signals[m_idx]
            mod_input = mod_sum if np.any(mod_sum != 0) else None
            signals[op_idx] = op.render(freq, duration_s, self.sr, mod_input)

        out = np.zeros(n, dtype=np.float64)
        for c in algo.carriers:
            if c in signals:
                out += signals[c]

        out *= velocity
        mx = np.max(np.abs(out))
        if mx > 1.0:
            out /= mx
        return out.astype(np.float32)

    @staticmethod
    def _topological_order(algo: FMAlgorithm) -> list[int]:
        visited: set[int] = set()
        order: list[int] = []

        def dfs(node: int):
            if node in visited:
                return
            visited.add(node)
            for dep in algo.connections.get(node, []):
                dfs(dep)
            order.append(node)

        for i in range(algo.n_operators):
            dfs(i)
        return list(reversed(order))
