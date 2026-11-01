"""Stochastic composition via Markov chains over pitch and rhythm."""
from __future__ import annotations
import random
from collections import defaultdict


class PitchMarkov:
    def __init__(self, order: int = 1):
        self.order = order
        self._transitions: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._trained = False

    def train(self, sequences: list[list[int]]) -> None:
        for seq in sequences:
            for i in range(self.order, len(seq)):
                ctx = tuple(seq[i - self.order : i])
                self._transitions[ctx][seq[i]] += 1
        self._trained = True

    def generate(self, seed: list[int], length: int, rng: random.Random | None = None) -> list[int]:
        if not self._trained:
            return seed[:length]
        rng = rng or random.Random()
        result = list(seed[-self.order :])
        for _ in range(length - len(result)):
            ctx = tuple(result[-self.order :])
            dist = self._transitions.get(ctx)
            if not dist:
                break
            total = sum(dist.values())
            r = rng.random() * total
            cumulative = 0.0
            chosen = list(dist.keys())[0]
            for pitch, count in dist.items():
                cumulative += count
                if cumulative >= r:
                    chosen = pitch
                    break
            result.append(chosen)
        return result

    def transition_probability(self, context: list[int], target: int) -> float:
        ctx = tuple(context[-self.order :])
        dist = self._transitions.get(ctx)
        if not dist:
            return 0.0
        total = sum(dist.values())
        return dist.get(target, 0) / total if total > 0 else 0.0


class RhythmMarkov:
    def __init__(self, order: int = 1):
        self.order = order
        self._transitions: dict[tuple, dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._trained = False

    def train(self, patterns: list[list[float]]) -> None:
        for pat in patterns:
            for i in range(self.order, len(pat)):
                ctx = tuple(pat[i - self.order : i])
                self._transitions[ctx][pat[i]] += 1
        self._trained = True

    def generate(self, seed: list[float], length: int, rng: random.Random | None = None) -> list[float]:
        if not self._trained:
            return seed[:length]
        rng = rng or random.Random()
        result = list(seed[-self.order :])
        for _ in range(length - len(result)):
            ctx = tuple(result[-self.order :])
            dist = self._transitions.get(ctx)
            if not dist:
                break
            total = sum(dist.values())
            r = rng.random() * total
            cumulative = 0.0
            chosen = list(dist.keys())[0]
            for dur, count in dist.items():
                cumulative += count
                if cumulative >= r:
                    chosen = dur
                    break
            result.append(chosen)
        return result


class MarkovComposer:
    def __init__(self, pitch_order: int = 1, rhythm_order: int = 1):
        self.pitch_chain = PitchMarkov(pitch_order)
        self.rhythm_chain = RhythmMarkov(rhythm_order)

    def train(
        self, pitch_sequences: list[list[int]], rhythm_patterns: list[list[float]],
    ) -> None:
        self.pitch_chain.train(pitch_sequences)
        self.rhythm_chain.train(rhythm_patterns)

    def compose(
        self,
        pitch_seed: list[int],
        rhythm_seed: list[float],
        length: int = 16,
        seed: int | None = None,
    ) -> list[tuple[int, float]]:
        rng = random.Random(seed) if seed is not None else random.Random()
        pitches = self.pitch_chain.generate(pitch_seed, length, rng)
        rhythms = self.rhythm_chain.generate(rhythm_seed, length, rng)
        min_len = min(len(pitches), len(rhythms))
        return list(zip(pitches[:min_len], rhythms[:min_len]))

    def entropy(self) -> dict[str, float]:
        import math
        result = {}
        for name, chain in [("pitch", self.pitch_chain), ("rhythm", self.rhythm_chain)]:
            h_values = []
            for ctx, dist in chain._transitions.items():
                total = sum(dist.values())
                if total == 0:
                    continue
                h = 0.0
                for count in dist.values():
                    p = count / total
                    if p > 0:
                        h -= p * math.log2(p)
                h_values.append(h)
            result[name] = sum(h_values) / len(h_values) if h_values else 0.0
        return result
