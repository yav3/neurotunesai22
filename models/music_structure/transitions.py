"""Section transition analysis: type classification, energy profiles, flow graphs."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TransitionType(str, Enum):
    DIRECT_CUT = "direct_cut"
    CROSSFADE = "crossfade"
    FILL = "fill"
    BREAKDOWN = "breakdown"
    BUILDUP = "buildup"
    UNKNOWN = "unknown"


@dataclass
class Transition:
    """A detected transition between musical sections."""
    frame: int
    from_label: str
    to_label: str
    strength: float
    duration_frames: int = 1


@dataclass
class TransitionInfo:
    boundary_idx: int
    transition_type: TransitionType
    energy_change: float
    spectral_contrast: float


def energy_profile(features: np.ndarray, window_size: int = 8) -> np.ndarray:
    n = features.shape[0]
    if n == 0:
        return np.array([])
    energy = np.sum(features ** 2, axis=1)
    if n < window_size:
        return energy
    smoothed = np.convolve(energy, np.ones(window_size) / window_size, mode="same")
    return smoothed


def spectral_contrast_at_boundary(
    features: np.ndarray,
    boundary_idx: int,
    context: int = 8,
) -> float:
    n = features.shape[0]
    start_before = max(0, boundary_idx - context)
    end_after = min(n, boundary_idx + context)
    before = features[start_before:boundary_idx]
    after = features[boundary_idx:end_after]
    if len(before) == 0 or len(after) == 0:
        return 0.0
    mean_before = before.mean(axis=0)
    mean_after = after.mean(axis=0)
    return float(np.linalg.norm(mean_after - mean_before))


class TransitionDetector:
    """Classify transition types based on energy and spectral features."""

    def __init__(self, context_frames: int = 16, smoothing: int = 3) -> None:
        self.context_frames = context_frames
        self.smoothing = smoothing

    def detect_transition_type(
        self,
        features: np.ndarray,
        boundary_idx: int,
    ) -> TransitionType:
        n = features.shape[0]
        half = self.context_frames // 2
        start = max(0, boundary_idx - half)
        end = min(n, boundary_idx + half)
        before = features[start:boundary_idx]
        after = features[boundary_idx:end]
        if len(before) == 0 or len(after) == 0:
            return TransitionType.UNKNOWN

        energy_before = np.mean(np.sum(before ** 2, axis=1))
        energy_after = np.mean(np.sum(after ** 2, axis=1))
        if energy_before < 1e-10 and energy_after < 1e-10:
            return TransitionType.UNKNOWN

        ratio = energy_after / max(energy_before, 1e-10)
        spectral_diff = spectral_contrast_at_boundary(features, boundary_idx, half)

        if ratio < 0.2:
            return TransitionType.BREAKDOWN
        elif ratio > 3.0:
            return TransitionType.BUILDUP
        elif spectral_diff < 0.5 and 0.5 < ratio < 2.0:
            return TransitionType.CROSSFADE
        elif spectral_diff > 2.0:
            return TransitionType.DIRECT_CUT
        else:
            return TransitionType.DIRECT_CUT

    def analyze_all_transitions(
        self,
        features: np.ndarray,
        boundaries: list[int],
    ) -> list[TransitionInfo]:
        results = []
        for b in boundaries[1:]:
            tt = self.detect_transition_type(features, b)
            ep = energy_profile(features, window_size=8)
            half = self.context_frames // 2
            e_before = float(ep[max(0, b - 1)])
            e_after = float(ep[min(len(ep) - 1, b)])
            sc = spectral_contrast_at_boundary(features, b, half)
            results.append(
                TransitionInfo(
                    boundary_idx=b,
                    transition_type=tt,
                    energy_change=e_after - e_before,
                    spectral_contrast=sc,
                )
            )
        return results

    def detect(
        self,
        novelty: np.ndarray,
        labels: list[str],
        boundaries: list[int],
    ) -> list[Transition]:
        if len(boundaries) < 2 or len(labels) < 2:
            return []
        transitions = []
        for i in range(len(boundaries) - 1):
            b = boundaries[i + 1] if i + 1 < len(boundaries) else len(novelty) - 1
            if b >= len(novelty):
                b = len(novelty) - 1
            half = self.smoothing
            lo = max(0, b - half)
            hi = min(len(novelty), b + half + 1)
            strength = float(np.mean(novelty[lo:hi])) if hi > lo else 0.0
            from_label = labels[i] if i < len(labels) else "?"
            to_label = labels[i + 1] if i + 1 < len(labels) else "?"
            transitions.append(
                Transition(frame=b, from_label=from_label, to_label=to_label,
                           strength=strength, duration_frames=hi - lo)
            )
        return transitions

    def smoothness_score(self, transitions: list[Transition]) -> float:
        if not transitions:
            return 1.0
        return 1.0 - float(np.mean([t.strength for t in transitions]))


class TransitionGraph:
    """Directed graph of section transitions with Markov probabilities."""

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, list[TransitionType]]] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._labels: set[str] = set()

    def add_transition(
        self,
        from_label: str,
        to_label: str,
        transition_type: TransitionType | None = None,
    ) -> None:
        if transition_type is not None:
            self._edges.setdefault(from_label, {}).setdefault(to_label, []).append(
                transition_type
            )
        self._counts[(from_label, to_label)] = self._counts.get((from_label, to_label), 0) + 1
        self._labels.add(from_label)
        self._labels.add(to_label)

    def add_sequence(self, labels: list[str]) -> None:
        for i in range(len(labels) - 1):
            self.add_transition(labels[i], labels[i + 1])

    def build_graph(
        self,
        section_labels: list[str],
        transitions: list[TransitionInfo],
    ) -> dict[str, dict[str, list[str]]]:
        for i, ti in enumerate(transitions):
            if i + 1 < len(section_labels):
                self.add_transition(section_labels[i], section_labels[i + 1], ti.transition_type)
        result: dict[str, dict[str, list[str]]] = {}
        for src, targets in self._edges.items():
            result[src] = {}
            for tgt, types in targets.items():
                result[src][tgt] = [t.value for t in types]
        return result

    @property
    def nodes(self) -> set[str]:
        nodes = set(self._labels)
        for targets in self._edges.values():
            nodes.update(targets.keys())
        return nodes

    @property
    def n_edges(self) -> int:
        return sum(v for v in self._counts.values())

    def probability(self, from_label: str, to_label: str) -> float:
        total = sum(v for (f, _), v in self._counts.items() if f == from_label)
        if total == 0:
            return 0.0
        return self._counts.get((from_label, to_label), 0) / total

    def transition_matrix(self) -> np.ndarray:
        ordered = sorted(self._labels)
        n = len(ordered)
        mat = np.zeros((n, n))
        idx = {label: i for i, label in enumerate(ordered)}
        for (f, t), count in self._counts.items():
            mat[idx[f], idx[t]] = count
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        return mat / row_sums

    def entropy(self) -> float:
        mat = self.transition_matrix()
        total = 0.0
        n_rows = 0
        for i in range(mat.shape[0]):
            row = mat[i]
            row = row[row > 0]
            if len(row) > 0:
                total += float(-np.sum(row * np.log2(row)))
                n_rows += 1
        return total / max(n_rows, 1)

    def most_likely_next(self, from_label: str) -> str | None:
        best_label = None
        best_prob = 0.0
        for (f, t), count in self._counts.items():
            if f == from_label:
                p = self.probability(f, t)
                if p > best_prob:
                    best_prob = p
                    best_label = t
        return best_label
