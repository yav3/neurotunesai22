"""Hierarchical mood taxonomy with VAD coordinates and musical-cue mapping."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class MoodNode:
    name: str
    valence: float
    arousal: float
    dominance: float
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    musical_cues: dict[str, float] = field(default_factory=dict)


MOOD_TREE: dict[str, MoodNode] = {}

_MOOD_DEFS: list[tuple[str, float, float, float, str | None, dict[str, float]]] = [
    ("happy", 0.9, 0.7, 0.7, None, {"tempo": 130, "mode": 1.0, "brightness": 0.8}),
    ("joyful", 0.95, 0.8, 0.75, "happy", {"tempo": 140, "mode": 1.0, "brightness": 0.9}),
    ("euphoric", 0.95, 0.95, 0.8, "happy", {"tempo": 150, "mode": 1.0, "brightness": 0.95}),
    ("content", 0.8, 0.3, 0.6, "happy", {"tempo": 95, "mode": 1.0, "brightness": 0.6}),
    ("sad", 0.15, 0.3, 0.25, None, {"tempo": 70, "mode": 0.0, "brightness": 0.3}),
    ("melancholic", 0.2, 0.25, 0.3, "sad", {"tempo": 65, "mode": 0.0, "brightness": 0.25}),
    ("grieving", 0.05, 0.4, 0.15, "sad", {"tempo": 55, "mode": 0.0, "brightness": 0.2}),
    ("nostalgic", 0.4, 0.3, 0.35, "sad", {"tempo": 80, "mode": 0.3, "brightness": 0.4}),
    ("angry", 0.15, 0.85, 0.8, None, {"tempo": 145, "mode": 0.0, "brightness": 0.7}),
    ("furious", 0.05, 0.95, 0.9, "angry", {"tempo": 160, "mode": 0.0, "brightness": 0.8}),
    ("aggressive", 0.1, 0.9, 0.85, "angry", {"tempo": 155, "mode": 0.0, "brightness": 0.75}),
    ("fearful", 0.15, 0.8, 0.15, None, {"tempo": 110, "mode": 0.0, "brightness": 0.4}),
    ("anxious", 0.2, 0.75, 0.2, "fearful", {"tempo": 120, "mode": 0.1, "brightness": 0.45}),
    ("tense", 0.25, 0.7, 0.3, "fearful", {"tempo": 105, "mode": 0.2, "brightness": 0.5}),
    ("calm", 0.65, 0.15, 0.5, None, {"tempo": 75, "mode": 0.7, "brightness": 0.45}),
    ("peaceful", 0.7, 0.1, 0.55, "calm", {"tempo": 65, "mode": 0.8, "brightness": 0.4}),
    ("serene", 0.75, 0.08, 0.55, "calm", {"tempo": 60, "mode": 0.9, "brightness": 0.35}),
    ("energetic", 0.7, 0.9, 0.75, None, {"tempo": 140, "mode": 0.8, "brightness": 0.85}),
    ("excited", 0.8, 0.85, 0.7, "energetic", {"tempo": 135, "mode": 0.9, "brightness": 0.8}),
    ("triumphant", 0.85, 0.8, 0.9, "energetic", {"tempo": 125, "mode": 1.0, "brightness": 0.85}),
    ("mysterious", 0.4, 0.45, 0.4, None, {"tempo": 85, "mode": 0.3, "brightness": 0.35}),
    ("eerie", 0.25, 0.5, 0.3, "mysterious", {"tempo": 75, "mode": 0.1, "brightness": 0.25}),
    ("romantic", 0.8, 0.45, 0.55, None, {"tempo": 90, "mode": 0.8, "brightness": 0.55}),
    ("tender", 0.75, 0.25, 0.45, "romantic", {"tempo": 72, "mode": 0.9, "brightness": 0.45}),
]

for _name, _v, _a, _d, _parent, _cues in _MOOD_DEFS:
    MOOD_TREE[_name] = MoodNode(
        name=_name, valence=_v, arousal=_a, dominance=_d,
        parent=_parent, musical_cues=_cues,
    )
for _name, node in MOOD_TREE.items():
    if node.parent and node.parent in MOOD_TREE:
        MOOD_TREE[node.parent].children.append(_name)


def mood_distance(a: str, b: str) -> float:
    if a not in MOOD_TREE or b not in MOOD_TREE:
        return float("inf")
    ma, mb = MOOD_TREE[a], MOOD_TREE[b]
    return float(np.sqrt(
        (ma.valence - mb.valence) ** 2
        + (ma.arousal - mb.arousal) ** 2
        + (ma.dominance - mb.dominance) ** 2
    ))


def mood_ancestors(name: str) -> list[str]:
    chain: list[str] = []
    current = name
    seen: set[str] = set()
    while current in MOOD_TREE:
        parent = MOOD_TREE[current].parent
        if parent is None or parent in seen:
            break
        chain.append(parent)
        seen.add(parent)
        current = parent
    return chain


def mood_from_vad(valence: float, arousal: float, dominance: float) -> str:
    best_name = ""
    best_dist = float("inf")
    for name, node in MOOD_TREE.items():
        d = np.sqrt(
            (node.valence - valence) ** 2
            + (node.arousal - arousal) ** 2
            + (node.dominance - dominance) ** 2
        )
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def mood_from_features(tempo: float, mode: float, brightness: float) -> str:
    query = np.array([tempo, mode, brightness])
    best_name = ""
    best_dist = float("inf")
    for name, node in MOOD_TREE.items():
        cues = node.musical_cues
        cue_vec = np.array([
            cues.get("tempo", 100.0),
            cues.get("mode", 0.5),
            cues.get("brightness", 0.5),
        ])
        weights = np.array([1.0 / 150.0, 1.0, 1.0])
        d = float(np.sqrt(np.sum((weights * (query - cue_vec)) ** 2)))
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


VONNEGUT_ARCS = {
    "rags_to_riches": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "man_in_hole": [0.7, 0.5, 0.3, 0.2, 0.3, 0.5, 0.7, 0.8],
    "icarus": [0.3, 0.5, 0.7, 0.9, 0.8, 0.5, 0.3, 0.1],
    "cinderella": [0.3, 0.5, 0.7, 0.4, 0.2, 0.5, 0.8, 0.9],
    "oedipus": [0.7, 0.5, 0.3, 0.5, 0.7, 0.4, 0.2, 0.1],
}


@dataclass
class MoodTrajectory:
    moods: list[str] = field(default_factory=list)

    @property
    def valence_curve(self) -> list[float]:
        return [MOOD_TREE[m].valence for m in self.moods if m in MOOD_TREE]

    @property
    def arousal_curve(self) -> list[float]:
        return [MOOD_TREE[m].arousal for m in self.moods if m in MOOD_TREE]

    def arc_similarity(self, arc_name: str) -> float:
        if arc_name not in VONNEGUT_ARCS:
            return 0.0
        vc = np.array(self.valence_curve, dtype=float)
        ref = np.array(VONNEGUT_ARCS[arc_name], dtype=float)
        if len(vc) < 2:
            return 0.0
        vc_interp = np.interp(np.linspace(0, 1, len(ref)), np.linspace(0, 1, len(vc)), vc)
        vc_n = vc_interp - vc_interp.mean()
        ref_n = ref - ref.mean()
        denom = np.linalg.norm(vc_n) * np.linalg.norm(ref_n)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(vc_n, ref_n) / denom)

    def classify_arc(self) -> tuple[str, float]:
        best = ("", 0.0)
        for arc_name in VONNEGUT_ARCS:
            sim = self.arc_similarity(arc_name)
            if sim > best[1]:
                best = (arc_name, sim)
        return best
