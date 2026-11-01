"""Mood ontology: hierarchical mood/emotion taxonomy for music.

Implements a tree-structured mood ontology inspired by Vonnegut story arcs
and Russell's circumplex model. Maps between VAD (valence-arousal-dominance)
coordinates and categorical mood labels through the ontology hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MoodNode:
    name: str
    valence: float
    arousal: float
    dominance: float = 0.5
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def vad(self) -> tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)


@dataclass
class MoodTrajectory:
    name: str
    moods: list[str]
    timestamps: list[float]
    description: str = ""

    def interpolate(self, t: float, tree: dict[str, MoodNode]) -> tuple[float, float, float]:
        if not self.moods or not self.timestamps:
            return (0.5, 0.5, 0.5)
        t = np.clip(t, self.timestamps[0], self.timestamps[-1])
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= t <= self.timestamps[i + 1]:
                alpha = (t - self.timestamps[i]) / (self.timestamps[i + 1] - self.timestamps[i] + 1e-12)
                n1 = tree.get(self.moods[i])
                n2 = tree.get(self.moods[i + 1])
                if n1 is None or n2 is None:
                    return (0.5, 0.5, 0.5)
                v = n1.valence + alpha * (n2.valence - n1.valence)
                a = n1.arousal + alpha * (n2.arousal - n1.arousal)
                d = n1.dominance + alpha * (n2.dominance - n1.dominance)
                return (v, a, d)
        last = tree.get(self.moods[-1])
        if last is None:
            return (0.5, 0.5, 0.5)
        return last.vad


def _build_mood_tree() -> dict[str, MoodNode]:
    tree: dict[str, MoodNode] = {}

    tree["positive"] = MoodNode("positive", 0.8, 0.5, 0.6, None, ["joyful", "peaceful", "triumphant"])
    tree["negative"] = MoodNode("negative", 0.2, 0.5, 0.4, None, ["sad", "angry", "anxious"])
    tree["neutral"] = MoodNode("neutral", 0.5, 0.5, 0.5, None, ["contemplative", "mysterious"])

    tree["joyful"] = MoodNode("joyful", 0.9, 0.8, 0.7, "positive",
                              ["ecstatic", "playful", "euphoric"],
                              "High valence, high arousal positive affect")
    tree["peaceful"] = MoodNode("peaceful", 0.8, 0.2, 0.5, "positive",
                                ["serene", "dreamy", "tender"],
                                "High valence, low arousal positive affect")
    tree["triumphant"] = MoodNode("triumphant", 0.85, 0.9, 0.9, "positive",
                                  ["heroic", "majestic", "glorious"],
                                  "High valence, very high arousal and dominance")

    tree["sad"] = MoodNode("sad", 0.15, 0.2, 0.2, "negative",
                           ["melancholic", "grieving", "nostalgic"],
                           "Low valence, low arousal negative affect")
    tree["angry"] = MoodNode("angry", 0.15, 0.85, 0.8, "negative",
                             ["furious", "aggressive", "defiant"],
                             "Low valence, high arousal negative affect")
    tree["anxious"] = MoodNode("anxious", 0.25, 0.75, 0.25, "negative",
                               ["tense", "fearful", "uneasy"],
                               "Low valence, moderate-high arousal, low dominance")

    tree["contemplative"] = MoodNode("contemplative", 0.5, 0.3, 0.5, "neutral",
                                     [], "Balanced valence, low arousal")
    tree["mysterious"] = MoodNode("mysterious", 0.45, 0.55, 0.4, "neutral",
                                  [], "Slightly below-neutral valence, moderate arousal")

    tree["ecstatic"] = MoodNode("ecstatic", 0.95, 0.95, 0.8, "joyful")
    tree["playful"] = MoodNode("playful", 0.85, 0.7, 0.6, "joyful")
    tree["euphoric"] = MoodNode("euphoric", 0.9, 0.9, 0.75, "joyful")
    tree["serene"] = MoodNode("serene", 0.8, 0.15, 0.5, "peaceful")
    tree["dreamy"] = MoodNode("dreamy", 0.7, 0.2, 0.4, "peaceful")
    tree["tender"] = MoodNode("tender", 0.75, 0.25, 0.45, "peaceful")
    tree["heroic"] = MoodNode("heroic", 0.85, 0.9, 0.95, "triumphant")
    tree["majestic"] = MoodNode("majestic", 0.8, 0.7, 0.9, "triumphant")
    tree["glorious"] = MoodNode("glorious", 0.9, 0.85, 0.85, "triumphant")
    tree["melancholic"] = MoodNode("melancholic", 0.2, 0.25, 0.25, "sad")
    tree["grieving"] = MoodNode("grieving", 0.05, 0.3, 0.15, "sad")
    tree["nostalgic"] = MoodNode("nostalgic", 0.4, 0.2, 0.3, "sad")
    tree["furious"] = MoodNode("furious", 0.1, 0.95, 0.85, "angry")
    tree["aggressive"] = MoodNode("aggressive", 0.15, 0.9, 0.9, "angry")
    tree["defiant"] = MoodNode("defiant", 0.25, 0.8, 0.8, "angry")
    tree["tense"] = MoodNode("tense", 0.3, 0.8, 0.3, "anxious")
    tree["fearful"] = MoodNode("fearful", 0.15, 0.75, 0.15, "anxious")
    tree["uneasy"] = MoodNode("uneasy", 0.35, 0.6, 0.3, "anxious")

    return tree


MOOD_TREE = _build_mood_tree()

VONNEGUT_ARCS = {
    "rags_to_riches": MoodTrajectory(
        "rags_to_riches",
        ["sad", "contemplative", "joyful", "triumphant"],
        [0.0, 0.3, 0.7, 1.0],
        "Steady rise from misfortune to fortune",
    ),
    "riches_to_rags": MoodTrajectory(
        "riches_to_rags",
        ["triumphant", "joyful", "contemplative", "sad"],
        [0.0, 0.3, 0.7, 1.0],
        "Steady fall from fortune to misfortune",
    ),
    "man_in_a_hole": MoodTrajectory(
        "man_in_a_hole",
        ["joyful", "sad", "joyful"],
        [0.0, 0.5, 1.0],
        "Fall then rise — most common story arc",
    ),
    "icarus": MoodTrajectory(
        "icarus",
        ["sad", "triumphant", "sad"],
        [0.0, 0.5, 1.0],
        "Rise then fall",
    ),
    "cinderella": MoodTrajectory(
        "cinderella",
        ["sad", "joyful", "sad", "triumphant"],
        [0.0, 0.33, 0.66, 1.0],
        "Rise, fall, rise — the most popular arc",
    ),
    "oedipus": MoodTrajectory(
        "oedipus",
        ["joyful", "sad", "joyful", "sad"],
        [0.0, 0.33, 0.66, 1.0],
        "Fall, rise, fall",
    ),
}


def mood_distance(tree: dict[str, MoodNode], name_a: str, name_b: str) -> float:
    a = tree.get(name_a)
    b = tree.get(name_b)
    if a is None or b is None:
        return float("inf")
    return float(np.sqrt(
        (a.valence - b.valence) ** 2 +
        (a.arousal - b.arousal) ** 2 +
        (a.dominance - b.dominance) ** 2
    ))


def mood_ancestors(tree: dict[str, MoodNode], name: str) -> list[str]:
    ancestors = []
    current = name
    while current in tree and tree[current].parent is not None:
        current = tree[current].parent
        ancestors.append(current)
    return ancestors


def mood_from_vad(tree: dict[str, MoodNode], valence: float, arousal: float,
                  dominance: float = 0.5) -> str:
    best_name = "neutral"
    best_dist = float("inf")
    for name, node in tree.items():
        if node.children:
            continue
        d = np.sqrt(
            (node.valence - valence) ** 2 +
            (node.arousal - arousal) ** 2 +
            (node.dominance - dominance) ** 2
        )
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def mood_from_features(tree: dict[str, MoodNode],
                       tempo_norm: float, brightness: float,
                       mode: float, energy: float) -> str:
    valence = 0.45 * mode + 0.25 * (1 - abs(tempo_norm - 0.5) * 2) + 0.20 * brightness + 0.10
    arousal = 0.50 * tempo_norm + 0.30 * energy + 0.20 * brightness
    dominance = 0.60 * energy + 0.20 * brightness + 0.20 * tempo_norm
    valence = float(np.clip(valence, 0, 1))
    arousal = float(np.clip(arousal, 0, 1))
    dominance = float(np.clip(dominance, 0, 1))
    return mood_from_vad(tree, valence, arousal, dominance)
