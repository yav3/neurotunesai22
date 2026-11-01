"""Chinese traditional music: pentatonic modes, jianpu notation, instrument families."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


PENTATONIC_MODES: dict[str, tuple[int, ...]] = {
    "gong":   (0, 2, 4, 7, 9),
    "shang":  (0, 2, 5, 7, 10),
    "jiao":   (0, 3, 5, 8, 10),
    "zhi":    (0, 2, 5, 7, 9),
    "yu":     (0, 3, 5, 7, 10),
}

HEPTATONIC_SCALES: dict[str, tuple[int, ...]] = {
    "yayue": (0, 2, 4, 5, 7, 9, 11),
    "qingyue": (0, 2, 4, 6, 7, 9, 11),
    "yanyue": (0, 2, 3, 5, 7, 9, 10),
}

JIANPU_MAP = {1: "do", 2: "re", 3: "mi", 4: "fa", 5: "sol", 6: "la", 7: "si"}

INSTRUMENT_FAMILIES: dict[str, list[str]] = {
    "silk": ["erhu", "zhonghu", "pipa", "zheng", "guqin", "ruan", "sanxian", "liuqin"],
    "bamboo": ["dizi", "xiao", "sheng", "guan", "hulusi", "bawu"],
    "metal": ["luo", "bo", "yunluo", "bianzhong"],
    "stone": ["bianqing", "teqing"],
    "earth": ["xun"],
    "gourd": ["sheng_gourd"],
    "wood": ["muyu", "bangzi", "ban"],
    "skin": ["tanggu", "paigu", "dagu", "xiaogu", "biangu"],
}


@dataclass(frozen=True)
class ChineseMode:
    name: str
    intervals: tuple[int, ...]
    character: str
    season: str = ""


MODE_CATALOG: dict[str, ChineseMode] = {
    "gong":  ChineseMode("gong",  (0, 2, 4, 7, 9),  "majestic",   "earth"),
    "shang": ChineseMode("shang", (0, 2, 5, 7, 10),  "sorrowful",  "autumn"),
    "jiao":  ChineseMode("jiao",  (0, 3, 5, 8, 10),  "angry",      "spring"),
    "zhi":   ChineseMode("zhi",   (0, 2, 5, 7, 9),   "joyful",     "summer"),
    "yu":    ChineseMode("yu",    (0, 3, 5, 7, 10),   "melancholy", "winter"),
}


def classify_mode(pitch_classes: list[int], root: int = 0) -> list[tuple[str, float]]:
    """Classify a pitch set against Chinese pentatonic modes. Returns (mode, score) pairs."""
    pcs = frozenset((p - root) % 12 for p in pitch_classes)
    results = []
    for name, mode in MODE_CATALOG.items():
        mode_pcs = frozenset(mode.intervals)
        if not pcs:
            continue
        intersection = len(pcs & mode_pcs)
        union = len(pcs | mode_pcs)
        score = intersection / union if union > 0 else 0.0
        if score > 0.2:
            results.append((name, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def pitch_to_jianpu(midi_note: int, key_root: int = 60) -> str:
    """Convert MIDI note to jianpu number notation relative to key root.
    key_root: MIDI note of do (default C4=60)."""
    diff = midi_note - key_root
    octave_shift = diff // 12
    degree = diff % 12
    degree_map = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
    jianpu_num = degree_map.get(degree, 0)
    if jianpu_num == 0:
        closest = min(degree_map.keys(), key=lambda k: abs(k - degree))
        jianpu_num = degree_map[closest]
    result = str(jianpu_num)
    if octave_shift > 0:
        result += "·" * octave_shift
    elif octave_shift < 0:
        result += "," * abs(octave_shift)
    return result


def jianpu_sequence(midi_notes: list[int], key_root: int = 60) -> list[str]:
    return [pitch_to_jianpu(n, key_root) for n in midi_notes]


def pentatonic_interval_histogram(pitches: list[int]) -> np.ndarray:
    """Interval histogram for pitch sequence, mod-12."""
    if len(pitches) < 2:
        return np.zeros(12)
    hist = np.zeros(12)
    for i in range(len(pitches) - 1):
        interval = (pitches[i + 1] - pitches[i]) % 12
        hist[interval] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def mode_affinity_vector(pitches: list[int], root: int = 0) -> np.ndarray:
    """5D vector: affinity of pitch set to each of the 5 pentatonic modes."""
    scores = np.zeros(5)
    mode_names = list(MODE_CATALOG.keys())
    classified = classify_mode(pitches, root)
    name_to_score = dict(classified)
    for i, name in enumerate(mode_names):
        scores[i] = name_to_score.get(name, 0.0)
    return scores


def silk_bamboo_balance(instruments: list[str]) -> float:
    """Ratio of silk to bamboo instruments in an ensemble. Returns [-1, 1]:
    -1 = all bamboo, 0 = balanced, +1 = all silk."""
    silk = set(INSTRUMENT_FAMILIES["silk"])
    bamboo = set(INSTRUMENT_FAMILIES["bamboo"])
    n_silk = sum(1 for i in instruments if i in silk)
    n_bamboo = sum(1 for i in instruments if i in bamboo)
    total = n_silk + n_bamboo
    if total == 0:
        return 0.0
    return (n_silk - n_bamboo) / total


class QinTuning:
    """Guqin string tuning and harmonics model."""
    STANDARD_TUNING = (65.41, 73.42, 82.41, 98.0, 110.0, 130.81, 146.83)

    def __init__(self, tuning: tuple[float, ...] | None = None):
        self.tuning = tuning or self.STANDARD_TUNING

    def harmonic(self, string: int, node: int) -> float:
        """Frequency of harmonic at node position on a string (1-indexed).
        node: 1=fundamental, 2=octave, 3=fifth+octave, etc."""
        if string < 1 or string > len(self.tuning):
            return 0.0
        return self.tuning[string - 1] * node

    def all_harmonics(self, string: int, max_node: int = 7) -> list[float]:
        return [self.harmonic(string, n) for n in range(1, max_node + 1)]

    def open_string_intervals(self) -> list[float]:
        """Intervals between adjacent open strings in cents."""
        intervals = []
        for i in range(len(self.tuning) - 1):
            ratio = self.tuning[i + 1] / self.tuning[i]
            intervals.append(1200.0 * np.log2(ratio))
        return intervals
