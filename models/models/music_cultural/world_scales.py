"""World scales: maqam, raga, pentatonic systems, and cross-cultural scale matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class WorldScale:
    name: str
    intervals_cents: list[float]
    origin_region: str
    description: str = ""

    @property
    def n_notes(self) -> int:
        return len(self.intervals_cents)

    def cumulative_cents(self) -> list[float]:
        out: list[float] = [0.0]
        for iv in self.intervals_cents:
            out.append(out[-1] + iv)
        return out

    def to_frequencies(self, root_hz: float = 440.0) -> list[float]:
        return [root_hz * (2.0 ** (c / 1200.0)) for c in self.cumulative_cents()]

    def contains_interval(self, cents: float, tolerance: float = 25.0) -> bool:
        cum = self.cumulative_cents()
        return any(abs(c - cents) <= tolerance for c in cum)


# ---------------------------------------------------------------------------
# Registry of 35+ world scales
# ---------------------------------------------------------------------------

# Arabic Maqam (quarter-tone tradition, intervals in cents)
MAQAM_RAST = WorldScale("Maqam Rast", [200, 150, 150, 200, 200, 150, 150], "Middle East",
                        "Principal Arabic maqam, uses 3/4-tone steps")
MAQAM_BAYATI = WorldScale("Maqam Bayati", [150, 150, 200, 200, 150, 200, 150], "Middle East",
                          "Second most common maqam, melancholic quality")
MAQAM_HIJAZ = WorldScale("Maqam Hijaz", [100, 300, 100, 200, 100, 200, 200], "Middle East",
                         "Augmented-second maqam, dramatic and exotic")
MAQAM_SABA = WorldScale("Maqam Saba", [150, 150, 100, 200, 100, 200, 200], "Middle East",
                        "Sorrowful maqam with lowered 4th degree")
MAQAM_NAHAWAND = WorldScale("Maqam Nahawand", [200, 100, 200, 200, 100, 200, 200], "Middle East",
                            "Arabic natural minor equivalent")
MAQAM_KURD = WorldScale("Maqam Kurd", [100, 200, 200, 200, 100, 200, 200], "Middle East",
                        "Phrygian-like maqam")
MAQAM_AJAM = WorldScale("Maqam Ajam", [200, 200, 100, 200, 200, 200, 100], "Middle East",
                        "Major-scale equivalent in Arabic music")

# Indian Raga (Thaat system, 12-TET approximation)
RAGA_BILAWAL = WorldScale("Raga Bilawal", [200, 200, 100, 200, 200, 200, 100], "South Asia",
                          "Ionian / Major equivalent thaat")
RAGA_KAFI = WorldScale("Raga Kafi", [200, 100, 200, 200, 200, 100, 200], "South Asia",
                       "Dorian equivalent thaat")
RAGA_BHAIRAVI = WorldScale("Raga Bhairavi", [100, 200, 200, 200, 100, 200, 200], "South Asia",
                           "Phrygian equivalent, morning devotional raga")
RAGA_YAMAN = WorldScale("Raga Yaman", [200, 200, 200, 100, 200, 200, 100], "South Asia",
                        "Lydian equivalent, evening raga")
RAGA_KHAMAJ = WorldScale("Raga Khamaj", [200, 200, 100, 200, 200, 100, 200], "South Asia",
                         "Mixolydian equivalent thaat")
RAGA_ASAVARI = WorldScale("Raga Asavari", [200, 100, 200, 200, 100, 200, 200], "South Asia",
                          "Aeolian / natural minor equivalent")
RAGA_MARWA = WorldScale("Raga Marwa", [100, 300, 100, 200, 200, 200, 100], "South Asia",
                        "Unusual thaat with raised 4th and lowered 2nd")
RAGA_TODI = WorldScale("Raga Todi", [100, 200, 200, 100, 200, 200, 200], "South Asia",
                       "Serious, contemplative morning raga")
RAGA_PURVI = WorldScale("Raga Purvi", [100, 300, 100, 200, 100, 300, 100], "South Asia",
                        "Sunset raga with two augmented seconds")

# Japanese scales
HIRAJOSHI = WorldScale("Hirajoshi", [200, 100, 400, 100, 400], "Japan",
                       "Most common Japanese pentatonic scale")
KUMOI = WorldScale("Kumoi", [200, 100, 400, 200, 300], "Japan",
                   "Japanese pentatonic with raised 5th degree")
IN_SCALE = WorldScale("In", [100, 400, 200, 100, 400], "Japan",
                      "Insen-related, minor pentatonic variant")
YO_SCALE = WorldScale("Yo", [200, 300, 200, 200, 300], "Japan",
                      "Bright Japanese pentatonic (min-yo folk)")
IWATO = WorldScale("Iwato", [100, 400, 100, 400, 200], "Japan",
                   "Darkest Japanese scale, used in koto music")
MIYAKO_BUSHI = WorldScale("Miyako-bushi", [100, 400, 200, 100, 400], "Japan",
                          "Kyoto city-style scale, similar to In")

# Chinese pentatonic modes
GONG = WorldScale("Gong", [200, 200, 300, 200, 300], "China",
                  "Major pentatonic / Gong mode (do)")
SHANG = WorldScale("Shang", [200, 300, 200, 300, 200], "China",
                   "Second mode of Chinese pentatonic (re)")
JIAO = WorldScale("Jiao", [300, 200, 300, 200, 200], "China",
                  "Third mode of Chinese pentatonic (mi)")
ZHI = WorldScale("Zhi", [200, 300, 200, 200, 300], "China",
                 "Fifth mode of Chinese pentatonic (sol)")
YU = WorldScale("Yu", [300, 200, 200, 300, 200], "China",
                "Minor pentatonic / Yu mode (la)")

# Indonesian
SLENDRO = WorldScale("Slendro", [240, 240, 240, 240, 240], "Indonesia",
                     "Near-equal 5-note gamelan tuning (~240 cents)")
PELOG = WorldScale("Pelog", [120, 270, 130, 280, 110, 290, 0], "Indonesia",
                   "7-note unequal gamelan tuning (approx)")

# Ethiopian
TIZITA_MAJOR = WorldScale("Tizita Major", [200, 200, 300, 200, 300], "Ethiopia",
                          "Ethiopian major pentatonic (nostalgia)")
TIZITA_MINOR = WorldScale("Tizita Minor", [200, 100, 400, 200, 300], "Ethiopia",
                          "Ethiopian minor pentatonic variant")
BATI_MAJOR = WorldScale("Bati Major", [300, 200, 200, 300, 200], "Ethiopia",
                        "Ethiopian minor pentatonic")
BATI_MINOR = WorldScale("Bati Minor", [300, 200, 100, 400, 200], "Ethiopia",
                        "Ethiopian pentatonic with lowered 4th")
AMBASSEL = WorldScale("Ambassel", [100, 400, 200, 100, 400], "Ethiopia",
                      "Contemplative Ethiopian scale")

# Other world scales
BLUES = WorldScale("Blues", [300, 200, 100, 100, 300, 200], "African-American",
                   "Blues hexatonic with blue notes (b3, b5, b7)")
WHOLE_TONE = WorldScale("Whole Tone", [200, 200, 200, 200, 200, 200], "Western",
                        "Symmetric 6-note whole-tone scale")
HUNGARIAN_MINOR = WorldScale("Hungarian Minor", [200, 100, 300, 100, 100, 300, 100], "Central Europe",
                             "Double harmonic minor / Gypsy minor")
PERSIAN = WorldScale("Persian", [100, 300, 100, 100, 200, 300, 100], "Persia",
                     "Persian scale with two augmented seconds")
BYZANTINE = WorldScale("Byzantine", [100, 300, 100, 200, 100, 300, 100], "Byzantine Empire",
                       "Double harmonic / Hijaz Kar")


WORLD_SCALE_REGISTRY: list[WorldScale] = [
    MAQAM_RAST, MAQAM_BAYATI, MAQAM_HIJAZ, MAQAM_SABA, MAQAM_NAHAWAND, MAQAM_KURD, MAQAM_AJAM,
    RAGA_BILAWAL, RAGA_KAFI, RAGA_BHAIRAVI, RAGA_YAMAN, RAGA_KHAMAJ, RAGA_ASAVARI,
    RAGA_MARWA, RAGA_TODI, RAGA_PURVI,
    HIRAJOSHI, KUMOI, IN_SCALE, YO_SCALE, IWATO, MIYAKO_BUSHI,
    GONG, SHANG, JIAO, ZHI, YU,
    SLENDRO, PELOG,
    TIZITA_MAJOR, TIZITA_MINOR, BATI_MAJOR, BATI_MINOR, AMBASSEL,
    BLUES, WHOLE_TONE, HUNGARIAN_MINOR, PERSIAN, BYZANTINE,
]


class ScaleMatcher:
    @staticmethod
    def match(pitch_cents: Sequence[float],
              registry: list[WorldScale] | None = None,
              top_k: int = 5) -> list[tuple[WorldScale, float]]:
        if registry is None:
            registry = WORLD_SCALE_REGISTRY
        if not pitch_cents:
            return []
        query = np.sort(np.asarray(pitch_cents, dtype=np.float64) % 1200.0)
        results: list[tuple[WorldScale, float]] = []
        for scale in registry:
            cum = np.array(scale.cumulative_cents()[:-1], dtype=np.float64) % 1200.0
            cum = np.sort(cum)
            # Compute minimum total cent deviation (Hausdorff-like)
            if len(cum) == 0:
                continue
            # For each query pitch, find nearest scale pitch
            dists_q = np.array([np.min(np.abs(cum - q)) for q in query])
            dists_s = np.array([np.min(np.abs(query - s)) for s in cum])
            score = float(np.mean(dists_q) + np.mean(dists_s)) / 2.0
            results.append((scale, score))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    @staticmethod
    def by_region(region: str, registry: list[WorldScale] | None = None) -> list[WorldScale]:
        if registry is None:
            registry = WORLD_SCALE_REGISTRY
        return [s for s in registry if region.lower() in s.origin_region.lower()]
