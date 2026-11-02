"""Arabic maqam system — jins (tetrachords), ajnas combinations, modulation paths."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Jins:
    """A jins (pl. ajnas) is a tetrachord or trichord — the building block of maqam."""
    name: str
    intervals: tuple[float, ...]  # in quarter-tones (1 = quarter tone, 2 = semitone, 4 = whole tone)
    size: int  # number of notes (typically 3 or 4)

    @property
    def span_quartertones(self) -> float:
        return sum(self.intervals)


AJNAS: dict[str, Jins] = {
    "ajam": Jins("Ajam", (4, 4, 2), 4),           # whole, whole, half
    "nahawand": Jins("Nahawand", (4, 2, 4), 4),    # whole, half, whole (natural minor)
    "kurd": Jins("Kurd", (2, 4, 4), 4),            # half, whole, whole
    "hijaz": Jins("Hijaz", (2, 6, 2), 4),          # half, aug2nd, half
    "bayati": Jins("Bayati", (3, 3, 4), 4),        # 3/4, 3/4, whole (quarter-tone)
    "rast": Jins("Rast", (4, 3, 3), 4),            # whole, 3/4, 3/4 (quarter-tone)
    "saba": Jins("Saba", (3, 3, 2), 3),            # 3/4, 3/4, half (trichord)
    "sikah": Jins("Sikah", (3, 4, 3), 4),          # 3/4, whole, 3/4
}


@dataclass
class Maqam:
    name: str
    lower_jins: str
    upper_jins: str
    ghammaz: int  # dominant degree (1-indexed) where upper jins begins
    tonic_note: str = "C"

    def intervals(self) -> list[float]:
        lower = AJNAS[self.lower_jins].intervals
        upper = AJNAS[self.upper_jins].intervals
        return list(lower) + list(upper)

    def scale_quartertones(self) -> list[float]:
        """Cumulative quarter-tone positions from tonic."""
        ivs = self.intervals()
        positions = [0.0]
        for iv in ivs:
            positions.append(positions[-1] + iv)
        return positions


COMMON_MAQAMAT: dict[str, Maqam] = {
    "rast": Maqam("Rast", "rast", "rast", 5),
    "bayati": Maqam("Bayati", "bayati", "nahawand", 4),
    "saba": Maqam("Saba", "saba", "hijaz", 4),
    "hijaz": Maqam("Hijaz", "hijaz", "rast", 5),
    "nahawand": Maqam("Nahawand", "nahawand", "kurd", 5),
    "ajam": Maqam("Ajam", "ajam", "ajam", 5),
    "kurd": Maqam("Kurd", "kurd", "hijaz", 4),
    "sikah": Maqam("Sikah", "sikah", "rast", 5),
}


def modulation_distance(a: str, b: str) -> int:
    """Number of jins substitutions needed to modulate from maqam a to b."""
    if a not in COMMON_MAQAMAT or b not in COMMON_MAQAMAT:
        return -1
    ma, mb = COMMON_MAQAMAT[a], COMMON_MAQAMAT[b]
    dist = 0
    if ma.lower_jins != mb.lower_jins:
        dist += 1
    if ma.upper_jins != mb.upper_jins:
        dist += 1
    return dist


def common_jins(a: str, b: str) -> list[str]:
    """Return shared ajnas between two maqamat."""
    if a not in COMMON_MAQAMAT or b not in COMMON_MAQAMAT:
        return []
    ma, mb = COMMON_MAQAMAT[a], COMMON_MAQAMAT[b]
    shared = set()
    for j in [ma.lower_jins, ma.upper_jins]:
        if j in [mb.lower_jins, mb.upper_jins]:
            shared.add(j)
    return sorted(shared)


def quartertone_to_cents(qt: float) -> float:
    return qt * 50.0


def is_quarter_tone_maqam(name: str) -> bool:
    """Does this maqam use quarter-tone intervals (3 quarter-tone steps)?"""
    if name not in COMMON_MAQAMAT:
        return False
    m = COMMON_MAQAMAT[name]
    ivs = m.intervals()
    return any(iv == 3 for iv in ivs)


def sayr(maqam_name: str) -> dict[str, list[int]]:
    """Characteristic melodic movement (sayr) — ascending and descending degree order."""
    if maqam_name not in COMMON_MAQAMAT:
        return {"ascending": [], "descending": []}
    m = COMMON_MAQAMAT[maqam_name]
    n = len(m.intervals()) + 1
    ascending = list(range(1, n + 1))
    descending = list(range(n, 0, -1))
    return {"ascending": ascending, "descending": descending}
