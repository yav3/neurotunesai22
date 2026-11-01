"""Indian classical raga system: thaats, ragas, similarity, classification."""
from __future__ import annotations
from dataclasses import dataclass, field

SWARAS = ("Sa", "re", "Re", "ga", "Ga", "Ma", "ma", "Pa", "dha", "Dha", "ni", "Ni")
SWARA_SEMITONES = {s: i for i, s in enumerate(SWARAS)}

THAAT_SCALES: dict[str, tuple[str, ...]] = {
    "Bilawal":  ("Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"),
    "Khamaj":   ("Sa", "Re", "Ga", "Ma", "Pa", "Dha", "ni"),
    "Kafi":     ("Sa", "Re", "ga", "Ma", "Pa", "Dha", "ni"),
    "Asavari":  ("Sa", "Re", "ga", "Ma", "Pa", "dha", "ni"),
    "Bhairavi":  ("Sa", "re", "ga", "Ma", "Pa", "dha", "ni"),
    "Kalyan":   ("Sa", "Re", "Ga", "ma", "Pa", "Dha", "Ni"),
    "Marwa":    ("Sa", "re", "Ga", "ma", "Pa", "Dha", "Ni"),
    "Purvi":    ("Sa", "re", "Ga", "ma", "Pa", "dha", "Ni"),
    "Todi":     ("Sa", "re", "ga", "ma", "Pa", "dha", "Ni"),
    "Bhairav":  ("Sa", "re", "Ga", "Ma", "Pa", "dha", "Ni"),
}


@dataclass(frozen=True)
class Raga:
    name: str
    thaat: str
    aroha: tuple[str, ...]
    avaroha: tuple[str, ...]
    vadi: str
    samvadi: str
    rasa: str
    time_of_day: str = ""
    notes: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if not self.notes:
            object.__setattr__(self, "notes", frozenset(self.aroha) | frozenset(self.avaroha))


RAGA_CATALOG: dict[str, Raga] = {
    "Yaman": Raga("Yaman", "Kalyan", ("Sa", "Re", "Ga", "ma", "Pa", "Dha", "Ni"), ("Ni", "Dha", "Pa", "ma", "Ga", "Re", "Sa"), "Ga", "Ni", "shringara", "evening"),
    "Bhairav": Raga("Bhairav", "Bhairav", ("Sa", "re", "Ga", "Ma", "Pa", "dha", "Ni"), ("Ni", "dha", "Pa", "Ma", "Ga", "re", "Sa"), "Dha", "Re", "shanta", "morning"),
    "Malkauns": Raga("Malkauns", "Bhairavi", ("Sa", "ga", "Ma", "dha", "ni"), ("ni", "dha", "Ma", "ga", "Sa"), "Ma", "ni", "karuna", "late_night"),
    "Darbari": Raga("Darbari", "Asavari", ("Sa", "Re", "ga", "Ma", "Pa", "dha", "ni"), ("ni", "dha", "Pa", "Ma", "ga", "Re", "Sa"), "Re", "Pa", "veera", "late_night"),
    "Bhimpalasi": Raga("Bhimpalasi", "Kafi", ("Sa", "ga", "Ma", "Pa", "ni"), ("ni", "Dha", "Pa", "Ma", "ga", "Re", "Sa"), "Ma", "Sa", "shringara", "afternoon"),
    "Todi": Raga("Todi", "Todi", ("Sa", "re", "ga", "ma", "Pa", "dha", "Ni"), ("Ni", "dha", "Pa", "ma", "ga", "re", "Sa"), "dha", "ga", "karuna", "morning"),
    "Marwa": Raga("Marwa", "Marwa", ("Sa", "re", "Ga", "ma", "Dha", "Ni"), ("Ni", "Dha", "ma", "Ga", "re", "Sa"), "Dha", "Re", "veera", "evening"),
    "Bilawal": Raga("Bilawal", "Bilawal", ("Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"), ("Ni", "Dha", "Pa", "Ma", "Ga", "Re", "Sa"), "Dha", "Ga", "hasya", "morning"),
    "Khamaj": Raga("Khamaj", "Khamaj", ("Sa", "Re", "Ga", "Ma", "Pa", "Dha", "ni"), ("ni", "Dha", "Pa", "Ma", "Ga", "Re", "Sa"), "Ga", "ni", "shringara", "evening"),
    "Kafi": Raga("Kafi", "Kafi", ("Sa", "Re", "ga", "Ma", "Pa", "Dha", "ni"), ("ni", "Dha", "Pa", "Ma", "ga", "Re", "Sa"), "Pa", "Re", "shringara", "afternoon"),
    "Asavari": Raga("Asavari", "Asavari", ("Sa", "Re", "ga", "Ma", "Pa", "dha", "ni"), ("ni", "dha", "Pa", "Ma", "ga", "Re", "Sa"), "dha", "ga", "karuna", "morning"),
    "Bhairavi": Raga("Bhairavi", "Bhairavi", ("Sa", "re", "ga", "Ma", "Pa", "dha", "ni"), ("ni", "dha", "Pa", "Ma", "ga", "re", "Sa"), "Ma", "Sa", "karuna", "morning"),
    "Kalyan": Raga("Kalyan", "Kalyan", ("Sa", "Re", "Ga", "ma", "Pa", "Dha", "Ni"), ("Ni", "Dha", "Pa", "ma", "Ga", "Re", "Sa"), "Ga", "Ni", "shringara", "evening"),
    "Purvi": Raga("Purvi", "Purvi", ("Sa", "re", "Ga", "ma", "Pa", "dha", "Ni"), ("Ni", "dha", "Pa", "ma", "Ga", "re", "Sa"), "Ga", "Ni", "shanta", "evening"),
    "Lalit": Raga("Lalit", "Marwa", ("Sa", "re", "Ga", "Ma", "ma", "Dha", "Ni"), ("Ni", "Dha", "ma", "Ma", "Ga", "re", "Sa"), "Ma", "Ni", "shringara", "dawn"),
    "Kedar": Raga("Kedar", "Kalyan", ("Sa", "Ma", "ma", "Pa", "Dha", "Ni"), ("Ni", "Dha", "Pa", "ma", "Ma", "Ga", "Re", "Sa"), "Ma", "Sa", "shringara", "late_evening"),
    "Bageshri": Raga("Bageshri", "Kafi", ("Sa", "ga", "Ma", "Dha", "ni"), ("ni", "Dha", "Ma", "ga", "Re", "Sa"), "Ma", "Sa", "shringara", "late_night"),
    "Desh": Raga("Desh", "Khamaj", ("Sa", "Re", "Ma", "Pa", "Ni"), ("Ni", "Dha", "Pa", "Ma", "Ga", "Re", "Sa"), "Pa", "Re", "shringara", "late_evening"),
    "Hansadhwani": Raga("Hansadhwani", "Bilawal", ("Sa", "Re", "Ga", "Pa", "Ni"), ("Ni", "Pa", "Ga", "Re", "Sa"), "Re", "Pa", "hasya", "evening"),
    "Durga": Raga("Durga", "Bilawal", ("Sa", "Re", "Ma", "Pa", "Dha"), ("Dha", "Pa", "Ma", "Re", "Sa"), "Re", "Pa", "veera", "evening"),
}


def thaat_to_ragas(thaat: str) -> list[Raga]:
    return [r for r in RAGA_CATALOG.values() if r.thaat == thaat]


def raga_similarity(r1: Raga, r2: Raga) -> float:
    if not r1.notes or not r2.notes:
        return 0.0
    intersection = len(r1.notes & r2.notes)
    union = len(r1.notes | r2.notes)
    return intersection / union if union else 0.0


def classify_raga(notes: set[str]) -> list[tuple[Raga, float]]:
    frozen = frozenset(notes)
    scored = []
    for raga in RAGA_CATALOG.values():
        if not raga.notes:
            continue
        intersection = len(frozen & raga.notes)
        union = len(frozen | raga.notes)
        sim = intersection / union if union else 0.0
        if sim > 0.3:
            scored.append((raga, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
