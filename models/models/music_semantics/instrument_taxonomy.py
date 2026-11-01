"""Instrument classification: families, ranges, compatibility, ensemble suggestion."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instrument:
    name: str
    family: str
    subfamily: str
    pitch_range: tuple[int, int]
    is_pitched: bool = True
    common_genres: tuple[str, ...] = ()


INSTRUMENT_CATALOG: dict[str, Instrument] = {
    "violin": Instrument("violin", "strings", "bowed", (55, 103), True, ("classical", "folk", "jazz")),
    "viola": Instrument("viola", "strings", "bowed", (48, 93), True, ("classical", "chamber")),
    "cello": Instrument("cello", "strings", "bowed", (36, 84), True, ("classical", "chamber")),
    "double_bass": Instrument("double_bass", "strings", "bowed", (28, 67), True, ("classical", "jazz")),
    "acoustic_guitar": Instrument("acoustic_guitar", "strings", "plucked", (40, 88), True, ("folk", "pop", "country")),
    "electric_guitar": Instrument("electric_guitar", "strings", "plucked", (40, 88), True, ("rock", "blues", "metal", "jazz")),
    "bass_guitar": Instrument("bass_guitar", "strings", "plucked", (28, 60), True, ("rock", "pop", "funk", "jazz")),
    "harp": Instrument("harp", "strings", "plucked", (24, 103), True, ("classical", "folk")),
    "banjo": Instrument("banjo", "strings", "plucked", (50, 84), True, ("folk", "country", "bluegrass")),
    "sitar": Instrument("sitar", "strings", "plucked", (36, 84), True, ("world",)),
    "flute": Instrument("flute", "woodwinds", "edge-blown", (60, 96), True, ("classical", "jazz", "folk")),
    "clarinet": Instrument("clarinet", "woodwinds", "single-reed", (50, 94), True, ("classical", "jazz")),
    "oboe": Instrument("oboe", "woodwinds", "double-reed", (58, 91), True, ("classical",)),
    "bassoon": Instrument("bassoon", "woodwinds", "double-reed", (34, 75), True, ("classical",)),
    "saxophone": Instrument("saxophone", "woodwinds", "single-reed", (44, 87), True, ("jazz", "rnb", "funk", "rock")),
    "trumpet": Instrument("trumpet", "brass", "valved", (52, 84), True, ("jazz", "classical", "latin")),
    "trombone": Instrument("trombone", "brass", "slide", (40, 77), True, ("jazz", "classical")),
    "french_horn": Instrument("french_horn", "brass", "valved", (34, 77), True, ("classical",)),
    "tuba": Instrument("tuba", "brass", "valved", (24, 58), True, ("classical",)),
    "piano": Instrument("piano", "keyboard", "hammered", (21, 108), True, ("classical", "jazz", "pop", "rock")),
    "organ": Instrument("organ", "keyboard", "pipe/electronic", (24, 108), True, ("classical", "rock", "gospel")),
    "harpsichord": Instrument("harpsichord", "keyboard", "plucked", (29, 89), True, ("baroque",)),
    "synthesizer": Instrument("synthesizer", "electronic", "synthesis", (0, 127), True, ("electronic", "pop", "rock")),
    "drum_machine": Instrument("drum_machine", "electronic", "sampled", (0, 127), False, ("electronic", "hip_hop")),
    "drum_kit": Instrument("drum_kit", "percussion", "struck", (36, 84), False, ("rock", "pop", "jazz", "metal")),
    "timpani": Instrument("timpani", "percussion", "struck", (40, 55), True, ("classical",)),
    "marimba": Instrument("marimba", "percussion", "struck", (45, 96), True, ("classical", "world")),
    "vibraphone": Instrument("vibraphone", "percussion", "struck", (53, 89), True, ("jazz",)),
    "congas": Instrument("congas", "percussion", "hand-struck", (48, 72), False, ("latin", "world", "jazz")),
    "djembe": Instrument("djembe", "percussion", "hand-struck", (48, 72), False, ("world", "afrobeat")),
    "tabla": Instrument("tabla", "percussion", "hand-struck", (48, 84), True, ("world",)),
    "voice_soprano": Instrument("voice_soprano", "voice", "female", (60, 84), True, ("classical", "pop")),
    "voice_alto": Instrument("voice_alto", "voice", "female", (53, 77), True, ("classical", "jazz")),
    "voice_tenor": Instrument("voice_tenor", "voice", "male", (48, 72), True, ("classical", "pop")),
    "voice_bass": Instrument("voice_bass", "voice", "male", (40, 64), True, ("classical",)),
    "harmonica": Instrument("harmonica", "woodwinds", "free-reed", (48, 84), True, ("blues", "folk", "country")),
    "accordion": Instrument("accordion", "keyboard", "free-reed", (34, 96), True, ("folk", "world", "klezmer")),
    "koto": Instrument("koto", "strings", "plucked", (36, 96), True, ("world",)),
    "erhu": Instrument("erhu", "strings", "bowed", (55, 96), True, ("world",)),
    "oud": Instrument("oud", "strings", "plucked", (36, 72), True, ("world",)),
}

_FAMILY_COMPAT: dict[tuple[str, str], float] = {
    ("strings", "strings"): 0.9, ("strings", "woodwinds"): 0.8, ("strings", "brass"): 0.7,
    ("strings", "keyboard"): 0.85, ("strings", "percussion"): 0.6, ("strings", "voice"): 0.85,
    ("woodwinds", "woodwinds"): 0.85, ("woodwinds", "brass"): 0.7, ("woodwinds", "keyboard"): 0.8,
    ("brass", "brass"): 0.8, ("brass", "percussion"): 0.7, ("keyboard", "voice"): 0.9,
    ("percussion", "percussion"): 0.6, ("electronic", "electronic"): 0.85,
    ("voice", "voice"): 0.9,
}


def instrument_compatibility(i1: Instrument, i2: Instrument) -> float:
    if i1.name == i2.name:
        return 1.0
    key = (i1.family, i2.family) if i1.family <= i2.family else (i2.family, i1.family)
    base = _FAMILY_COMPAT.get(key, 0.5)
    shared_genres = set(i1.common_genres) & set(i2.common_genres)
    genre_bonus = min(len(shared_genres) * 0.05, 0.1)
    lo = max(i1.pitch_range[0], i2.pitch_range[0])
    hi = min(i1.pitch_range[1], i2.pitch_range[1])
    overlap = max(0, hi - lo)
    total_range = max(i1.pitch_range[1], i2.pitch_range[1]) - min(i1.pitch_range[0], i2.pitch_range[0])
    range_factor = overlap / total_range if total_range > 0 else 0.0
    return min(1.0, base + genre_bonus + range_factor * 0.1)


def ensemble_suggestion(genre: str, size: int) -> list[str]:
    genre_lower = genre.lower()
    candidates = [(name, inst) for name, inst in INSTRUMENT_CATALOG.items()
                  if any(g == genre_lower or genre_lower in g for g in inst.common_genres)]
    if not candidates:
        candidates = list(INSTRUMENT_CATALOG.items())
    families_used: dict[str, int] = {}
    selected: list[str] = []
    for name, inst in candidates:
        if len(selected) >= size:
            break
        family_count = families_used.get(inst.family, 0)
        if family_count < 2:
            selected.append(name)
            families_used[inst.family] = family_count + 1
    return selected[:size]
