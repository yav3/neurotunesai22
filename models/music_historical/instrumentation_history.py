"""Historical instrumentation timeline."""

from __future__ import annotations

from dataclasses import dataclass
from .periods import MusicalPeriod


@dataclass(frozen=True)
class InstrumentEra:
    instrument: str
    introduced_period: MusicalPeriod
    family: str
    range_semitones: int


INSTRUMENT_TIMELINE: list[InstrumentEra] = [
    InstrumentEra("organ", MusicalPeriod.MEDIEVAL, "keyboard", 61),
    InstrumentEra("lute", MusicalPeriod.MEDIEVAL, "string", 24),
    InstrumentEra("recorder", MusicalPeriod.MEDIEVAL, "woodwind", 20),
    InstrumentEra("shawm", MusicalPeriod.MEDIEVAL, "woodwind", 18),
    InstrumentEra("sackbut", MusicalPeriod.RENAISSANCE, "brass", 30),
    InstrumentEra("violin", MusicalPeriod.RENAISSANCE, "string", 44),
    InstrumentEra("viola", MusicalPeriod.RENAISSANCE, "string", 38),
    InstrumentEra("cello", MusicalPeriod.RENAISSANCE, "string", 40),
    InstrumentEra("harpsichord", MusicalPeriod.BAROQUE, "keyboard", 61),
    InstrumentEra("oboe", MusicalPeriod.BAROQUE, "woodwind", 33),
    InstrumentEra("bassoon", MusicalPeriod.BAROQUE, "woodwind", 40),
    InstrumentEra("trumpet", MusicalPeriod.BAROQUE, "brass", 27),
    InstrumentEra("french_horn", MusicalPeriod.CLASSICAL, "brass", 42),
    InstrumentEra("clarinet", MusicalPeriod.CLASSICAL, "woodwind", 39),
    InstrumentEra("piano", MusicalPeriod.CLASSICAL, "keyboard", 88),
    InstrumentEra("timpani", MusicalPeriod.CLASSICAL, "percussion", 12),
    InstrumentEra("tuba", MusicalPeriod.ROMANTIC, "brass", 36),
    InstrumentEra("saxophone", MusicalPeriod.ROMANTIC, "woodwind", 30),
    InstrumentEra("celesta", MusicalPeriod.ROMANTIC, "keyboard", 48),
    InstrumentEra("synthesizer", MusicalPeriod.MODERN, "electronic", 88),
]

_PERIOD_ORDER = list(MusicalPeriod)

_ORCHESTRA_SIZES: dict[MusicalPeriod, int] = {
    MusicalPeriod.MEDIEVAL: 5,
    MusicalPeriod.RENAISSANCE: 12,
    MusicalPeriod.BAROQUE: 25,
    MusicalPeriod.CLASSICAL: 40,
    MusicalPeriod.ROMANTIC: 80,
    MusicalPeriod.IMPRESSIONIST: 70,
    MusicalPeriod.MODERN: 90,
    MusicalPeriod.CONTEMPORARY: 60,
    MusicalPeriod.AVANT_GARDE: 30,
}


def available_instruments(period: MusicalPeriod) -> list[str]:
    idx = _PERIOD_ORDER.index(period)
    return [
        ie.instrument for ie in INSTRUMENT_TIMELINE
        if _PERIOD_ORDER.index(ie.introduced_period) <= idx
    ]


def orchestra_size_by_period(period: MusicalPeriod) -> int:
    return _ORCHESTRA_SIZES.get(period, 30)


def instrument_family_balance(instrument_list: list[str]) -> dict[str, float]:
    family_map = {ie.instrument: ie.family for ie in INSTRUMENT_TIMELINE}
    families: dict[str, int] = {}
    for inst in instrument_list:
        fam = family_map.get(inst, "unknown")
        families[fam] = families.get(fam, 0) + 1
    total = max(sum(families.values()), 1)
    return {k: v / total for k, v in families.items()}
