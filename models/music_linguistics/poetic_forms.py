"""Poetic form templates: structural constraints for 25+ global verse forms."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MeterConstraint(Enum):
    FREE = "free"
    IAMBIC_PENTAMETER = "iambic_pentameter"
    IAMBIC_TETRAMETER = "iambic_tetrameter"
    IAMBIC_TRIMETER = "iambic_trimeter"
    TROCHAIC_TETRAMETER = "trochaic_tetrameter"
    ALEXANDRINE = "alexandrine"
    HENDECASYLLABLE = "hendecasyllable"
    COMMON_METER = "common_meter"
    BALLAD_METER = "ballad_meter"
    SYLLABIC = "syllabic"
    QUANTITATIVE = "quantitative"
    MORA_TIMED = "mora_timed"


@dataclass
class StanzaTemplate:
    lines: int
    syllables_per_line: list[int | None] = field(default_factory=list)
    rhyme_scheme: str = ""
    meter: MeterConstraint = MeterConstraint.FREE
    refrain_lines: list[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class PoeticForm:
    name: str
    origin: str
    stanzas: list[StanzaTemplate]
    total_lines: int | None = None
    repeating: bool = False
    turn_line: int | None = None
    couplet_close: bool = False
    envoi: StanzaTemplate | None = None
    notes: str = ""

    @property
    def min_lines(self) -> int:
        if self.total_lines is not None:
            return self.total_lines
        return sum(s.lines for s in self.stanzas)

    def syllable_map(self) -> list[int | None]:
        result: list[int | None] = []
        for s in self.stanzas:
            if s.syllables_per_line:
                result.extend(s.syllables_per_line)
            else:
                result.extend([None] * s.lines)
        if self.envoi and self.envoi.syllables_per_line:
            result.extend(self.envoi.syllables_per_line)
        elif self.envoi:
            result.extend([None] * self.envoi.lines)
        return result


SONNET_SHAKESPEAREAN = PoeticForm(
    name="Shakespearean Sonnet",
    origin="English",
    stanzas=[
        StanzaTemplate(4, rhyme_scheme="ABAB", meter=MeterConstraint.IAMBIC_PENTAMETER),
        StanzaTemplate(4, rhyme_scheme="CDCD", meter=MeterConstraint.IAMBIC_PENTAMETER),
        StanzaTemplate(4, rhyme_scheme="EFEF", meter=MeterConstraint.IAMBIC_PENTAMETER),
        StanzaTemplate(2, rhyme_scheme="GG", meter=MeterConstraint.IAMBIC_PENTAMETER),
    ],
    total_lines=14,
    turn_line=9,
    couplet_close=True,
)

SONNET_PETRARCHAN = PoeticForm(
    name="Petrarchan Sonnet",
    origin="Italian",
    stanzas=[
        StanzaTemplate(8, rhyme_scheme="ABBAABBA", meter=MeterConstraint.IAMBIC_PENTAMETER),
        StanzaTemplate(6, rhyme_scheme="CDECDE", meter=MeterConstraint.IAMBIC_PENTAMETER),
    ],
    total_lines=14,
    turn_line=9,
)

SONNET_SPENSERIAN = PoeticForm(
    name="Spenserian Sonnet",
    origin="English",
    stanzas=[
        StanzaTemplate(4, rhyme_scheme="ABAB", meter=MeterConstraint.IAMBIC_PENTAMETER),
        StanzaTemplate(4, rhyme_scheme="BCBC", meter=MeterConstraint.IAMBIC_PENTAMETER),
        StanzaTemplate(4, rhyme_scheme="CDCD", meter=MeterConstraint.IAMBIC_PENTAMETER),
        StanzaTemplate(2, rhyme_scheme="EE", meter=MeterConstraint.IAMBIC_PENTAMETER),
    ],
    total_lines=14,
    couplet_close=True,
    notes="Interlocking rhyme scheme links quatrains",
)

HAIKU = PoeticForm(
    name="Haiku",
    origin="Japanese",
    stanzas=[StanzaTemplate(3, syllables_per_line=[5, 7, 5], meter=MeterConstraint.MORA_TIMED)],
    total_lines=3,
    notes="Traditional: kigo (season word) + kireji (cutting word)",
)

TANKA = PoeticForm(
    name="Tanka",
    origin="Japanese",
    stanzas=[StanzaTemplate(5, syllables_per_line=[5, 7, 5, 7, 7], meter=MeterConstraint.MORA_TIMED)],
    total_lines=5,
    turn_line=4,
    notes="Upper poem (kami-no-ku) 5-7-5 + lower poem (shimo-no-ku) 7-7",
)

GHAZAL = PoeticForm(
    name="Ghazal",
    origin="Arabic/Persian",
    stanzas=[
        StanzaTemplate(2, rhyme_scheme="AA", meter=MeterConstraint.QUANTITATIVE),
    ],
    repeating=True,
    notes="5-15 couplets; radif (refrain) + qafia (rhyme) end each second line; maqta names the poet",
)

VILLANELLE = PoeticForm(
    name="Villanelle",
    origin="French/Italian",
    stanzas=[
        StanzaTemplate(3, rhyme_scheme="ABA", meter=MeterConstraint.IAMBIC_PENTAMETER, refrain_lines=[0]),
        StanzaTemplate(3, rhyme_scheme="ABA", meter=MeterConstraint.IAMBIC_PENTAMETER, refrain_lines=[2]),
        StanzaTemplate(3, rhyme_scheme="ABA", meter=MeterConstraint.IAMBIC_PENTAMETER, refrain_lines=[0]),
        StanzaTemplate(3, rhyme_scheme="ABA", meter=MeterConstraint.IAMBIC_PENTAMETER, refrain_lines=[2]),
        StanzaTemplate(3, rhyme_scheme="ABA", meter=MeterConstraint.IAMBIC_PENTAMETER, refrain_lines=[0]),
        StanzaTemplate(4, rhyme_scheme="ABAA", meter=MeterConstraint.IAMBIC_PENTAMETER, refrain_lines=[2, 3]),
    ],
    total_lines=19,
    notes="Two refrains (A1, A2) alternate as final lines; both close the quatrain",
)

SESTINA = PoeticForm(
    name="Sestina",
    origin="Occitan/Provençal",
    stanzas=[
        StanzaTemplate(6, notes="End-words rotate: 1-2-3-4-5-6"),
        StanzaTemplate(6, notes="End-words: 6-1-5-2-4-3"),
        StanzaTemplate(6, notes="End-words: 3-6-4-1-2-5"),
        StanzaTemplate(6, notes="End-words: 5-3-2-6-1-4"),
        StanzaTemplate(6, notes="End-words: 4-5-1-3-6-2"),
        StanzaTemplate(6, notes="End-words: 2-4-6-5-3-1"),
    ],
    total_lines=39,
    envoi=StanzaTemplate(3, notes="Tercet embeds all 6 end-words (2 per line)"),
    notes="No rhyme scheme; structural constraint is end-word rotation (retrogradatio cruciata)",
)

LIMERICK = PoeticForm(
    name="Limerick",
    origin="English/Irish",
    stanzas=[StanzaTemplate(5, syllables_per_line=[8, 8, 5, 5, 8], rhyme_scheme="AABBA")],
    total_lines=5,
    notes="Anapestic rhythm; typically humorous",
)

RUBA_I = PoeticForm(
    name="Rubāʿī",
    origin="Persian",
    stanzas=[StanzaTemplate(4, rhyme_scheme="AABA", meter=MeterConstraint.QUANTITATIVE)],
    total_lines=4,
    notes="Quatrain form; Omar Khayyám's Rubáiyát. Third line unrhymed creates tension.",
)

PANTOUM = PoeticForm(
    name="Pantoum",
    origin="Malay",
    stanzas=[
        StanzaTemplate(4, rhyme_scheme="ABAB", notes="Lines 2,4 become lines 1,3 of next stanza"),
    ],
    repeating=True,
    notes="Interlocking repetition; final stanza's lines 2,4 = first stanza's lines 3,1",
)

TERZA_RIMA = PoeticForm(
    name="Terza Rima",
    origin="Italian",
    stanzas=[StanzaTemplate(3, rhyme_scheme="ABA", meter=MeterConstraint.HENDECASYLLABLE)],
    repeating=True,
    couplet_close=True,
    notes="Chain rhyme: ABA BCB CDC ... YZY Z(Z). Dante's Commedia.",
)

OTTAVA_RIMA = PoeticForm(
    name="Ottava Rima",
    origin="Italian",
    stanzas=[StanzaTemplate(8, rhyme_scheme="ABABABCC", meter=MeterConstraint.HENDECASYLLABLE)],
    repeating=True,
    couplet_close=True,
    notes="Byron's Don Juan; Ariosto's Orlando Furioso",
)

BALLAD = PoeticForm(
    name="Ballad",
    origin="English/Scottish",
    stanzas=[
        StanzaTemplate(4, rhyme_scheme="ABCB", meter=MeterConstraint.BALLAD_METER,
                       syllables_per_line=[8, 6, 8, 6]),
    ],
    repeating=True,
    notes="Alternating 4-stress/3-stress lines; narrative tradition",
)

ODE_HORATIAN = PoeticForm(
    name="Horatian Ode",
    origin="Latin",
    stanzas=[StanzaTemplate(4, meter=MeterConstraint.FREE)],
    repeating=True,
    notes="Regular stanzas, contemplative tone; Keats's great odes",
)

ODE_PINDARIC = PoeticForm(
    name="Pindaric Ode",
    origin="Greek",
    stanzas=[
        StanzaTemplate(lines=0, notes="Strophe — variable length"),
        StanzaTemplate(lines=0, notes="Antistrophe — mirrors strophe"),
        StanzaTemplate(lines=0, notes="Epode — different structure"),
    ],
    notes="Triadic structure; choral performance; highly irregular",
)

SIJO = PoeticForm(
    name="Sijo",
    origin="Korean",
    stanzas=[StanzaTemplate(3, syllables_per_line=[15, 15, 15])],
    total_lines=3,
    turn_line=3,
    notes="Each line ~14-16 syllables in 2 phrases; twist in final line's first phrase",
)

QASIDA = PoeticForm(
    name="Qaṣīda",
    origin="Arabic",
    stanzas=[StanzaTemplate(2, rhyme_scheme="AA")],
    repeating=True,
    notes="Monorhyme couplets (AA BA CA DA ...); nasib (prelude) → rahil → madih (praise)",
)

RENGA_UNIT = PoeticForm(
    name="Renga (unit)",
    origin="Japanese",
    stanzas=[
        StanzaTemplate(3, syllables_per_line=[5, 7, 5], meter=MeterConstraint.MORA_TIMED),
        StanzaTemplate(2, syllables_per_line=[7, 7], meter=MeterConstraint.MORA_TIMED),
    ],
    total_lines=5,
    notes="Collaborative linked verse; alternates 5-7-5 and 7-7 stanzas",
)

ALEXANDRINE_COUPLET = PoeticForm(
    name="Alexandrine Couplet",
    origin="French",
    stanzas=[StanzaTemplate(2, syllables_per_line=[12, 12], rhyme_scheme="AA",
                            meter=MeterConstraint.ALEXANDRINE)],
    repeating=True,
    notes="12-syllable lines with caesura at 6th syllable; Racine, Corneille",
)

SAPPHIC_STANZA = PoeticForm(
    name="Sapphic Stanza",
    origin="Greek",
    stanzas=[StanzaTemplate(4, syllables_per_line=[11, 11, 11, 5],
                            meter=MeterConstraint.QUANTITATIVE)],
    repeating=True,
    notes="Three lesser Sapphic lines (11 syl) + one Adonic (5 syl)",
)

BLUES_STANZA = PoeticForm(
    name="Blues Stanza",
    origin="African-American",
    stanzas=[StanzaTemplate(3, rhyme_scheme="AAB")],
    repeating=True,
    notes="First line stated, repeated (possibly varied), then resolved; 12-bar blues lyric form",
)

COUPLET_HEROIC = PoeticForm(
    name="Heroic Couplet",
    origin="English",
    stanzas=[StanzaTemplate(2, rhyme_scheme="AA", meter=MeterConstraint.IAMBIC_PENTAMETER)],
    repeating=True,
    notes="Pope, Dryden; end-stopped or enjambed",
)

DOHA = PoeticForm(
    name="Dohā",
    origin="Hindi/Rajasthani",
    stanzas=[StanzaTemplate(2, syllables_per_line=[24, 24])],
    repeating=True,
    notes="Each line = 13 + 11 matra (beats). Kabir, Rahim, Tulsidas.",
)

COMMON_METER = PoeticForm(
    name="Common Meter",
    origin="English (hymn)",
    stanzas=[
        StanzaTemplate(4, syllables_per_line=[8, 6, 8, 6], rhyme_scheme="ABAB",
                       meter=MeterConstraint.COMMON_METER),
    ],
    repeating=True,
    notes="8-6-8-6 iambic; Emily Dickinson; fits 'Amazing Grace'",
)

AWDL = PoeticForm(
    name="Awdl",
    origin="Welsh",
    stanzas=[StanzaTemplate(4)],
    repeating=True,
    notes="Cynghanedd (consonant harmony) required; complex internal sound patterns",
)


FORM_REGISTRY: dict[str, PoeticForm] = {
    "shakespearean_sonnet": SONNET_SHAKESPEAREAN,
    "petrarchan_sonnet": SONNET_PETRARCHAN,
    "spenserian_sonnet": SONNET_SPENSERIAN,
    "haiku": HAIKU,
    "tanka": TANKA,
    "ghazal": GHAZAL,
    "villanelle": VILLANELLE,
    "sestina": SESTINA,
    "limerick": LIMERICK,
    "rubaiyat": RUBA_I,
    "pantoum": PANTOUM,
    "terza_rima": TERZA_RIMA,
    "ottava_rima": OTTAVA_RIMA,
    "ballad": BALLAD,
    "horatian_ode": ODE_HORATIAN,
    "pindaric_ode": ODE_PINDARIC,
    "sijo": SIJO,
    "qasida": QASIDA,
    "renga": RENGA_UNIT,
    "alexandrine_couplet": ALEXANDRINE_COUPLET,
    "sapphic_stanza": SAPPHIC_STANZA,
    "blues_stanza": BLUES_STANZA,
    "heroic_couplet": COUPLET_HEROIC,
    "doha": DOHA,
    "common_meter": COMMON_METER,
    "awdl": AWDL,
}


def get_form(name: str) -> PoeticForm | None:
    return FORM_REGISTRY.get(name.lower().replace(" ", "_").replace("'", "").replace("ā", "a"))


def list_forms() -> list[str]:
    return sorted(FORM_REGISTRY.keys())


def forms_by_origin(origin: str) -> list[PoeticForm]:
    origin_lower = origin.lower()
    return [f for f in FORM_REGISTRY.values() if origin_lower in f.origin.lower()]


def validate_line_count(lines: list[str], form: PoeticForm) -> bool:
    if form.total_lines is not None:
        return len(lines) == form.total_lines
    if form.repeating:
        unit = form.stanzas[0].lines if form.stanzas else 0
        return unit > 0 and len(lines) % unit == 0
    return len(lines) >= form.min_lines


def validate_syllable_counts(syllable_counts: list[int], form: PoeticForm,
                             tolerance: int = 1) -> list[bool]:
    smap = form.syllable_map()
    results = []
    for i, count in enumerate(syllable_counts):
        if i < len(smap) and smap[i] is not None:
            results.append(abs(count - smap[i]) <= tolerance)
        else:
            results.append(True)
    return results
