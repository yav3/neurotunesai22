"""Style-specific arrangement templates."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ArrangingStyle(Enum):
    BIG_BAND = "big_band"
    STRING_QUARTET = "string_quartet"
    CHAMBER = "chamber"
    CHORAL_SATB = "choral_satb"
    JAZZ_COMBO = "jazz_combo"
    ORCHESTRAL = "orchestral"
    ROCK_BAND = "rock_band"
    ELECTRONIC = "electronic"


@dataclass
class StyleTemplate:
    style: ArrangingStyle
    instruments: list[str]
    typical_ranges: dict[str, tuple[int, int]]
    typical_textures: list[str]
    voice_count: int


STYLE_TEMPLATES: dict[ArrangingStyle, StyleTemplate] = {
    ArrangingStyle.BIG_BAND: StyleTemplate(
        style=ArrangingStyle.BIG_BAND,
        instruments=["trumpet1", "trumpet2", "trumpet3", "trumpet4",
                     "trombone1", "trombone2", "trombone3", "trombone4",
                     "alto_sax1", "alto_sax2", "tenor_sax1", "tenor_sax2", "bari_sax",
                     "piano", "bass", "drums"],
        typical_ranges={"trumpet1": (55, 84), "trombone1": (40, 72),
                        "alto_sax1": (56, 80), "tenor_sax1": (49, 75),
                        "bari_sax": (36, 68), "piano": (21, 108), "bass": (28, 60)},
        typical_textures=["block_chords", "soli", "shout_chorus", "unison_melody"],
        voice_count=16,
    ),
    ArrangingStyle.STRING_QUARTET: StyleTemplate(
        style=ArrangingStyle.STRING_QUARTET,
        instruments=["violin1", "violin2", "viola", "cello"],
        typical_ranges={"violin1": (55, 96), "violin2": (55, 88),
                        "viola": (48, 84), "cello": (36, 76)},
        typical_textures=["homophonic", "fugal", "melody_accompaniment"],
        voice_count=4,
    ),
    ArrangingStyle.CHAMBER: StyleTemplate(
        style=ArrangingStyle.CHAMBER,
        instruments=["flute", "oboe", "clarinet", "bassoon", "horn"],
        typical_ranges={"flute": (60, 96), "oboe": (58, 87),
                        "clarinet": (50, 89), "bassoon": (34, 72), "horn": (41, 77)},
        typical_textures=["dialogic", "homophonic", "solo_with_support"],
        voice_count=5,
    ),
    ArrangingStyle.CHORAL_SATB: StyleTemplate(
        style=ArrangingStyle.CHORAL_SATB,
        instruments=["soprano", "alto", "tenor", "bass"],
        typical_ranges={"soprano": (60, 81), "alto": (55, 74),
                        "tenor": (48, 69), "bass": (40, 62)},
        typical_textures=["homophonic", "antiphonal", "canon", "free_polyphony"],
        voice_count=4,
    ),
    ArrangingStyle.JAZZ_COMBO: StyleTemplate(
        style=ArrangingStyle.JAZZ_COMBO,
        instruments=["lead", "piano", "bass", "drums"],
        typical_ranges={"lead": (55, 84), "piano": (21, 108), "bass": (28, 60)},
        typical_textures=["head_arrangement", "comping", "walking_bass"],
        voice_count=4,
    ),
    ArrangingStyle.ORCHESTRAL: StyleTemplate(
        style=ArrangingStyle.ORCHESTRAL,
        instruments=["violin1", "violin2", "viola", "cello", "contrabass",
                     "flute", "oboe", "clarinet", "bassoon",
                     "horn", "trumpet", "trombone", "tuba",
                     "timpani", "percussion"],
        typical_ranges={"violin1": (55, 96), "cello": (36, 76),
                        "flute": (60, 96), "trumpet": (55, 84),
                        "contrabass": (28, 60), "tuba": (28, 58)},
        typical_textures=["tutti", "antiphonal", "solo_with_orchestra", "tremolo_sustain"],
        voice_count=15,
    ),
    ArrangingStyle.ROCK_BAND: StyleTemplate(
        style=ArrangingStyle.ROCK_BAND,
        instruments=["vocal", "guitar1", "guitar2", "bass", "drums"],
        typical_ranges={"vocal": (48, 79), "guitar1": (40, 84),
                        "guitar2": (40, 84), "bass": (28, 60)},
        typical_textures=["power_chords", "riff_based", "call_response"],
        voice_count=5,
    ),
    ArrangingStyle.ELECTRONIC: StyleTemplate(
        style=ArrangingStyle.ELECTRONIC,
        instruments=["lead_synth", "pad", "bass_synth", "drums", "fx"],
        typical_ranges={"lead_synth": (48, 96), "pad": (36, 84),
                        "bass_synth": (24, 60)},
        typical_textures=["layered", "arpeggiated", "ambient_pad", "rhythmic_sequence"],
        voice_count=5,
    ),
}


def assign_voices(
    melody: list[int],
    chord_pcs: list[int],
    style: ArrangingStyle,
) -> dict[str, list[int]]:
    """Distribute melody and harmony across a style's instruments."""
    template = STYLE_TEMPLATES[style]
    instruments = template.instruments
    result: dict[str, list[int]] = {}
    result[instruments[0]] = list(melody)
    if len(instruments) > 1 and chord_pcs:
        for i, pc in enumerate(chord_pcs):
            inst_idx = 1 + (i % (len(instruments) - 1))
            inst = instruments[inst_idx]
            lo = template.typical_ranges.get(inst, (48, 72))[0]
            octave = lo // 12
            pitch = pc % 12 + octave * 12
            result.setdefault(inst, []).append(pitch)
    return result


def double_at_octave(pitches: list[int], interval: int = 12) -> list[int]:
    """Add octave doublings to a pitch list."""
    return pitches + [p + interval for p in pitches]


def unison_passage(pitches: list[int], instruments: list[str]) -> dict[str, list[int]]:
    """All instruments play the same pitches."""
    return {inst: list(pitches) for inst in instruments}


def tutti_score(
    melody: list[int],
    harmony: list[list[int]],
    style: ArrangingStyle,
) -> dict[str, list[list[int]]]:
    """Full arrangement: melody on lead, harmony distributed across remaining voices."""
    template = STYLE_TEMPLATES[style]
    instruments = template.instruments
    result: dict[str, list[list[int]]] = {inst: [] for inst in instruments}
    for beat_idx, mel_note in enumerate(melody):
        result[instruments[0]].append([mel_note])
        if beat_idx < len(harmony):
            chord = harmony[beat_idx]
            for i, pc in enumerate(chord):
                inst_idx = 1 + (i % (len(instruments) - 1))
                inst = instruments[inst_idx]
                lo = template.typical_ranges.get(inst, (48, 72))[0]
                pitch = pc % 12 + (lo // 12) * 12
                result[inst].append([pitch])
    return result
