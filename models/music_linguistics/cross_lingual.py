"""Cross-lingual analysis: script detection, transliteration, language features for singing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScriptFamily(Enum):
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    GREEK = "greek"
    ARABIC = "arabic"
    DEVANAGARI = "devanagari"
    CJK = "cjk"
    HANGUL = "hangul"
    KANA = "kana"
    THAI = "thai"
    ETHIOPIC = "ethiopic"
    HEBREW = "hebrew"
    GEORGIAN = "georgian"
    ARMENIAN = "armenian"
    BENGALI = "bengali"
    TAMIL = "tamil"
    UNKNOWN = "unknown"


_SCRIPT_RANGES: list[tuple[int, int, ScriptFamily]] = [
    (0x0041, 0x024F, ScriptFamily.LATIN),
    (0x0400, 0x04FF, ScriptFamily.CYRILLIC),
    (0x0370, 0x03FF, ScriptFamily.GREEK),
    (0x0600, 0x06FF, ScriptFamily.ARABIC),
    (0x0900, 0x097F, ScriptFamily.DEVANAGARI),
    (0x0980, 0x09FF, ScriptFamily.BENGALI),
    (0x0B80, 0x0BFF, ScriptFamily.TAMIL),
    (0x0E00, 0x0E7F, ScriptFamily.THAI),
    (0x10A0, 0x10FF, ScriptFamily.GEORGIAN),
    (0x0530, 0x058F, ScriptFamily.ARMENIAN),
    (0x0590, 0x05FF, ScriptFamily.HEBREW),
    (0x1200, 0x137F, ScriptFamily.ETHIOPIC),
    (0x3040, 0x30FF, ScriptFamily.KANA),
    (0x3400, 0x9FFF, ScriptFamily.CJK),
    (0xAC00, 0xD7AF, ScriptFamily.HANGUL),
    (0xF900, 0xFAFF, ScriptFamily.CJK),
    (0x20000, 0x2A6DF, ScriptFamily.CJK),
]


def detect_script(char: str) -> ScriptFamily:
    cp = ord(char)
    for lo, hi, fam in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return fam
    return ScriptFamily.UNKNOWN


def detect_scripts(text: str) -> dict[ScriptFamily, int]:
    counts: dict[ScriptFamily, int] = {}
    for ch in text:
        if ch.isalpha():
            s = detect_script(ch)
            counts[s] = counts.get(s, 0) + 1
    return counts


def dominant_script(text: str) -> ScriptFamily:
    counts = detect_scripts(text)
    if not counts:
        return ScriptFamily.UNKNOWN
    return max(counts, key=lambda k: counts[k])


@dataclass
class LanguageProfile:
    code: str
    name: str
    script: ScriptFamily
    syllable_structure: str
    tonal: bool = False
    mora_timed: bool = False
    stress_timed: bool = False
    syllable_timed: bool = False
    singing_notes: str = ""


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "en": LanguageProfile("en", "English", ScriptFamily.LATIN, "(C)(C)(C)V(C)(C)(C)(C)",
                          stress_timed=True,
                          singing_notes="Schwa reduction complicates sustained vowels"),
    "es": LanguageProfile("es", "Spanish", ScriptFamily.LATIN, "(C)V(C)",
                          syllable_timed=True,
                          singing_notes="Open syllables ideal for bel canto; 5 pure vowels"),
    "it": LanguageProfile("it", "Italian", ScriptFamily.LATIN, "(C)(C)V(C)(C)",
                          syllable_timed=True,
                          singing_notes="Classical singing language; open vowels, gemination"),
    "fr": LanguageProfile("fr", "French", ScriptFamily.LATIN, "(C)(C)(C)V(C)(C)(C)",
                          syllable_timed=True,
                          singing_notes="Nasal vowels, liaison, mute-e conventions"),
    "de": LanguageProfile("de", "German", ScriptFamily.LATIN, "(C)(C)(C)V(C)(C)(C)(C)",
                          stress_timed=True,
                          singing_notes="Vowel length contrast; Auslautverhärtung"),
    "pt": LanguageProfile("pt", "Portuguese", ScriptFamily.LATIN, "(C)(C)V(C)(C)",
                          syllable_timed=True,
                          singing_notes="Nasal vowels; bossa nova legato style"),
    "ja": LanguageProfile("ja", "Japanese", ScriptFamily.KANA, "(C)V",
                          mora_timed=True,
                          singing_notes="Mora = rhythmic unit; each kana = 1 note in enka/J-pop"),
    "ko": LanguageProfile("ko", "Korean", ScriptFamily.HANGUL, "(C)V(C)",
                          syllable_timed=True,
                          singing_notes="Jamo blocks; tensed/aspirated contrast in rap delivery"),
    "zh": LanguageProfile("zh", "Mandarin", ScriptFamily.CJK, "(C)V(N)",
                          tonal=True,
                          singing_notes="Tonal contour–melody interaction; most pop ignores tones"),
    "ar": LanguageProfile("ar", "Arabic", ScriptFamily.ARABIC, "(C)V(C)(C)",
                          stress_timed=True,
                          singing_notes="Pharyngeals, melismatic maqam singing tradition"),
    "hi": LanguageProfile("hi", "Hindi", ScriptFamily.DEVANAGARI, "(C)(C)V(C)(C)",
                          mora_timed=True,
                          singing_notes="Aspirated stops; Bollywood vocal ornamentation"),
    "sw": LanguageProfile("sw", "Swahili", ScriptFamily.LATIN, "(N)(C)V",
                          syllable_timed=True,
                          singing_notes="Prenasalized onsets; Bantu 5-vowel system"),
    "tr": LanguageProfile("tr", "Turkish", ScriptFamily.LATIN, "(C)V(C)(C)",
                          syllable_timed=True,
                          singing_notes="Vowel harmony; agglutinative suffixes affect phrasing"),
    "ru": LanguageProfile("ru", "Russian", ScriptFamily.CYRILLIC, "(C)(C)(C)(C)V(C)(C)(C)(C)",
                          stress_timed=True,
                          singing_notes="Vowel reduction off-stress; rich consonant clusters"),
    "el": LanguageProfile("el", "Greek", ScriptFamily.GREEK, "(C)(C)V(C)(C)",
                          syllable_timed=True,
                          singing_notes="Rebetiko microtonal inflection; stress accent"),
    "am": LanguageProfile("am", "Amharic", ScriptFamily.ETHIOPIC, "CV(C)",
                          stress_timed=True,
                          singing_notes="Fidel syllabary; ejective consonants in traditional music"),
    "ta": LanguageProfile("ta", "Tamil", ScriptFamily.TAMIL, "(C)V(C)",
                          mora_timed=True,
                          singing_notes="Carnatic music tradition; retroflex series"),
    "th": LanguageProfile("th", "Thai", ScriptFamily.THAI, "(C)(C)V(C)",
                          tonal=True,
                          singing_notes="5-tone system; luk thung / mor lam traditions"),
    "ka": LanguageProfile("ka", "Georgian", ScriptFamily.GEORGIAN, "(C)(C)(C)(C)V(C)(C)",
                          stress_timed=True,
                          singing_notes="Polyphonic singing (UNESCO); complex clusters"),
    "hy": LanguageProfile("hy", "Armenian", ScriptFamily.ARMENIAN, "(C)(C)V(C)(C)",
                          stress_timed=True,
                          singing_notes="Duduk tradition; final-stress pattern"),
}


ROMANIZATION_TABLES: dict[str, dict[str, str]] = {
    "cyrillic_basic": {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "ye", "ё": "yo",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    },
    "hiragana_romaji": {
        "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
        "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
        "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
        "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
        "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
        "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
        "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
        "や": "ya", "ゆ": "yu", "よ": "yo",
        "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
        "わ": "wa", "を": "wo", "ん": "n",
    },
    "hangul_onset": {
        "ㄱ": "g", "ㄲ": "kk", "ㄴ": "n", "ㄷ": "d", "ㄸ": "tt",
        "ㄹ": "r", "ㅁ": "m", "ㅂ": "b", "ㅃ": "pp", "ㅅ": "s",
        "ㅆ": "ss", "ㅇ": "", "ㅈ": "j", "ㅉ": "jj", "ㅊ": "ch",
        "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "h",
    },
}


def transliterate(text: str, table_name: str) -> str:
    table = ROMANIZATION_TABLES.get(table_name, {})
    result = []
    for ch in text:
        lower = ch.lower()
        if lower in table:
            mapped = table[lower]
            result.append(mapped.upper() if ch.isupper() and mapped else mapped)
        else:
            result.append(ch)
    return "".join(result)


@dataclass
class SingabilityProfile:
    language: str
    open_syllable_ratio: float
    vowel_richness: int
    tonal_complexity: float
    rhythm_regularity: str
    overall_score: float


def estimate_singing_difficulty(lang_code: str) -> SingabilityProfile:
    profile = LANGUAGE_PROFILES.get(lang_code)
    if profile is None:
        return SingabilityProfile(lang_code, 0.5, 5, 0.0, "unknown", 0.5)

    structure = profile.syllable_structure
    open_ratio = 0.7 if structure.endswith("V") or structure.endswith("V(C)") else 0.4
    vowel_richness = {"en": 14, "es": 5, "it": 7, "fr": 15, "de": 15, "pt": 13,
                      "ja": 5, "ko": 10, "zh": 7, "ar": 6, "hi": 10, "sw": 5,
                      "tr": 8, "ru": 10, "el": 5, "am": 7, "ta": 5, "th": 9,
                      "ka": 5, "hy": 6}.get(lang_code, 5)
    tonal = 0.3 if profile.tonal else 0.0
    rhythm = "syllable" if profile.syllable_timed else "stress" if profile.stress_timed else "mora"
    rhythm_bonus = {"syllable": 0.1, "mora": 0.15, "stress": 0.0}.get(rhythm, 0.0)
    score = min(1.0, 0.3 * open_ratio + 0.2 * (1.0 - vowel_richness / 20) + 0.2 * (1.0 - tonal) + 0.15 * rhythm_bonus + 0.15)
    return SingabilityProfile(
        language=profile.name,
        open_syllable_ratio=open_ratio,
        vowel_richness=vowel_richness,
        tonal_complexity=tonal,
        rhythm_regularity=rhythm,
        overall_score=round(score, 3),
    )


@dataclass
class TextStats:
    char_count: int
    word_count: int
    line_count: int
    scripts: dict[ScriptFamily, int]
    dominant_script: ScriptFamily
    unique_words: int
    lexical_density: float


def analyze_text(text: str) -> TextStats:
    lines = [l for l in text.split("\n") if l.strip()]
    words = text.split()
    unique = set(w.lower() for w in words)
    scripts = detect_scripts(text)
    ds = dominant_script(text) if text.strip() else ScriptFamily.UNKNOWN
    density = len(unique) / max(len(words), 1)
    return TextStats(
        char_count=len(text),
        word_count=len(words),
        line_count=len(lines),
        scripts=scripts,
        dominant_script=ds,
        unique_words=len(unique),
        lexical_density=round(density, 4),
    )
