"""Multilingual poetry and lyrics analysis: corpus sources, prosody, cross-lingual comparison."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusSource:
    name: str
    language: str
    iso639: str
    licence: str
    tradition: str = ""
    era: str = ""
    estimated_size: int = 0
    description: str = ""


OPEN_CORPUS_SOURCES: list[CorpusSource] = [
    CorpusSource("Project Gutenberg Poetry", "English", "en", "Public Domain",
                 "Western", "mixed", 50000, "English-language poetry from PG"),
    CorpusSource("Chinese Poetry Dataset", "Chinese", "zh", "Public Domain",
                 "Chinese Classical", "Tang-Song-Yuan", 310000,
                 "Tang, Song, Yuan dynasty poems — 全唐诗 + 宋词 + 元曲"),
    CorpusSource("Sanskrit Kavya Corpus", "Sanskrit", "sa", "Public Domain",
                 "South Asian", "Classical", 5000,
                 "Kalidasa, Bharavi, Magha, Bhartrihari collections"),
    CorpusSource("Hindi Doha Collection", "Hindi", "hi", "Public Domain",
                 "South Asian", "Medieval", 8000,
                 "Kabir, Tulsidas, Rahim, Surdas dohas and pads"),
    CorpusSource("Japanese Haiku Collection", "Japanese", "ja", "Public Domain",
                 "East Asian", "Edo", 20000,
                 "Basho, Buson, Issa, Shiki haiku"),
    CorpusSource("Japanese Manyoshu", "Japanese", "ja", "Public Domain",
                 "East Asian", "Nara", 4516,
                 "Oldest extant anthology of Japanese waka poetry"),
    CorpusSource("Arabic Classical Poetry", "Arabic", "ar", "Public Domain",
                 "Middle Eastern", "Pre-Islamic to Abbasid", 30000,
                 "Al-Muallaqat, Mutanabbi, Abu Nuwas, Rumi (Arabic)"),
    CorpusSource("Persian Divan Poetry", "Persian", "fa", "Public Domain",
                 "Middle Eastern", "Medieval", 40000,
                 "Rumi, Hafez, Saadi, Khayyam, Ferdowsi (Shahnameh)"),
    CorpusSource("Turkish Divan Poetry", "Turkish", "tr", "Public Domain",
                 "Middle Eastern", "Ottoman", 15000,
                 "Fuzuli, Baki, Nedim, Nabi gazel and kaside"),
    CorpusSource("Latin Poetry Corpus", "Latin", "la", "Public Domain",
                 "Western Classical", "Antiquity", 12000,
                 "Virgil, Ovid, Horace, Catullus, Lucretius"),
    CorpusSource("Ancient Greek Poetry", "Greek", "grc", "Public Domain",
                 "Western Classical", "Antiquity", 8000,
                 "Homer, Sappho, Pindar, Aeschylus choral odes"),
    CorpusSource("Kalevala (Finnish)", "Finnish", "fi", "Public Domain",
                 "Nordic", "19th century collection", 22795,
                 "Lönnrot's compilation — trochaic tetrameter oral tradition"),
    CorpusSource("Edda (Old Norse)", "Old Norse", "non", "Public Domain",
                 "Nordic", "Medieval", 3000,
                 "Poetic Edda — fornyrðislag and ljóðaháttr meters"),
    CorpusSource("Korean Sijo Collection", "Korean", "ko", "Public Domain",
                 "East Asian", "Joseon", 6000,
                 "Classical sijo: 3-line, 44-46 syllable form"),
    CorpusSource("Spanish Copla & Romancero", "Spanish", "es", "Public Domain",
                 "Iberian", "Medieval-Golden Age", 25000,
                 "Romancero Viejo, Lorca, Machado (pre-1930)"),
    CorpusSource("Portuguese Cantigas", "Portuguese", "pt", "Public Domain",
                 "Iberian", "Medieval-Modern", 10000,
                 "Cantigas de amigo/amor + Brazilian Modernist poetry"),
    CorpusSource("French Chanson & Verse", "French", "fr", "Public Domain",
                 "Western", "Medieval-Modern", 20000,
                 "Troubadour, Ronsard, Hugo, Baudelaire, Verlaine, Rimbaud"),
    CorpusSource("German Lieder Texts", "German", "de", "Public Domain",
                 "Western", "Romantic", 15000,
                 "Goethe, Schiller, Heine, Rilke, Mörike art song texts"),
    CorpusSource("Irish/Scottish Gaelic Poetry", "Gaelic", "ga", "Public Domain",
                 "Celtic", "Medieval-Modern", 4000,
                 "Bardic verse, amhrán meter, Ossian tradition"),
    CorpusSource("Ethiopian Qene Poetry", "Amharic/Ge'ez", "am", "Public Domain",
                 "East African", "Medieval-Modern", 3000,
                 "Double-meaning (wax and gold) ecclesiastical verse"),
    CorpusSource("Swahili Utenzi/Shairi", "Swahili", "sw", "Public Domain",
                 "East African", "18th-20th century", 5000,
                 "Utenzi wa Tambuka, Shairi verse forms"),
    CorpusSource("Yoruba Oriki Poetry", "Yoruba", "yo", "Public Domain",
                 "West African", "Oral tradition", 2000,
                 "Praise poetry — oriki orile, oriki personal"),
    CorpusSource("Vietnamese Ca Dao", "Vietnamese", "vi", "Public Domain",
                 "Southeast Asian", "Folk", 8000,
                 "Lục bát (6-8 syllable couplets) folk poetry"),
    CorpusSource("Malay/Indonesian Pantun", "Malay", "ms", "Public Domain",
                 "Southeast Asian", "Folk-Classical", 6000,
                 "ABAB quatrain with foreshadowing/meaning couplets"),
    CorpusSource("Tamil Sangam Poetry", "Tamil", "ta", "Public Domain",
                 "South Asian", "Classical", 2381,
                 "Sangam anthologies: Kuruntokai, Akananuru, Purananuru"),
    CorpusSource("Hebrew Liturgical Poetry", "Hebrew", "he", "Public Domain",
                 "Middle Eastern", "Medieval", 5000,
                 "Piyyut: Yannai, Eleazar ha-Kalir, Ibn Gabirol"),
    CorpusSource("Nahuatl Flower Songs", "Nahuatl", "nah", "Public Domain",
                 "Mesoamerican", "Pre-Columbian", 500,
                 "Cantares Mexicanos, Romances de los Señores"),
    CorpusSource("Quechua Harawi", "Quechua", "qu", "Public Domain",
                 "Andean", "Pre-Columbian to Colonial", 300,
                 "Love songs and elegies from Inca oral tradition"),
]


METER_TYPES: dict[str, dict] = {
    "iambic_pentameter": {"feet": 5, "pattern": "01", "tradition": "English"},
    "iambic_tetrameter": {"feet": 4, "pattern": "01", "tradition": "English"},
    "trochaic_tetrameter": {"feet": 4, "pattern": "10", "tradition": "Finnish/English"},
    "dactylic_hexameter": {"feet": 6, "pattern": "100", "tradition": "Greek/Latin"},
    "alexandrine": {"feet": 6, "pattern": "01", "tradition": "French"},
    "hendecasyllable": {"syllables": 11, "pattern": "mixed", "tradition": "Italian"},
    "octosyllable": {"syllables": 8, "pattern": "mixed", "tradition": "Spanish/French"},
    "haiku_mora": {"mora": [5, 7, 5], "tradition": "Japanese"},
    "tanka_mora": {"mora": [5, 7, 5, 7, 7], "tradition": "Japanese"},
    "sijo_syllable": {"syllables": [14, 14, 14], "tradition": "Korean"},
    "ghazal_bahr": {"pattern": "quantitative", "tradition": "Arabic/Persian/Urdu"},
    "shi_regulated": {"characters": [5, 7], "tradition": "Chinese"},
    "ci_cipai": {"pattern": "tonal", "tradition": "Chinese Song"},
    "fornyrdislag": {"syllables": [4, 4], "alliterative": True, "tradition": "Old Norse"},
    "doha": {"matras": [13, 11], "tradition": "Hindi"},
    "luc_bat": {"syllables": [6, 8], "tradition": "Vietnamese"},
    "pantun": {"lines": 4, "syllables_per_line": [8, 12], "tradition": "Malay"},
    "free_verse": {"pattern": "none", "tradition": "Modern/Universal"},
    "blank_verse": {"feet": 5, "pattern": "01", "rhyme": False, "tradition": "English"},
    "sprung_rhythm": {"pattern": "stress-timed", "tradition": "English (Hopkins)"},
}


@dataclass
class ProsodyAnalysis:
    meter_type: str = "unknown"
    syllable_counts: list[int] = field(default_factory=list)
    stress_regularity: float = 0.0
    line_length_variance: float = 0.0
    enjambment_ratio: float = 0.0
    caesura_positions: list[float] = field(default_factory=list)
    rhyme_density: float = 0.0
    alliteration_density: float = 0.0


class MultilingualTokenizer:
    _CJK_RANGES = [
        (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
        (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
    ]
    _HIRAGANA = (0x3040, 0x309F)
    _KATAKANA = (0x30A0, 0x30FF)
    _DEVANAGARI = (0x0900, 0x097F)
    _ARABIC = (0x0600, 0x06FF)
    _HANGUL = (0xAC00, 0xD7AF)

    def _is_cjk(self, ch: str) -> bool:
        cp = ord(ch)
        return any(lo <= cp <= hi for lo, hi in self._CJK_RANGES)

    def _is_kana(self, ch: str) -> bool:
        cp = ord(ch)
        return self._HIRAGANA[0] <= cp <= self._HIRAGANA[1] or self._KATAKANA[0] <= cp <= self._KATAKANA[1]

    def tokenize(self, text: str, language: str = "auto") -> list[str]:
        if language == "auto":
            language = self.detect_script(text)
        if language in ("zh", "ja"):
            return self._tokenize_cjk(text)
        if language in ("ko",):
            return self._tokenize_hangul(text)
        return self._tokenize_alphabetic(text)

    def _tokenize_cjk(self, text: str) -> list[str]:
        tokens: list[str] = []
        buf = []
        for ch in text:
            if self._is_cjk(ch) or self._is_kana(ch):
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                tokens.append(ch)
            elif ch.isspace() or ch in "，。、！？；：""''()《》「」":
                if buf:
                    tokens.append("".join(buf))
                    buf = []
            else:
                buf.append(ch)
        if buf:
            tokens.append("".join(buf))
        return tokens

    def _tokenize_hangul(self, text: str) -> list[str]:
        return [w for w in re.split(r'\s+', text.strip()) if w]

    def _tokenize_alphabetic(self, text: str) -> list[str]:
        return re.findall(r"[\w']+", text)

    def detect_script(self, text: str) -> str:
        counts: dict[str, int] = {}
        for ch in text:
            if self._is_cjk(ch):
                counts["zh"] = counts.get("zh", 0) + 1
            elif self._is_kana(ch):
                counts["ja"] = counts.get("ja", 0) + 1
            elif self._DEVANAGARI[0] <= ord(ch) <= self._DEVANAGARI[1]:
                counts["hi"] = counts.get("hi", 0) + 1
            elif self._ARABIC[0] <= ord(ch) <= self._ARABIC[1]:
                counts["ar"] = counts.get("ar", 0) + 1
            elif self._HANGUL[0] <= ord(ch) <= self._HANGUL[1]:
                counts["ko"] = counts.get("ko", 0) + 1
            elif ch.isalpha():
                counts["lat"] = counts.get("lat", 0) + 1
        if not counts:
            return "en"
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    def syllable_count(self, text: str, language: str = "en") -> int:
        if language in ("zh",):
            return sum(1 for ch in text if self._is_cjk(ch))
        if language in ("ja",):
            return sum(1 for ch in text if self._is_cjk(ch) or self._is_kana(ch))
        if language in ("ko",):
            return sum(1 for ch in text if self._HANGUL[0] <= ord(ch) <= self._HANGUL[1])
        return self._syllable_count_alphabetic(text)

    def _syllable_count_alphabetic(self, text: str) -> int:
        words = self._tokenize_alphabetic(text.lower())
        total = 0
        for w in words:
            vowels = re.findall(r'[aeiouyàáâãäåèéêëìíîïòóôõöùúûüæœ]+', w)
            count = len(vowels)
            if w.endswith('e') and count > 1:
                count -= 1
            total += max(count, 1)
        return total


class MultilingualPoetryAnalyzer:
    def __init__(self) -> None:
        self._tok = MultilingualTokenizer()

    def analyze_prosody(self, text: str, language: str = "en") -> ProsodyAnalysis:
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            return ProsodyAnalysis()
        syl_counts = [self._tok.syllable_count(l, language) for l in lines]
        mean_syl = sum(syl_counts) / len(syl_counts) if syl_counts else 0
        variance = sum((s - mean_syl) ** 2 for s in syl_counts) / len(syl_counts) if syl_counts else 0

        rhyme_d = self._rhyme_density(lines)
        allit_d = self._alliteration_density(lines)
        enjamb = self._enjambment_ratio(lines)
        caesura = self._caesura_positions(lines)
        meter = self._classify_meter(syl_counts, language)

        return ProsodyAnalysis(
            meter_type=meter,
            syllable_counts=syl_counts,
            stress_regularity=1.0 / (1.0 + math.sqrt(variance)),
            line_length_variance=variance,
            enjambment_ratio=enjamb,
            caesura_positions=caesura,
            rhyme_density=rhyme_d,
            alliteration_density=allit_d,
        )

    def _classify_meter(self, syl_counts: list[int], lang: str) -> str:
        if not syl_counts:
            return "unknown"
        mean = sum(syl_counts) / len(syl_counts)
        if lang == "ja":
            if len(syl_counts) == 3 and syl_counts == [5, 7, 5]:
                return "haiku"
            if len(syl_counts) == 5 and syl_counts == [5, 7, 5, 7, 7]:
                return "tanka"
        if lang == "zh":
            if all(s == 5 for s in syl_counts):
                return "wuyan (five-character)"
            if all(s == 7 for s in syl_counts):
                return "qiyan (seven-character)"
        if lang == "vi" and len(syl_counts) >= 2:
            pairs = [(syl_counts[i], syl_counts[i+1]) for i in range(0, len(syl_counts) - 1, 2)]
            if all(a in (6,) and b in (8,) for a, b in pairs):
                return "luc_bat"
        variance = sum((s - mean) ** 2 for s in syl_counts) / len(syl_counts)
        cv = math.sqrt(variance) / mean if mean > 0 else 1.0
        if cv < 0.1:
            if 9 <= mean <= 11:
                return "iambic_pentameter"
            if 7 <= mean <= 9:
                return "iambic_tetrameter"
            if 11 <= mean <= 13:
                return "alexandrine"
            return f"regular_{int(round(mean))}_syllable"
        if cv > 0.4:
            return "free_verse"
        return "mixed_meter"

    def _rhyme_density(self, lines: list[str]) -> float:
        if len(lines) < 2:
            return 0.0
        endings = []
        for l in lines:
            words = re.findall(r"[\w']+", l.lower())
            if words:
                w = words[-1]
                endings.append(w[-3:] if len(w) >= 3 else w)
            else:
                endings.append("")
        rhymes = 0
        for i in range(len(endings)):
            for j in range(i + 1, min(i + 5, len(endings))):
                if endings[i] and endings[j] and endings[i] == endings[j]:
                    rhymes += 1
                    break
        return rhymes / len(lines)

    def _alliteration_density(self, lines: list[str]) -> float:
        if not lines:
            return 0.0
        total_words = 0
        alliterating = 0
        for l in lines:
            words = re.findall(r"[\w']+", l.lower())
            if len(words) < 2:
                continue
            total_words += len(words)
            for i in range(1, len(words)):
                if words[i][0] == words[i-1][0] and words[i][0].isalpha():
                    alliterating += 1
        return alliterating / total_words if total_words > 0 else 0.0

    def _enjambment_ratio(self, lines: list[str]) -> float:
        if len(lines) < 2:
            return 0.0
        enjambed = 0
        for l in lines[:-1]:
            stripped = l.rstrip()
            if stripped and stripped[-1] not in ".!?;:,—–":
                enjambed += 1
        return enjambed / (len(lines) - 1)

    def _caesura_positions(self, lines: list[str]) -> list[float]:
        positions: list[float] = []
        for l in lines:
            for sep in (",", ";", "—", "–", "||", "//"):
                idx = l.find(sep)
                if idx > 0:
                    positions.append(idx / len(l))
                    break
        return positions

    def cross_lingual_similarity(self, text_a: str, lang_a: str,
                                  text_b: str, lang_b: str) -> float:
        pa = self.analyze_prosody(text_a, lang_a)
        pb = self.analyze_prosody(text_b, lang_b)
        features_a = [
            pa.stress_regularity,
            pa.rhyme_density,
            pa.alliteration_density,
            pa.enjambment_ratio,
            pa.line_length_variance / (pa.line_length_variance + 1),
        ]
        features_b = [
            pb.stress_regularity,
            pb.rhyme_density,
            pb.alliteration_density,
            pb.enjambment_ratio,
            pb.line_length_variance / (pb.line_length_variance + 1),
        ]
        dot = sum(a * b for a, b in zip(features_a, features_b))
        mag_a = math.sqrt(sum(a * a for a in features_a))
        mag_b = math.sqrt(sum(b * b for b in features_b))
        if mag_a < 1e-12 or mag_b < 1e-12:
            return 0.0
        return dot / (mag_a * mag_b)

    def meter_distribution(self, texts_with_langs: list[tuple[str, str]]) -> dict[str, int]:
        dist: dict[str, int] = {}
        for text, lang in texts_with_langs:
            pa = self.analyze_prosody(text, lang)
            dist[pa.meter_type] = dist.get(pa.meter_type, 0) + 1
        return dist
