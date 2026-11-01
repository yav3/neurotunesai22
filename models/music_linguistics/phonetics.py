"""Phonetic analysis: IPA inventories, syllable counting, phonotactics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


IPA_VOWELS = set("ɑɒæɐɶəɵɘɜɞɤɨʉʊɪɛɔʌaeiouɝ")
IPA_CONSONANTS = set("pbtdkɡʔmnɲŋɴfvθðszʃʒɕʑçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟrɾɽwβɸ")


@dataclass
class PhonemeInventory:
    language: str
    vowels: list[str]
    consonants: list[str]
    tones: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def total(self) -> int:
        return len(self.vowels) + len(self.consonants) + len(self.tones)

    @property
    def vowel_ratio(self) -> float:
        t = self.total
        return len(self.vowels) / t if t > 0 else 0.0


INVENTORIES: dict[str, PhonemeInventory] = {
    "english": PhonemeInventory(
        "English",
        vowels=["iː", "ɪ", "eɪ", "ɛ", "æ", "ɑː", "ɒ", "ɔː", "oʊ", "ʊ", "uː", "ʌ", "ə", "ɜː"],
        consonants=["p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z",
                     "ʃ", "ʒ", "h", "tʃ", "dʒ", "m", "n", "ŋ", "l", "ɹ", "j", "w"],
    ),
    "spanish": PhonemeInventory(
        "Spanish",
        vowels=["a", "e", "i", "o", "u"],
        consonants=["p", "b", "t", "d", "k", "ɡ", "f", "θ", "s", "x", "tʃ",
                     "m", "n", "ɲ", "l", "ʎ", "ɾ", "r", "j", "w"],
        notes="5-vowel system; highly singable due to open vowels",
    ),
    "japanese": PhonemeInventory(
        "Japanese",
        vowels=["a", "i", "ɯ", "e", "o"],
        consonants=["k", "ɡ", "s", "z", "t", "d", "n", "h", "b", "p", "m",
                     "j", "ɾ", "w", "ɴ", "ts", "tɕ", "dʑ", "ɕ"],
        notes="Mora-timed; CV syllable structure ideal for singing",
    ),
    "hindi": PhonemeInventory(
        "Hindi",
        vowels=["ə", "aː", "ɪ", "iː", "ʊ", "uː", "eː", "ɛː", "oː", "ɔː"],
        consonants=["p", "pʰ", "b", "bʱ", "t̪", "t̪ʰ", "d̪", "d̪ʱ", "ʈ", "ʈʰ",
                     "ɖ", "ɖʱ", "k", "kʰ", "ɡ", "ɡʱ", "tʃ", "tʃʰ", "dʒ", "dʒʱ",
                     "f", "s", "ʃ", "h", "m", "n", "ɳ", "l", "ɾ", "j", "ʋ"],
        notes="Aspirated/unaspirated contrast; nasal vowels in singing",
    ),
    "arabic": PhonemeInventory(
        "Arabic",
        vowels=["a", "aː", "i", "iː", "u", "uː"],
        consonants=["b", "t", "θ", "dʒ", "ħ", "x", "d", "ð", "r", "z", "s", "ʃ",
                     "sˤ", "dˤ", "tˤ", "ðˤ", "ʕ", "ɣ", "f", "q", "k", "l", "m",
                     "n", "h", "w", "j", "ʔ"],
        notes="Pharyngeal and emphatic consonants; melismatic singing tradition",
    ),
    "mandarin": PhonemeInventory(
        "Mandarin",
        vowels=["a", "o", "ɤ", "i", "u", "y", "ɚ"],
        consonants=["p", "pʰ", "m", "f", "t", "tʰ", "n", "l", "k", "kʰ", "x",
                     "tɕ", "tɕʰ", "ɕ", "ts", "tsʰ", "s", "tʂ", "tʂʰ", "ʂ", "ʐ"],
        tones=["1-high", "2-rising", "3-dipping", "4-falling", "5-neutral"],
        notes="Tone language; tonal contours interact with melodic contour",
    ),
    "french": PhonemeInventory(
        "French",
        vowels=["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ɔ", "a", "ɑ",
                 "ɛ̃", "œ̃", "ɔ̃", "ɑ̃"],
        consonants=["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ",
                     "m", "n", "ɲ", "ŋ", "l", "ʁ", "j", "w", "ɥ"],
        notes="Rich vowel system with nasals; liaison affects sung phrasing",
    ),
    "german": PhonemeInventory(
        "German",
        vowels=["iː", "ɪ", "yː", "ʏ", "eː", "ɛ", "øː", "œ", "aː", "a",
                 "oː", "ɔ", "uː", "ʊ", "ə"],
        consonants=["p", "b", "t", "d", "k", "ɡ", "ʔ", "pf", "ts", "tʃ", "f", "v",
                     "s", "z", "ʃ", "ʒ", "ç", "x", "h", "m", "n", "ŋ", "l", "ʁ", "j"],
        notes="Vowel length distinction; final devoicing in singing",
    ),
    "portuguese": PhonemeInventory(
        "Portuguese",
        vowels=["a", "ɐ", "e", "ɛ", "i", "o", "ɔ", "u", "ɐ̃", "ẽ", "ĩ", "õ", "ũ"],
        consonants=["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ",
                     "m", "n", "ɲ", "l", "ʎ", "ɾ", "ʁ", "j", "w"],
        notes="Nasal vowels central to bossa nova singing style",
    ),
    "korean": PhonemeInventory(
        "Korean",
        vowels=["a", "ɛ", "e", "o", "u", "ɯ", "i", "ø", "y", "ʌ"],
        consonants=["p", "pʰ", "p͈", "t", "tʰ", "t͈", "k", "kʰ", "k͈",
                     "tɕ", "tɕʰ", "t͈ɕ", "s", "s͈", "h", "m", "n", "ŋ", "l"],
        notes="Three-way laryngeal contrast (lax/aspirated/tense)",
    ),
    "swahili": PhonemeInventory(
        "Swahili",
        vowels=["a", "e", "i", "o", "u"],
        consonants=["b", "tʃ", "d", "ð", "f", "ɡ", "h", "dʒ", "k", "l", "m",
                     "n", "ɲ", "ŋ", "p", "ɾ", "s", "ʃ", "t", "θ", "v", "w", "j", "z"],
        notes="5-vowel Bantu system; prenasalized stops in singing",
    ),
}


def count_syllables_english(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    vowels = set("aeiouy")
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    return max(count, 1)


def count_syllables_spanish(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    vowels = set("aeiouáéíóú")
    strong = set("aeoáéó")
    count = 0
    prev_vowel = False
    prev_strong = False
    for ch in word:
        is_v = ch in vowels
        is_strong = ch in strong
        if is_v:
            if not prev_vowel:
                count += 1
            elif is_strong and prev_strong:
                count += 1
        prev_vowel = is_v
        prev_strong = is_strong if is_v else False
    return max(count, 1)


def count_syllables_japanese(text: str) -> int:
    mora = 0
    skip_next = False
    for i, ch in enumerate(text):
        if skip_next:
            skip_next = False
            continue
        if '぀' <= ch <= 'ゟ' or '゠' <= ch <= 'ヿ':
            mora += 1
            if ch in ('っ', 'ッ'):
                pass
        elif ch.isalpha():
            mora += 1
    return max(mora, 1) if text.strip() else 0


def singability_score(inventory: PhonemeInventory) -> float:
    v_ratio = inventory.vowel_ratio
    open_vowels = sum(1 for v in inventory.vowels if any(c in v for c in "aɑɒæ"))
    open_ratio = open_vowels / max(len(inventory.vowels), 1)
    tone_penalty = 0.1 * len(inventory.tones)
    return max(0.0, min(1.0, 0.4 * v_ratio + 0.4 * open_ratio + 0.2 - tone_penalty))
