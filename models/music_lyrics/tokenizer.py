"""Multilingual tokenizer with language detection and syllable estimation."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter


def normalize_text(text: str, strip_diacritics: bool = False, lowercase: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    if strip_diacritics:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    return text


_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0xF900, 0xFAFF),    # CJK Compatibility
    (0x2F800, 0x2FA1F),  # CJK Supplement
]

_HANGUL_RANGE = (0xAC00, 0xD7AF)
_HIRAGANA_RANGE = (0x3040, 0x309F)
_KATAKANA_RANGE = (0x30A0, 0x30FF)
_THAI_RANGE = (0x0E00, 0x0E7F)
_ARABIC_RANGE = (0x0600, 0x06FF)
_DEVANAGARI_RANGE = (0x0900, 0x097F)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_hangul(ch: str) -> bool:
    cp = ord(ch)
    return _HANGUL_RANGE[0] <= cp <= _HANGUL_RANGE[1]


def _is_japanese_kana(ch: str) -> bool:
    cp = ord(ch)
    return (_HIRAGANA_RANGE[0] <= cp <= _HIRAGANA_RANGE[1] or
            _KATAKANA_RANGE[0] <= cp <= _KATAKANA_RANGE[1])


def word_tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    current = []
    for ch in text:
        if _is_cjk(ch) or _is_hangul(ch) or _is_japanese_kana(ch):
            if current:
                word = "".join(current)
                if word.strip():
                    tokens.append(word.strip())
                current = []
            tokens.append(ch)
        elif ch.isspace() or ch in ".,;:!?\"'()[]{}—–-…。、！？「」『』（）":
            if current:
                word = "".join(current)
                if word.strip():
                    tokens.append(word.strip())
                current = []
        else:
            current.append(ch)
    if current:
        word = "".join(current)
        if word.strip():
            tokens.append(word.strip())
    return tokens


_VOWELS_LATIN = set("aeiouyáéíóúàèìòùâêîôûäëïöüãõ")
_VOWELS_LATIN.update("AEIOUYÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÃÕ")


def syllable_estimate(word: str) -> int:
    if not word:
        return 0
    if any(_is_cjk(c) for c in word):
        return sum(1 for c in word if _is_cjk(c))
    if any(_is_hangul(c) for c in word):
        return sum(1 for c in word if _is_hangul(c))
    if any(_is_japanese_kana(c) for c in word):
        count = 0
        for c in word:
            if _is_japanese_kana(c):
                if c not in "ーっッんン":
                    count += 1
        return max(count, 1)
    w = word.lower()
    if w.endswith("le") and len(w) > 2 and w[-3] not in "aeiouy":
        w = w[:-2]
        count = 1
    else:
        count = 0
    in_vowel = False
    for ch in w:
        is_v = ch in _VOWELS_LATIN
        if is_v and not in_vowel:
            count += 1
        in_vowel = is_v
    if w.endswith("e") and count > 1:
        count -= 1
    if w.endswith("ed") and count > 1 and len(w) > 3 and w[-3] not in "td":
        count -= 1
    return max(count, 1)


_LANGUAGE_TRIGRAMS: dict[str, set[str]] = {
    "en": {"the", "ing", "tion", "and", "ent", "ion", "tio", "for", "ate"},
    "fr": {"les", "des", "ent", "que", "ion", "ait", "ous", "ant", "eur"},
    "de": {"ein", "und", "der", "die", "sch", "ich", "den", "ung", "ber"},
    "es": {"que", "ción", "los", "las", "del", "ado", "nte", "por", "con"},
    "it": {"che", "zione", "gli", "per", "ell", "ato", "con", "del", "una"},
    "pt": {"que", "ção", "dos", "das", "ado", "nte", "por", "com", "uma"},
    "nl": {"een", "het", "van", "aar", "oor", "ijk", "ing", "ver", "ter"},
    "ru": {"ный", "ого", "ать", "ить", "ени", "ест", "ова", "ста", "ост"},
    "ja": {"の", "に", "は", "を", "た", "が", "で", "て", "る"},
    "zh": {"的", "了", "在", "是", "我", "有", "不", "人", "这"},
    "ko": {"은", "는", "이", "가", "를", "에", "의", "다", "한"},
    "ar": {"في", "من", "على", "أن", "إلى", "هذا", "ال", "لا", "ما"},
    "hi": {"का", "के", "में", "है", "की", "और", "से", "को", "ने"},
    "tr": {"bir", "lar", "ler", "ile", "dan", "dır", "yor", "için", "var"},
    "sv": {"och", "att", "det", "som", "för", "den", "var", "med", "han"},
}

_SCRIPT_LANG_MAP: dict[str, str] = {
    "ARABIC": "ar",
    "DEVANAGARI": "hi",
    "HANGUL": "ko",
    "CJK": "zh",
    "HIRAGANA": "ja",
    "KATAKANA": "ja",
    "THAI": "th",
    "CYRILLIC": "ru",
    "GREEK": "el",
    "HEBREW": "he",
    "ARMENIAN": "hy",
    "GEORGIAN": "ka",
    "ETHIOPIC": "am",
    "BENGALI": "bn",
    "TAMIL": "ta",
    "TELUGU": "te",
    "GUJARATI": "gu",
    "GURMUKHI": "pa",
}


def _dominant_script(text: str) -> str | None:
    script_counts: Counter[str] = Counter()
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        name = unicodedata.name(ch, "UNKNOWN")
        for script_key in _SCRIPT_LANG_MAP:
            if script_key in name:
                script_counts[script_key] += 1
                break
    if not script_counts:
        return None
    dominant = script_counts.most_common(1)[0]
    if dominant[1] >= 3:
        return dominant[0]
    return None


def detect_language(text: str) -> str:
    script = _dominant_script(text)
    if script and script in _SCRIPT_LANG_MAP:
        return _SCRIPT_LANG_MAP[script]

    words = text.lower().split()
    if len(words) < 3:
        return "en"

    bigrams = Counter()
    trigrams = Counter()
    for w in words:
        for i in range(len(w) - 1):
            bigrams[w[i:i+2]] += 1
        for i in range(len(w) - 2):
            trigrams[w[i:i+3]] += 1

    word_set = set(words)
    best_lang = "en"
    best_score = 0
    for lang, patterns in _LANGUAGE_TRIGRAMS.items():
        score = sum(1 for p in patterns if p in word_set or p in text.lower())
        if score > best_score:
            best_score = score
            best_lang = lang
    return best_lang
