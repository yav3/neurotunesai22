"""Rhyme detection: suffix matching, scheme detection, alliteration."""
from __future__ import annotations

import re

from models.music_lyrics.tokenizer import word_tokenize, normalize_text


_VOWELS = set("aeiouy")


def phonetic_suffix(word: str, min_len: int = 2) -> str:
    w = normalize_text(word, strip_diacritics=True, lowercase=True)
    w = re.sub(r"[^a-z]", "", w)
    if len(w) <= min_len:
        return w
    for i in range(max(0, len(w) - 4), len(w)):
        if w[i] in _VOWELS:
            suffix = w[i:]
            if len(suffix) >= min_len:
                return suffix
    return w[-min_len:]


def _last_word(line: str) -> str:
    tokens = word_tokenize(line.strip())
    if not tokens:
        return ""
    return tokens[-1].lower()


def _rhymes(w1: str, w2: str, threshold: int = 2) -> bool:
    if not w1 or not w2:
        return False
    s1 = phonetic_suffix(w1)
    s2 = phonetic_suffix(w2)
    if s1 == s2 and w1.lower() != w2.lower():
        return True
    common = 0
    for a, b in zip(reversed(s1), reversed(s2)):
        if a == b:
            common += 1
        else:
            break
    return common >= threshold


def detect_rhyme_scheme(text: str, max_lines: int = 32) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()][:max_lines]
    if len(lines) < 2:
        return ""

    last_words = [_last_word(l) for l in lines]
    labels: list[str] = []
    label_map: dict[str, str] = {}
    next_label = 0

    for i, w in enumerate(last_words):
        matched = False
        for j in range(i):
            if _rhymes(w, last_words[j]):
                labels.append(labels[j])
                matched = True
                break
        if not matched:
            label = chr(ord("A") + next_label % 26)
            labels.append(label)
            next_label += 1

    return "".join(labels)


def end_rhyme_score(text: str) -> float:
    scheme = detect_rhyme_scheme(text)
    if len(scheme) < 2:
        return 0.0
    unique = len(set(scheme))
    total = len(scheme)
    if unique >= total:
        return 0.0
    return 1.0 - (unique / total)


def alliteration_score(text: str) -> float:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0

    scores = []
    for line in lines:
        words = [w.lower() for w in word_tokenize(line) if len(w) > 1]
        if len(words) < 2:
            scores.append(0.0)
            continue
        initials = [w[0] for w in words if w[0].isalpha()]
        if len(initials) < 2:
            scores.append(0.0)
            continue
        from collections import Counter
        counts = Counter(initials)
        max_count = counts.most_common(1)[0][1]
        scores.append((max_count - 1) / (len(initials) - 1) if len(initials) > 1 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0
