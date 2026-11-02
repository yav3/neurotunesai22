"""Lyrics analysis: syllable density, rhyme detection, repetition, sentiment, alignment."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


# ---------------------------------------------------------------------------
# Syllable estimation (English heuristic)
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouyAEIOUY")


def _count_syllables(word: str) -> int:
    """Estimate syllable count for an English word."""
    word = word.strip().lower()
    if not word:
        return 0
    # remove trailing silent-e
    if len(word) > 2 and word.endswith("e") and word[-2] not in "aeiouy":
        word = word[:-1]
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)


def syllable_count(text: str) -> int:
    """Total syllables in a line of text."""
    words = re.findall(r"[a-zA-Z']+", text)
    return sum(_count_syllables(w) for w in words)


def syllable_density(lines: list[str], beats_per_line: float | list[float]) -> np.ndarray:
    """Syllables per beat for each line.

    Parameters
    ----------
    lines : list[str]
        Lyric lines.
    beats_per_line : float or list[float]
        Beats per line (scalar broadcasts).

    Returns
    -------
    np.ndarray
        Syllable density per line.
    """
    if isinstance(beats_per_line, (int, float)):
        bpl = np.full(len(lines), float(beats_per_line))
    else:
        bpl = np.asarray(beats_per_line, dtype=float)
    densities = np.array([syllable_count(l) for l in lines], dtype=float)
    return densities / np.maximum(bpl, 1e-12)


# ---------------------------------------------------------------------------
# Rhyme detection
# ---------------------------------------------------------------------------

def _last_word(line: str) -> str:
    words = re.findall(r"[a-zA-Z']+", line)
    return words[-1].lower() if words else ""


def _ending(word: str, n: int = 3) -> str:
    return word[-n:] if len(word) >= n else word


def rhyme_similarity(a: str, b: str) -> float:
    """Simple rhyme score: fraction of shared trailing characters (max 4)."""
    wa, wb = _last_word(a), _last_word(b)
    if not wa or not wb:
        return 0.0
    max_n = min(len(wa), len(wb), 4)
    shared = 0
    for i in range(1, max_n + 1):
        if wa[-i] == wb[-i]:
            shared += 1
        else:
            break
    return shared / max_n


def detect_rhyme_scheme(lines: list[str], threshold: float = 0.5) -> list[str]:
    """Assign rhyme-scheme labels (A, B, C, ...) to lines.

    Lines whose last words share enough trailing characters get the same label.
    """
    if not lines:
        return []
    labels: list[str] = []
    groups: list[str] = []  # representative last-word per label
    current_label = ord("A")
    for line in lines:
        lw = _last_word(line)
        matched = False
        for i, rep in enumerate(groups):
            if rhyme_similarity(line, rep + " " + rep) >= threshold:
                # compare actual ending
                wa, wb = lw, _last_word(rep + " " + rep)
                max_n = min(len(wa), len(wb), 4)
                shared = 0
                for k in range(1, max_n + 1):
                    if wa[-k] == wb[-k]:
                        shared += 1
                    else:
                        break
                if shared / max(max_n, 1) >= threshold:
                    labels.append(chr(ord("A") + i))
                    matched = True
                    break
        if not matched:
            labels.append(chr(current_label))
            groups.append(line)
            current_label += 1
    return labels


# ---------------------------------------------------------------------------
# Repetition / chorus-verse structure
# ---------------------------------------------------------------------------

def _line_tokens(line: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", line)]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def line_similarity_matrix(lines: list[str]) -> np.ndarray:
    """Jaccard similarity between all line pairs."""
    n = len(lines)
    tokens = [_line_tokens(l) for l in lines]
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = _jaccard(tokens[i], tokens[j])
    return mat


@dataclass
class RepetitionStructure:
    """Detected repetition structure."""
    section_labels: list[str]   # e.g. ["verse", "verse", "chorus", "verse", "chorus"]
    repetition_score: float     # 0–1, fraction of lines that repeat


def detect_repetition(lines: list[str], sim_threshold: float = 0.6) -> RepetitionStructure:
    """Detect chorus/verse structure from text similarity."""
    if not lines:
        return RepetitionStructure([], 0.0)
    sim = line_similarity_matrix(lines)
    n = len(lines)
    # count how many OTHER lines each line is similar to
    repeat_counts = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j and sim[i, j] >= sim_threshold:
                repeat_counts[i] += 1

    labels: list[str] = []
    for i in range(n):
        labels.append("chorus" if repeat_counts[i] >= 2 else "verse")

    rep_score = float(np.mean(repeat_counts > 0))
    return RepetitionStructure(labels, rep_score)


# ---------------------------------------------------------------------------
# Sentiment (mini lexicon, music-relevant)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "love", "happy", "joy", "dance", "sing", "free", "beautiful", "shine",
    "bright", "dream", "sweet", "smile", "laugh", "warm", "hope", "peace",
    "light", "alive", "fun", "wonderful", "magic", "paradise", "heaven",
    "sunshine", "golden", "baby", "honey", "angel", "celebrate", "together",
    "fly", "rise", "good", "best", "great", "amazing", "bliss", "glory",
    "tender", "kind", "gentle", "delight", "passion", "heart", "soul",
    "groove", "rhythm", "harmony", "melody", "music",
}

_NEGATIVE_WORDS = {
    "sad", "cry", "pain", "hurt", "dark", "alone", "lost", "broken",
    "fear", "hate", "die", "death", "tears", "cold", "empty", "gone",
    "shadow", "fall", "burn", "scream", "bleed", "wound", "sorrow",
    "grief", "hell", "nightmare", "storm", "rage", "anger", "bitter",
    "lonely", "miss", "regret", "sin", "suffer", "war", "kill", "ghost",
    "devil", "grave", "doom", "fade", "drown", "chains", "prison",
    "trapped", "shatter", "wither", "ruin", "despair",
}


@dataclass
class SentimentResult:
    polarity: float       # -1 to +1
    positive_count: int
    negative_count: int
    word_count: int


def sentiment_polarity(text: str) -> SentimentResult:
    """Lexicon-based sentiment polarity for lyrics."""
    words = [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    total = pos + neg
    polarity = (pos - neg) / total if total > 0 else 0.0
    return SentimentResult(polarity=polarity, positive_count=pos,
                           negative_count=neg, word_count=len(words))


def sentiment_trajectory(lines: list[str]) -> np.ndarray:
    """Per-line sentiment polarity trajectory."""
    return np.array([sentiment_polarity(l).polarity for l in lines], dtype=float)


# ---------------------------------------------------------------------------
# Lyric–melody alignment
# ---------------------------------------------------------------------------

def lyric_melody_alignment(
    syllables_per_phrase: list[int],
    notes_per_phrase: list[int],
) -> float:
    """Score how well syllable counts match note counts (0–1, 1 = perfect).

    Uses normalised absolute deviation: 1 - mean(|s - n| / max(s, n)).
    """
    if not syllables_per_phrase or not notes_per_phrase:
        return 0.0
    n = min(len(syllables_per_phrase), len(notes_per_phrase))
    s = np.array(syllables_per_phrase[:n], dtype=float)
    m = np.array(notes_per_phrase[:n], dtype=float)
    denom = np.maximum(s, m)
    denom = np.where(denom == 0, 1.0, denom)
    deviations = np.abs(s - m) / denom
    return float(1.0 - np.mean(deviations))
