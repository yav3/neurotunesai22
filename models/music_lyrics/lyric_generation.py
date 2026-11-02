"""Lyric generation: syllable-aware line construction, rhyme constraints, verse/chorus structure."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class SyllableTemplate:
    counts: List[int]
    name: str = "custom"

COMMON_TEMPLATES: Dict[str, SyllableTemplate] = {
    "ballad": SyllableTemplate([8, 6, 8, 6], "ballad"),
    "limerick": SyllableTemplate([8, 8, 5, 5, 8], "limerick"),
    "haiku": SyllableTemplate([5, 7, 5], "haiku"),
    "sonnet": SyllableTemplate([10] * 14, "sonnet"),
    "blues": SyllableTemplate([8, 8, 8], "blues"),
    "rap_16": SyllableTemplate([12] * 16, "rap_16"),
    "pop_verse": SyllableTemplate([8, 8, 8, 8], "pop_verse"),
    "pop_chorus": SyllableTemplate([6, 6, 8, 6], "pop_chorus"),
}


@dataclass
class RhymeScheme:
    pattern: str
    name: str = ""

RHYME_SCHEMES: Dict[str, RhymeScheme] = {
    "couplet": RhymeScheme("AABB"),
    "alternate": RhymeScheme("ABAB"),
    "enclosed": RhymeScheme("ABBA"),
    "monorhyme": RhymeScheme("AAAA"),
    "limerick": RhymeScheme("AABBA"),
    "terza_rima": RhymeScheme("ABA BCB CDC"),
    "free": RhymeScheme(""),
}


@dataclass
class PhonemeCluster:
    vowel: str
    coda: str
    stress: bool = True

    @property
    def rhyme_key(self) -> str:
        return f"{self.vowel}_{self.coda}"


VOWEL_CLASSES = {
    "long_a": ["eɪ", "aɪ"],
    "long_e": ["iː", "i"],
    "long_o": ["oʊ", "aʊ"],
    "short_a": ["æ", "ɛ"],
    "short_o": ["ɒ", "ɔ"],
    "schwa": ["ə"],
}


def vowel_similarity(v1: str, v2: str) -> float:
    for cls_vowels in VOWEL_CLASSES.values():
        if v1 in cls_vowels and v2 in cls_vowels:
            return 1.0
    return 0.0


def rhyme_distance(a: PhonemeCluster, b: PhonemeCluster) -> float:
    if a.rhyme_key == b.rhyme_key:
        return 0.0
    v_sim = vowel_similarity(a.vowel, b.vowel)
    c_sim = 1.0 if a.coda == b.coda else 0.0
    return 1.0 - 0.6 * v_sim - 0.4 * c_sim


@dataclass
class GeneratedLine:
    text: str
    syllables: int
    rhyme_group: str = ""
    stress_pattern: List[int] = field(default_factory=list)


@dataclass
class GeneratedVerse:
    lines: List[GeneratedLine]
    template: str = ""
    rhyme_scheme: str = ""


@dataclass
class SongStructure:
    sections: List[Tuple[str, int]]

COMMON_STRUCTURES: Dict[str, SongStructure] = {
    "verse_chorus": SongStructure([("verse", 4), ("chorus", 4), ("verse", 4), ("chorus", 4)]),
    "aaba": SongStructure([("A", 8), ("A", 8), ("B", 8), ("A", 8)]),
    "blues_12bar": SongStructure([("I", 4), ("IV", 2), ("I", 2), ("V", 1), ("IV", 1), ("I", 2)]),
    "pop": SongStructure([("verse", 4), ("prechorus", 2), ("chorus", 4),
                           ("verse", 4), ("prechorus", 2), ("chorus", 4), ("bridge", 4), ("chorus", 4)]),
    "ballad": SongStructure([("verse", 4), ("verse", 4), ("chorus", 4), ("verse", 4), ("chorus", 4)]),
}


def count_syllables_approx(word: str) -> int:
    word = word.lower().strip(".,!?;:'\"()-")
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
    return max(1, count)


def line_syllable_count(line: str) -> int:
    return sum(count_syllables_approx(w) for w in line.split())


def stress_from_syllable_count(n: int, meter: str = "iambic") -> List[int]:
    if meter == "iambic":
        return [i % 2 for i in range(n)]
    elif meter == "trochaic":
        return [(i + 1) % 2 for i in range(n)]
    elif meter == "dactylic":
        return [1 if i % 3 == 0 else 0 for i in range(n)]
    elif meter == "anapestic":
        return [1 if i % 3 == 2 else 0 for i in range(n)]
    return [0] * n


def enforce_syllable_count(words: List[str], target: int) -> List[str]:
    result: List[str] = []
    total = 0
    for w in words:
        sc = count_syllables_approx(w)
        if total + sc > target:
            break
        result.append(w)
        total += sc
    return result


def end_phoneme(word: str) -> str:
    word = word.lower().strip(".,!?;:'\"()-")
    vowels = set("aeiouy")
    last_vowel_idx = -1
    for i in range(len(word) - 1, -1, -1):
        if word[i] in vowels:
            last_vowel_idx = i
            break
    if last_vowel_idx < 0:
        return word[-2:] if len(word) >= 2 else word
    return word[last_vowel_idx:]


def words_rhyme(a: str, b: str) -> bool:
    return end_phoneme(a) == end_phoneme(b)


def pick_rhyming_word(target_ending: str, vocabulary: List[str]) -> Optional[str]:
    candidates = [w for w in vocabulary if end_phoneme(w) == target_ending]
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


@dataclass
class MarkovLyricModel:
    order: int = 2
    transitions: Dict[tuple, Dict[str, int]] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)

    def train(self, corpus: List[str]) -> None:
        all_words: List[str] = []
        for line in corpus:
            words = line.lower().split()
            all_words.extend(words)
        self.vocabulary = list(set(all_words))
        for i in range(len(all_words) - self.order):
            ctx = tuple(all_words[i:i + self.order])
            nxt = all_words[i + self.order]
            if ctx not in self.transitions:
                self.transitions[ctx] = {}
            self.transitions[ctx][nxt] = self.transitions[ctx].get(nxt, 0) + 1

    def generate_words(self, n: int, seed: Optional[tuple] = None) -> List[str]:
        if not self.transitions:
            return []
        if seed is None or seed not in self.transitions:
            keys = list(self.transitions.keys())
            seed = keys[int(np.random.randint(len(keys)))]
        result = list(seed)
        ctx = seed
        for _ in range(n - self.order):
            if ctx not in self.transitions:
                keys = list(self.transitions.keys())
                ctx = keys[int(np.random.randint(len(keys)))]
            dist = self.transitions[ctx]
            words_list = list(dist.keys())
            counts = np.array(list(dist.values()), dtype=float)
            probs = counts / counts.sum()
            idx = np.random.choice(len(words_list), p=probs)
            w = words_list[idx]
            result.append(w)
            ctx = tuple(result[-self.order:])
        return result

    def generate_line(self, target_syllables: int) -> GeneratedLine:
        words = self.generate_words(target_syllables * 2)
        selected = enforce_syllable_count(words, target_syllables)
        text = " ".join(selected)
        sc = line_syllable_count(text)
        return GeneratedLine(text=text, syllables=sc)


def generate_verse(model: MarkovLyricModel, template: SyllableTemplate,
                   rhyme_scheme: str = "") -> GeneratedVerse:
    lines: List[GeneratedLine] = []
    rhyme_endings: Dict[str, str] = {}
    for i, syl_count in enumerate(template.counts):
        line = model.generate_line(syl_count)
        if rhyme_scheme and i < len(rhyme_scheme):
            group = rhyme_scheme[i]
            line.rhyme_group = group
            if group in rhyme_endings and line.text.split():
                target = rhyme_endings[group]
                replacement = pick_rhyming_word(target, model.vocabulary)
                if replacement:
                    words = line.text.split()
                    words[-1] = replacement
                    line.text = " ".join(words)
                    line.syllables = line_syllable_count(line.text)
            elif line.text.split():
                rhyme_endings[group] = end_phoneme(line.text.split()[-1])
        lines.append(line)
    return GeneratedVerse(lines=lines, template=template.name, rhyme_scheme=rhyme_scheme)


def song_section_count(structure_name: str) -> Dict[str, int]:
    struct = COMMON_STRUCTURES.get(structure_name)
    if struct is None:
        return {}
    counts: Dict[str, int] = {}
    for sec_name, _ in struct.sections:
        counts[sec_name] = counts.get(sec_name, 0) + 1
    return counts


def total_lines(structure_name: str) -> int:
    struct = COMMON_STRUCTURES.get(structure_name)
    if struct is None:
        return 0
    return sum(n for _, n in struct.sections)


def lyric_density(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    unique = len(set(w.lower() for w in words))
    return unique / len(words)


def repetition_ratio(lines: List[str]) -> float:
    if not lines:
        return 0.0
    normalized = [l.lower().strip() for l in lines]
    unique = len(set(normalized))
    return 1.0 - unique / len(normalized)
