from __future__ import annotations

import re
from dataclasses import dataclass, field

from models.music_nlp.multilingual import MultilingualProcessor


STRESS_DICTIONARY: dict[str, list[int]] = {
    "a": [0], "the": [0], "an": [0], "and": [0], "but": [0], "or": [0],
    "of": [0], "in": [0], "on": [0], "at": [0], "to": [0], "for": [0],
    "is": [0], "am": [0], "are": [0], "was": [0], "were": [0],
    "be": [1], "been": [1], "do": [1], "did": [1], "has": [1], "had": [1],
    "about": [0, 1], "above": [0, 1], "across": [0, 1], "after": [1, 0],
    "again": [0, 1], "against": [0, 1], "ago": [0, 1], "ahead": [0, 1],
    "along": [0, 1], "also": [1, 0], "always": [1, 0], "among": [0, 1],
    "another": [0, 1, 0], "answer": [1, 0], "any": [1, 0], "apple": [1, 0],
    "away": [0, 1], "baby": [1, 0], "battle": [1, 0], "beauty": [1, 0],
    "because": [0, 1], "become": [0, 1], "before": [0, 1], "begin": [0, 1],
    "behind": [0, 1], "believe": [0, 1], "below": [0, 1], "beneath": [0, 1],
    "beside": [0, 1], "between": [0, 1], "beyond": [0, 1], "bitter": [1, 0],
    "body": [1, 0], "broken": [1, 0], "brother": [1, 0],
    "candle": [1, 0], "captain": [1, 0], "carry": [1, 0], "castle": [1, 0],
    "center": [1, 0], "certain": [1, 0], "children": [1, 0],
    "city": [1, 0], "color": [1, 0], "coming": [1, 0], "common": [1, 0],
    "complete": [0, 1], "concern": [0, 1], "contain": [0, 1],
    "country": [1, 0], "create": [0, 1], "crimson": [1, 0],
    "danger": [1, 0], "darkness": [1, 0], "daughter": [1, 0],
    "deliver": [0, 1, 0], "demand": [0, 1], "depend": [0, 1],
    "desire": [0, 1], "destroy": [0, 1], "divine": [0, 1],
    "doctor": [1, 0], "early": [1, 0], "easy": [1, 0], "empty": [1, 0],
    "enough": [0, 1], "enter": [1, 0], "even": [1, 0], "evening": [1, 0, 0],
    "ever": [1, 0], "every": [1, 0, 0], "evil": [1, 0], "expect": [0, 1],
    "explain": [0, 1], "extreme": [0, 1],
    "fallen": [1, 0], "family": [1, 0, 0], "father": [1, 0],
    "finger": [1, 0], "flower": [1, 0], "follow": [1, 0],
    "forget": [0, 1], "forever": [0, 1, 0], "forward": [1, 0],
    "freedom": [1, 0], "frozen": [1, 0], "future": [1, 0],
    "garden": [1, 0], "gather": [1, 0], "gentle": [1, 0], "glory": [1, 0],
    "golden": [1, 0], "govern": [1, 0],
    "happy": [1, 0], "heaven": [1, 0], "heavy": [1, 0], "hidden": [1, 0],
    "hollow": [1, 0], "holy": [1, 0], "honor": [1, 0], "human": [1, 0],
    "hunger": [1, 0], "hundred": [1, 0],
    "imagine": [0, 1, 0], "indeed": [0, 1], "inside": [0, 1],
    "instead": [0, 1], "into": [1, 0],
    "journey": [1, 0], "judgment": [1, 0], "justice": [1, 0],
    "kingdom": [1, 0], "knowledge": [1, 0],
    "laughter": [1, 0], "letter": [1, 0], "listen": [1, 0],
    "little": [1, 0], "living": [1, 0], "lonely": [1, 0], "lover": [1, 0],
    "master": [1, 0], "matter": [1, 0], "middle": [1, 0],
    "midnight": [1, 0], "mirror": [1, 0], "moment": [1, 0],
    "monster": [1, 0], "morning": [1, 0], "mother": [1, 0],
    "mountain": [1, 0], "murder": [1, 0], "music": [1, 0],
    "mystery": [1, 0, 0],
    "nature": [1, 0], "never": [1, 0], "nothing": [1, 0], "number": [1, 0],
    "ocean": [1, 0], "only": [1, 0], "open": [1, 0], "order": [1, 0],
    "other": [1, 0], "over": [1, 0],
    "palace": [1, 0], "paper": [1, 0], "passion": [1, 0],
    "patient": [1, 0], "people": [1, 0], "perfect": [1, 0],
    "perhaps": [0, 1], "picture": [1, 0], "pleasure": [1, 0],
    "power": [1, 0], "present": [1, 0], "prison": [1, 0],
    "promise": [1, 0], "protect": [0, 1], "provide": [0, 1],
    "quiet": [1, 0],
    "reason": [1, 0], "receive": [0, 1], "remain": [0, 1],
    "remember": [0, 1, 0], "remove": [0, 1], "repeat": [0, 1],
    "return": [0, 1], "river": [1, 0], "royal": [1, 0], "ruler": [1, 0],
    "running": [1, 0],
    "sacred": [1, 0], "safety": [1, 0], "season": [1, 0],
    "secret": [1, 0], "shadow": [1, 0], "silver": [1, 0],
    "simple": [1, 0], "sister": [1, 0], "slowly": [1, 0],
    "soldier": [1, 0], "something": [1, 0], "sorrow": [1, 0],
    "spirit": [1, 0], "spoken": [1, 0], "standing": [1, 0],
    "story": [1, 0], "stranger": [1, 0], "suffer": [1, 0],
    "summer": [1, 0], "sunrise": [1, 0], "surprise": [0, 1],
    "surrender": [0, 1, 0], "survive": [0, 1],
    "teacher": [1, 0], "temple": [1, 0], "tender": [1, 0],
    "thousand": [1, 0], "tiger": [1, 0], "today": [0, 1],
    "together": [0, 1, 0], "tomorrow": [0, 1, 0], "trouble": [1, 0],
    "truly": [1, 0], "under": [1, 0], "until": [0, 1], "upon": [0, 1],
    "very": [1, 0], "village": [1, 0], "vision": [1, 0], "voices": [1, 0],
    "walking": [1, 0], "water": [1, 0], "weather": [1, 0],
    "whisper": [1, 0], "wicked": [1, 0], "window": [1, 0],
    "winter": [1, 0], "wisdom": [1, 0], "within": [0, 1],
    "without": [0, 1], "woman": [1, 0], "wonder": [1, 0],
    "yellow": [1, 0], "yesterday": [1, 0, 0],
}

_UNSTRESSED_PREFIXES = {"a", "be", "de", "dis", "en", "em", "ex", "in",
                         "mis", "non", "pre", "re", "un"}
_STRESSED_SUFFIXES = {"ade", "aine", "aise", "ance", "ence", "ense",
                       "esque", "ette", "ique", "ise", "ize", "oon",
                       "ude", "ure"}

VOWELS = set("aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿ")


@dataclass
class ProsodyFeatures:
    meter_type: str = "free"
    meter_regularity: float = 0.0
    stress_pattern: list[int] = field(default_factory=list)
    feet_per_line_avg: float = 0.0
    caesura_positions: list[int] = field(default_factory=list)
    enjambment_count: int = 0
    alliteration_score: float = 0.0
    assonance_score: float = 0.0
    consonance_score: float = 0.0


class ProsodyAnalyzer:

    def __init__(self) -> None:
        self._mlp = MultilingualProcessor()

    def analyze(self, text: str) -> ProsodyFeatures:
        if not text or not text.strip():
            return ProsodyFeatures()

        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            return ProsodyFeatures()

        line_stresses: list[list[int]] = []
        all_stress: list[int] = []
        for ln in lines:
            words = re.findall(r"[a-zA-ZÀ-ÿ'']+", ln)
            stress_seq: list[int] = []
            for w in words:
                stress_seq.extend(self._estimate_stress(w))
            line_stresses.append(stress_seq)
            all_stress.extend(stress_seq)

        meter_type, regularity = self._detect_meter(line_stresses)

        feet_counts = []
        for ls in line_stresses:
            stressed = sum(ls)
            feet_counts.append(max(stressed, 1))
        feet_avg = sum(feet_counts) / max(len(feet_counts), 1)

        caesurae = self._find_caesurae(lines)
        enjamb = self._count_enjambment(lines)
        allit = self._alliteration_score(lines)
        asson = self._assonance_score(text)
        conson = self._consonance_score(text)

        return ProsodyFeatures(
            meter_type=meter_type,
            meter_regularity=round(regularity, 4),
            stress_pattern=all_stress,
            feet_per_line_avg=round(feet_avg, 2),
            caesura_positions=caesurae,
            enjambment_count=enjamb,
            alliteration_score=round(allit, 4),
            assonance_score=round(asson, 4),
            consonance_score=round(conson, 4),
        )

    def _estimate_stress(self, word: str) -> list[int]:
        w = word.lower().strip("''")
        if not w:
            return []
        if w in STRESS_DICTIONARY:
            return list(STRESS_DICTIONARY[w])
        return self._heuristic_stress(w)

    def _heuristic_stress(self, word: str) -> list[int]:
        syllables = self._split_syllables(word)
        n = len(syllables)
        if n == 0:
            return []
        if n == 1:
            if word in {"a", "an", "the", "and", "or", "but", "of", "in",
                        "on", "at", "to", "for", "by", "up", "so", "no",
                        "if", "as", "it", "my", "me", "he", "we"}:
                return [0]
            return [1]

        pattern = [0] * n

        for prefix in _UNSTRESSED_PREFIXES:
            if word.startswith(prefix) and len(word) > len(prefix) + 1:
                pattern[0] = 0
                if n >= 2:
                    pattern[1] = 1
                return pattern

        for suffix in _STRESSED_SUFFIXES:
            if word.endswith(suffix):
                pattern[-1] = 1
                return pattern

        if word.endswith("ing") or word.endswith("tion") or word.endswith("ness"):
            if n >= 2:
                pattern[-2] = 1
                return pattern

        pattern[0] = 1
        return pattern

    @staticmethod
    def _split_syllables(word: str) -> list[str]:
        word = word.lower()
        syllables: list[str] = []
        current = ""
        prev_vowel = False
        for ch in word:
            is_v = ch in VOWELS
            if is_v:
                if prev_vowel:
                    current += ch
                else:
                    if current and not prev_vowel:
                        current += ch
                    else:
                        if current:
                            syllables.append(current)
                        current = ch
                prev_vowel = True
            else:
                current += ch
                if prev_vowel:
                    prev_vowel = False
                    pass
                prev_vowel = False
        if current:
            if syllables and not any(c in VOWELS for c in current):
                syllables[-1] += current
            else:
                syllables.append(current)
        if not syllables:
            syllables = [word] if word else []
        return syllables

    def _detect_meter(
        self, line_stresses: list[list[int]],
    ) -> tuple[str, float]:
        if not line_stresses:
            return ("free", 0.0)

        patterns = {
            "iambic": [0, 1],
            "trochaic": [1, 0],
            "anapestic": [0, 0, 1],
            "dactylic": [1, 0, 0],
            "spondaic": [1, 1],
        }

        best_name = "free"
        best_reg = 0.0

        for name, foot in patterns.items():
            foot_len = len(foot)
            total_matches = 0
            total_positions = 0
            for stress in line_stresses:
                if len(stress) < 2:
                    continue
                for i in range(len(stress)):
                    foot_pos = i % foot_len
                    if stress[i] == foot[foot_pos]:
                        total_matches += 1
                    total_positions += 1
            if total_positions == 0:
                continue
            reg = total_matches / total_positions
            if reg > best_reg:
                best_reg = reg
                best_name = name

        if best_reg < 0.55:
            return ("free", best_reg)
        return (best_name, best_reg)

    def _alliteration_score(self, lines: list[str]) -> float:
        if not lines:
            return 0.0
        count = 0
        for ln in lines:
            words = re.findall(r"[a-zA-ZÀ-ÿ]+", ln)
            if len(words) < 2:
                continue
            initials = [w[0].lower() for w in words if w]
            consonant_initials = [c for c in initials if c not in VOWELS]
            if len(consonant_initials) < 2:
                continue
            from collections import Counter
            freq = Counter(consonant_initials)
            max_freq = max(freq.values())
            if max_freq >= 2 and max_freq / len(consonant_initials) >= 0.4:
                count += 1
        return count / max(len(lines), 1)

    def _assonance_score(self, text: str) -> float:
        vowel_seq = re.findall(r"[aeiouy]+", text.lower())
        if len(vowel_seq) < 2:
            return 0.0
        from collections import Counter
        freq = Counter(vowel_seq)
        repeated = sum(c for c in freq.values() if c > 1)
        return repeated / max(len(vowel_seq), 1)

    def _consonance_score(self, text: str) -> float:
        consonant_clusters = re.findall(r"[bcdfghjklmnpqrstvwxyz]{2,}", text.lower())
        if len(consonant_clusters) < 2:
            return 0.0
        from collections import Counter
        freq = Counter(consonant_clusters)
        repeated = sum(c for c in freq.values() if c > 1)
        return repeated / max(len(consonant_clusters), 1)

    @staticmethod
    def _find_caesurae(lines: list[str]) -> list[int]:
        positions: list[int] = []
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            mid = len(stripped) // 2
            window = stripped[max(mid - 3, 0):min(mid + 4, len(stripped))]
            if re.search(r"[,;:—–\-]|\s{2,}", window):
                positions.append(i)
        return positions

    @staticmethod
    def _count_enjambment(lines: list[str]) -> int:
        count = 0
        for i in range(len(lines) - 1):
            stripped = lines[i].rstrip()
            if not stripped:
                continue
            last_char = stripped[-1]
            if last_char not in ".!?,;:—–\"')]}":
                count += 1
        return count
