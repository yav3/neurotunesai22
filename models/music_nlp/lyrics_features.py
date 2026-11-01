from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from models.music_nlp.multilingual import MultilingualProcessor


POSITIVE_WORDS: frozenset[str] = frozenset({
    "love", "happy", "joy", "beautiful", "wonderful", "amazing", "great",
    "good", "best", "sunshine", "smile", "laugh", "dance", "celebrate",
    "dream", "hope", "light", "bright", "warm", "peace", "free", "freedom",
    "paradise", "heaven", "angel", "star", "glory", "sweet", "gentle",
    "kind", "grace", "bliss", "delight", "charm", "radiant", "magic",
    "alive", "triumph", "victory", "rise", "soar", "glow", "bloom",
    "treasure", "precious", "golden", "divine", "harmony", "melody",
    "sing", "together", "forever", "embrace", "kiss", "heart", "soul",
    "spirit", "faith", "trust", "believe", "inspire", "passion", "desire",
    "ecstasy", "euphoria", "elation", "jubilant", "merry", "cheerful",
    "glad", "pleased", "content", "satisfied", "grateful", "thankful",
    "blessed", "serene", "calm", "tender", "adore", "cherish", "devotion",
    "loyalty", "courage", "brave", "bold", "strong", "power", "energy",
    "vibrant", "fantastic", "marvelous", "splendid", "superb", "elegant",
    "lovely", "perfect", "pure", "innocent", "fresh", "new", "dawn",
    "sunrise", "spring", "flower", "garden", "ocean", "sky", "flying",
    "wings", "rainbow", "miracle", "wonder", "enchant", "sparkle", "shine",
})

NEGATIVE_WORDS: frozenset[str] = frozenset({
    "hate", "sad", "cry", "pain", "hurt", "break", "broken", "lost",
    "alone", "lonely", "dark", "cold", "fear", "afraid", "death", "die",
    "kill", "blood", "war", "fight", "anger", "angry", "rage", "fury",
    "storm", "shadow", "ghost", "hell", "devil", "demon", "evil", "sin",
    "guilt", "shame", "regret", "sorrow", "grief", "mourn", "weep",
    "tear", "tears", "suffer", "agony", "torture", "torment", "despair",
    "hopeless", "helpless", "worthless", "empty", "void", "nothing",
    "never", "gone", "goodbye", "farewell", "end", "fall", "fallen",
    "crash", "burn", "fire", "destroy", "ruin", "waste", "decay",
    "rot", "poison", "venom", "toxic", "sick", "disease", "wound",
    "scar", "bleed", "drown", "choke", "scream", "shout", "curse",
    "damn", "cruel", "harsh", "bitter", "vile", "wretched", "misery",
    "miserable", "tragic", "doom", "gloom", "dread", "horror", "terror",
    "panic", "anxiety", "stress", "tension", "burden", "chains", "prison",
    "trapped", "cage", "slave", "betray", "deceive", "lie", "lies",
    "cheat", "steal", "corrupt", "twisted", "shattered", "wreck",
    "collapse", "fail", "failure", "defeat", "loss", "abandon",
    "forsake", "neglect", "reject", "deny", "refuse", "cold", "freeze",
})


@dataclass
class LyricsFeatures:
    word_count: int = 0
    unique_word_count: int = 0
    lexical_diversity: float = 0.0
    line_count: int = 0
    avg_words_per_line: float = 0.0
    syllable_count_approx: int = 0
    rhyme_density: float = 0.0
    repetition_score: float = 0.0
    question_density: float = 0.0
    exclamation_density: float = 0.0
    sentiment_valence_approx: float = 0.5
    language_detected: str = "en"
    vocabulary_richness: float = 0.0
    rhyme_scheme: str = ""


class LyricsAnalyzer:

    def __init__(self) -> None:
        self._mlp = MultilingualProcessor()

    def analyze(self, text: str) -> LyricsFeatures:
        if not text or not text.strip():
            return LyricsFeatures()

        lang, _ = self._mlp.detect_language(text)
        tokens = self._mlp.tokenize(text, lang)
        lower_tokens = [t.lower() for t in tokens]
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]

        word_count = len(tokens)
        unique_words = set(lower_tokens)
        unique_count = len(unique_words)
        lexical_div = unique_count / max(word_count, 1)

        syllable_total = sum(self._count_syllables(w) for w in tokens)

        hapax = sum(1 for w in unique_words if lower_tokens.count(w) == 1)
        vocab_richness = hapax / max(unique_count, 1)

        rhyme_den = self._compute_rhyme_density(lines)
        rep_score = self._repetition_score(lines)

        sentences = re.split(r"[.!?\n]+", text)
        sentence_count = max(len([s for s in sentences if s.strip()]), 1)
        q_count = text.count("?")
        e_count = text.count("!")
        q_density = q_count / sentence_count
        e_density = e_count / sentence_count

        pos_count = sum(1 for w in lower_tokens if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in lower_tokens if w in NEGATIVE_WORDS)
        emo_total = pos_count + neg_count
        sentiment = (pos_count / emo_total) if emo_total > 0 else 0.5

        rhyme_sch = self._rhyme_scheme(lines)

        return LyricsFeatures(
            word_count=word_count,
            unique_word_count=unique_count,
            lexical_diversity=round(lexical_div, 4),
            line_count=len(lines),
            avg_words_per_line=round(word_count / max(len(lines), 1), 2),
            syllable_count_approx=syllable_total,
            rhyme_density=round(rhyme_den, 4),
            repetition_score=round(rep_score, 4),
            question_density=round(q_density, 4),
            exclamation_density=round(e_density, 4),
            sentiment_valence_approx=round(sentiment, 4),
            language_detected=lang,
            vocabulary_richness=round(vocab_richness, 4),
            rhyme_scheme=rhyme_sch,
        )

    def _count_syllables(self, word: str) -> int:
        return self._mlp.syllable_count(word, "en")

    def _detect_rhyme(self, line1: str, line2: str) -> bool:
        w1 = self._last_word(line1)
        w2 = self._last_word(line2)
        if not w1 or not w2:
            return False
        if w1 == w2:
            return False
        s1 = w1[-3:] if len(w1) >= 3 else w1[-2:] if len(w1) >= 2 else w1
        s2 = w2[-3:] if len(w2) >= 3 else w2[-2:] if len(w2) >= 2 else w2
        if s1 == s2:
            return True
        if len(w1) >= 2 and len(w2) >= 2 and w1[-2:] == w2[-2:]:
            return True
        return False

    def _rhyme_scheme(self, lines: list[str]) -> str:
        if not lines:
            return ""
        last_words = [self._last_word(ln) for ln in lines]
        scheme: list[str] = []
        label_map: dict[str, str] = {}
        next_label = 0
        for i, w in enumerate(last_words):
            if not w:
                scheme.append("X")
                continue
            found = False
            for j in range(i):
                prev_w = last_words[j]
                if prev_w and self._detect_rhyme(lines[i], lines[j]):
                    scheme.append(scheme[j])
                    found = True
                    break
            if not found:
                lbl = chr(ord("A") + next_label % 26)
                scheme.append(lbl)
                next_label += 1
        return "".join(scheme)

    def _repetition_score(self, lines: list[str]) -> float:
        if len(lines) <= 1:
            return 0.0
        normalised = [ln.strip().lower() for ln in lines]
        unique = set(normalised)
        repeated = len(normalised) - len(unique)
        return repeated / len(normalised)

    def _compute_rhyme_density(self, lines: list[str]) -> float:
        if len(lines) < 2:
            return 0.0
        pairs = 0
        rhymes = 0
        for i in range(len(lines) - 1):
            if self._detect_rhyme(lines[i], lines[i + 1]):
                rhymes += 1
            pairs += 1
        for i in range(0, len(lines) - 2, 2):
            if i + 2 < len(lines):
                if self._detect_rhyme(lines[i], lines[i + 2]):
                    rhymes += 1
                pairs += 1
        return rhymes / max(pairs, 1)

    @staticmethod
    def _last_word(line: str) -> str:
        words = re.findall(r"[a-zA-ZÀ-ÿ]+", line)
        return words[-1].lower() if words else ""
