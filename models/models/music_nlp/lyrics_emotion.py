from __future__ import annotations

import re
from dataclasses import dataclass, field

from models.music_nlp.multilingual import MultilingualProcessor


PLUTCHIK_EMOTIONS = (
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
)

EMOTION_LEXICON: dict[str, tuple[str, float]] = {
    # JOY
    "love": ("joy", 0.90), "happy": ("joy", 0.90), "joy": ("joy", 0.95),
    "bliss": ("joy", 0.90), "delight": ("joy", 0.85), "ecstasy": ("joy", 0.95),
    "elation": ("joy", 0.90), "euphoria": ("joy", 0.95), "glad": ("joy", 0.75),
    "cheerful": ("joy", 0.80), "merry": ("joy", 0.80), "jubilant": ("joy", 0.85),
    "pleased": ("joy", 0.70), "content": ("joy", 0.65), "satisfied": ("joy", 0.65),
    "grateful": ("joy", 0.75), "thankful": ("joy", 0.75), "blessed": ("joy", 0.80),
    "wonderful": ("joy", 0.85), "amazing": ("joy", 0.80), "beautiful": ("joy", 0.75),
    "smile": ("joy", 0.75), "laugh": ("joy", 0.80), "celebrate": ("joy", 0.85),
    "paradise": ("joy", 0.90), "heaven": ("joy", 0.85), "sunshine": ("joy", 0.80),
    "dance": ("joy", 0.75), "sing": ("joy", 0.70), "play": ("joy", 0.65),
    "treasure": ("joy", 0.75), "precious": ("joy", 0.70), "golden": ("joy", 0.65),
    "divine": ("joy", 0.80), "glory": ("joy", 0.80), "triumph": ("joy", 0.85),
    "victory": ("joy", 0.85), "magic": ("joy", 0.75), "dream": ("joy", 0.70),
    "adore": ("joy", 0.85), "cherish": ("joy", 0.80), "embrace": ("joy", 0.75),
    "kiss": ("joy", 0.75), "caress": ("joy", 0.70),
    # TRUST
    "trust": ("trust", 0.90), "faith": ("trust", 0.85), "believe": ("trust", 0.80),
    "loyal": ("trust", 0.85), "devotion": ("trust", 0.85), "honest": ("trust", 0.80),
    "sincere": ("trust", 0.80), "true": ("trust", 0.75), "reliable": ("trust", 0.80),
    "depend": ("trust", 0.70), "safe": ("trust", 0.75), "secure": ("trust", 0.75),
    "protect": ("trust", 0.75), "comfort": ("trust", 0.70), "promise": ("trust", 0.80),
    "gentle": ("trust", 0.70), "kind": ("trust", 0.75), "tender": ("trust", 0.70),
    "grace": ("trust", 0.75), "mercy": ("trust", 0.70), "forgive": ("trust", 0.75),
    "patience": ("trust", 0.70), "calm": ("trust", 0.65), "peace": ("trust", 0.80),
    "steady": ("trust", 0.65), "certain": ("trust", 0.65), "sure": ("trust", 0.60),
    "courage": ("trust", 0.75), "brave": ("trust", 0.70), "strength": ("trust", 0.70),
    "bond": ("trust", 0.75), "together": ("trust", 0.70), "forever": ("trust", 0.75),
    "companion": ("trust", 0.70), "friend": ("trust", 0.75), "ally": ("trust", 0.70),
    # FEAR
    "fear": ("fear", 0.90), "afraid": ("fear", 0.85), "terror": ("fear", 0.95),
    "horror": ("fear", 0.90), "panic": ("fear", 0.90), "dread": ("fear", 0.85),
    "fright": ("fear", 0.85), "scare": ("fear", 0.80), "scary": ("fear", 0.75),
    "creep": ("fear", 0.70), "shiver": ("fear", 0.70), "tremble": ("fear", 0.75),
    "scream": ("fear", 0.80), "nightmare": ("fear", 0.85), "ghost": ("fear", 0.70),
    "haunt": ("fear", 0.75), "shadow": ("fear", 0.65), "dark": ("fear", 0.60),
    "danger": ("fear", 0.80), "threat": ("fear", 0.80), "risk": ("fear", 0.65),
    "anxiety": ("fear", 0.80), "worry": ("fear", 0.70), "nervous": ("fear", 0.70),
    "tense": ("fear", 0.65), "uneasy": ("fear", 0.65), "doom": ("fear", 0.85),
    "helpless": ("fear", 0.80), "vulnerable": ("fear", 0.70), "flee": ("fear", 0.75),
    "escape": ("fear", 0.70), "hide": ("fear", 0.65), "cower": ("fear", 0.75),
    "phobia": ("fear", 0.85), "abyss": ("fear", 0.75), "void": ("fear", 0.70),
    # SURPRISE
    "surprise": ("surprise", 0.85), "shock": ("surprise", 0.85),
    "astonish": ("surprise", 0.85), "amaze": ("surprise", 0.80),
    "wonder": ("surprise", 0.75), "awe": ("surprise", 0.80),
    "startle": ("surprise", 0.80), "stun": ("surprise", 0.80),
    "gasp": ("surprise", 0.75), "sudden": ("surprise", 0.65),
    "unexpected": ("surprise", 0.75), "miracle": ("surprise", 0.80),
    "incredible": ("surprise", 0.70), "unbelievable": ("surprise", 0.75),
    "extraordinary": ("surprise", 0.70), "remarkable": ("surprise", 0.65),
    "strange": ("surprise", 0.60), "mysterious": ("surprise", 0.65),
    "curious": ("surprise", 0.60), "bewitch": ("surprise", 0.65),
    "enchant": ("surprise", 0.65), "bewilder": ("surprise", 0.70),
    "mystify": ("surprise", 0.65), "marvel": ("surprise", 0.70),
    # SADNESS
    "sad": ("sadness", 0.85), "sorrow": ("sadness", 0.90), "grief": ("sadness", 0.90),
    "cry": ("sadness", 0.80), "weep": ("sadness", 0.85), "tear": ("sadness", 0.75),
    "tears": ("sadness", 0.75), "mourn": ("sadness", 0.85), "lament": ("sadness", 0.85),
    "suffer": ("sadness", 0.80), "pain": ("sadness", 0.80), "hurt": ("sadness", 0.75),
    "agony": ("sadness", 0.90), "anguish": ("sadness", 0.90), "despair": ("sadness", 0.90),
    "hopeless": ("sadness", 0.85), "misery": ("sadness", 0.90),
    "miserable": ("sadness", 0.85), "tragic": ("sadness", 0.85),
    "lonely": ("sadness", 0.80), "alone": ("sadness", 0.75), "lost": ("sadness", 0.70),
    "broken": ("sadness", 0.80), "shattered": ("sadness", 0.85), "empty": ("sadness", 0.75),
    "hollow": ("sadness", 0.70), "wretched": ("sadness", 0.85), "gloom": ("sadness", 0.75),
    "melancholy": ("sadness", 0.85), "somber": ("sadness", 0.70),
    "regret": ("sadness", 0.75), "sorry": ("sadness", 0.65),
    "goodbye": ("sadness", 0.70), "farewell": ("sadness", 0.75),
    "death": ("sadness", 0.85), "die": ("sadness", 0.80), "grave": ("sadness", 0.75),
    "tomb": ("sadness", 0.75), "funeral": ("sadness", 0.80), "widow": ("sadness", 0.70),
    # DISGUST
    "disgust": ("disgust", 0.90), "repulse": ("disgust", 0.85),
    "revolting": ("disgust", 0.85), "vile": ("disgust", 0.85),
    "loathe": ("disgust", 0.85), "abhor": ("disgust", 0.85),
    "nausea": ("disgust", 0.80), "sick": ("disgust", 0.65),
    "filth": ("disgust", 0.80), "dirty": ("disgust", 0.65),
    "corrupt": ("disgust", 0.75), "rotten": ("disgust", 0.75),
    "decay": ("disgust", 0.70), "stench": ("disgust", 0.75),
    "poison": ("disgust", 0.75), "toxic": ("disgust", 0.75),
    "ugly": ("disgust", 0.65), "grotesque": ("disgust", 0.80),
    "foul": ("disgust", 0.80), "nasty": ("disgust", 0.70),
    "contempt": ("disgust", 0.80), "scorn": ("disgust", 0.75),
    "sneer": ("disgust", 0.70), "mock": ("disgust", 0.65),
    "betray": ("disgust", 0.80), "traitor": ("disgust", 0.80),
    "liar": ("disgust", 0.75), "cheat": ("disgust", 0.75),
    "hypocrite": ("disgust", 0.80), "coward": ("disgust", 0.70),
    # ANGER
    "anger": ("anger", 0.90), "angry": ("anger", 0.85), "rage": ("anger", 0.95),
    "fury": ("anger", 0.95), "wrath": ("anger", 0.90), "hate": ("anger", 0.90),
    "hatred": ("anger", 0.90), "hostile": ("anger", 0.80), "violent": ("anger", 0.85),
    "fight": ("anger", 0.75), "war": ("anger", 0.80), "battle": ("anger", 0.70),
    "kill": ("anger", 0.85), "murder": ("anger", 0.90), "blood": ("anger", 0.70),
    "revenge": ("anger", 0.85), "vengeance": ("anger", 0.85),
    "curse": ("anger", 0.75), "damn": ("anger", 0.70),
    "burn": ("anger", 0.70), "fire": ("anger", 0.65), "flame": ("anger", 0.60),
    "storm": ("anger", 0.65), "thunder": ("anger", 0.65),
    "crush": ("anger", 0.75), "smash": ("anger", 0.75), "destroy": ("anger", 0.80),
    "shout": ("anger", 0.70), "roar": ("anger", 0.70), "growl": ("anger", 0.65),
    "fierce": ("anger", 0.70), "savage": ("anger", 0.75), "cruel": ("anger", 0.80),
    "brutal": ("anger", 0.80), "ruthless": ("anger", 0.80),
    "rebel": ("anger", 0.65), "resist": ("anger", 0.60), "defy": ("anger", 0.65),
    # ANTICIPATION
    "anticipation": ("anticipation", 0.80), "hope": ("anticipation", 0.80),
    "expect": ("anticipation", 0.70), "wait": ("anticipation", 0.60),
    "wish": ("anticipation", 0.75), "yearn": ("anticipation", 0.80),
    "desire": ("anticipation", 0.80), "longing": ("anticipation", 0.80),
    "crave": ("anticipation", 0.75), "hunger": ("anticipation", 0.65),
    "thirst": ("anticipation", 0.65), "seek": ("anticipation", 0.65),
    "search": ("anticipation", 0.60), "quest": ("anticipation", 0.70),
    "adventure": ("anticipation", 0.70), "journey": ("anticipation", 0.65),
    "future": ("anticipation", 0.70), "tomorrow": ("anticipation", 0.65),
    "dawn": ("anticipation", 0.70), "beginning": ("anticipation", 0.65),
    "start": ("anticipation", 0.60), "new": ("anticipation", 0.55),
    "promise": ("anticipation", 0.70), "plan": ("anticipation", 0.60),
    "prepare": ("anticipation", 0.60), "ready": ("anticipation", 0.60),
    "eager": ("anticipation", 0.75), "excited": ("anticipation", 0.80),
    "thrill": ("anticipation", 0.80), "suspense": ("anticipation", 0.75),
    "curious": ("anticipation", 0.65), "restless": ("anticipation", 0.65),
}

VAD_LEXICON: dict[str, tuple[float, float, float]] = {
    # (valence, arousal, dominance)  in [0, 1]
    "love": (0.93, 0.64, 0.63), "hate": (0.12, 0.83, 0.56),
    "happy": (0.96, 0.74, 0.72), "sad": (0.12, 0.39, 0.28),
    "joy": (0.98, 0.78, 0.72), "anger": (0.17, 0.87, 0.60),
    "fear": (0.12, 0.82, 0.18), "surprise": (0.73, 0.85, 0.40),
    "trust": (0.84, 0.38, 0.66), "disgust": (0.11, 0.69, 0.44),
    "hope": (0.89, 0.55, 0.60), "despair": (0.07, 0.60, 0.12),
    "peace": (0.89, 0.14, 0.62), "war": (0.14, 0.88, 0.58),
    "beauty": (0.92, 0.50, 0.60), "ugly": (0.16, 0.44, 0.32),
    "freedom": (0.92, 0.60, 0.82), "prison": (0.10, 0.56, 0.14),
    "smile": (0.92, 0.55, 0.64), "cry": (0.18, 0.62, 0.22),
    "laugh": (0.94, 0.78, 0.68), "weep": (0.14, 0.54, 0.20),
    "dance": (0.90, 0.78, 0.64), "mourn": (0.10, 0.42, 0.22),
    "sing": (0.88, 0.66, 0.60), "scream": (0.18, 0.92, 0.40),
    "dream": (0.82, 0.42, 0.46), "nightmare": (0.10, 0.82, 0.18),
    "sunshine": (0.92, 0.62, 0.58), "darkness": (0.16, 0.52, 0.30),
    "light": (0.86, 0.54, 0.58), "shadow": (0.28, 0.42, 0.34),
    "heaven": (0.92, 0.48, 0.60), "hell": (0.08, 0.80, 0.30),
    "angel": (0.90, 0.40, 0.52), "demon": (0.12, 0.78, 0.48),
    "sweet": (0.88, 0.40, 0.50), "bitter": (0.18, 0.46, 0.36),
    "warm": (0.84, 0.40, 0.52), "cold": (0.20, 0.36, 0.34),
    "fire": (0.50, 0.88, 0.56), "ice": (0.32, 0.32, 0.42),
    "alive": (0.88, 0.72, 0.72), "dead": (0.10, 0.30, 0.14),
    "birth": (0.84, 0.62, 0.54), "death": (0.10, 0.52, 0.22),
    "heart": (0.78, 0.54, 0.52), "blood": (0.30, 0.68, 0.42),
    "soul": (0.74, 0.44, 0.48), "ghost": (0.28, 0.56, 0.30),
    "star": (0.82, 0.48, 0.46), "storm": (0.32, 0.82, 0.44),
    "ocean": (0.72, 0.52, 0.42), "desert": (0.32, 0.38, 0.36),
    "flower": (0.86, 0.38, 0.44), "thorn": (0.22, 0.48, 0.38),
    "river": (0.72, 0.42, 0.40), "mountain": (0.66, 0.44, 0.52),
    "silence": (0.50, 0.12, 0.42), "thunder": (0.36, 0.86, 0.44),
    "rain": (0.52, 0.38, 0.34), "wind": (0.54, 0.52, 0.36),
    "gentle": (0.84, 0.24, 0.48), "fierce": (0.30, 0.86, 0.62),
    "brave": (0.82, 0.68, 0.78), "coward": (0.16, 0.48, 0.12),
    "strong": (0.78, 0.64, 0.82), "weak": (0.22, 0.34, 0.12),
    "victory": (0.92, 0.82, 0.86), "defeat": (0.12, 0.52, 0.10),
    "glory": (0.90, 0.72, 0.78), "shame": (0.10, 0.52, 0.14),
    "desire": (0.76, 0.78, 0.58), "indifference": (0.38, 0.12, 0.42),
    "passion": (0.82, 0.84, 0.62), "apathy": (0.24, 0.10, 0.28),
    "embrace": (0.86, 0.56, 0.56), "abandon": (0.14, 0.58, 0.22),
    "rise": (0.80, 0.66, 0.72), "fall": (0.24, 0.58, 0.22),
    "free": (0.90, 0.60, 0.80), "trapped": (0.10, 0.66, 0.10),
    "together": (0.86, 0.52, 0.60), "alone": (0.20, 0.36, 0.24),
    "forever": (0.76, 0.46, 0.50), "never": (0.24, 0.38, 0.34),
    "home": (0.84, 0.32, 0.60), "exile": (0.14, 0.48, 0.18),
    "mother": (0.86, 0.42, 0.52), "orphan": (0.18, 0.42, 0.16),
    "child": (0.82, 0.50, 0.36), "ancient": (0.54, 0.30, 0.42),
    "young": (0.78, 0.62, 0.52), "old": (0.46, 0.28, 0.40),
    "spring": (0.84, 0.54, 0.50), "winter": (0.36, 0.32, 0.36),
    "morning": (0.78, 0.50, 0.50), "midnight": (0.38, 0.40, 0.36),
    "sacred": (0.78, 0.36, 0.52), "profane": (0.18, 0.56, 0.42),
    "whisper": (0.60, 0.26, 0.34), "shout": (0.38, 0.86, 0.58),
    "caress": (0.84, 0.46, 0.48), "strike": (0.18, 0.82, 0.58),
    "healing": (0.86, 0.38, 0.54), "wound": (0.14, 0.58, 0.24),
    "blessing": (0.90, 0.42, 0.56), "curse": (0.12, 0.72, 0.48),
    "innocent": (0.82, 0.30, 0.36), "guilty": (0.12, 0.54, 0.22),
    "miracle": (0.92, 0.72, 0.52), "tragedy": (0.10, 0.64, 0.18),
    "promise": (0.80, 0.48, 0.56), "betrayal": (0.08, 0.76, 0.24),
    "garden": (0.82, 0.34, 0.44), "wilderness": (0.48, 0.52, 0.36),
    "king": (0.72, 0.52, 0.82), "slave": (0.08, 0.48, 0.06),
    "diamond": (0.82, 0.52, 0.54), "dust": (0.30, 0.22, 0.24),
    "gold": (0.80, 0.54, 0.60), "rust": (0.24, 0.26, 0.28),
}


@dataclass
class LyricsEmotionProfile:
    valence: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5
    plutchik_scores: dict[str, float] = field(default_factory=dict)
    emotional_intensity: float = 0.0
    emotional_words_ratio: float = 0.0
    emotion_arc_text: list[float] = field(default_factory=list)


class LyricsEmotionAnalyzer:

    def __init__(self) -> None:
        self._mlp = MultilingualProcessor()

    def analyze(self, text: str) -> LyricsEmotionProfile:
        if not text or not text.strip():
            return LyricsEmotionProfile(
                plutchik_scores={e: 0.0 for e in PLUTCHIK_EMOTIONS},
            )

        tokens = self._mlp.tokenize(text, "en")
        lower_tokens = [t.lower() for t in tokens]

        plutchik: dict[str, float] = {e: 0.0 for e in PLUTCHIK_EMOTIONS}
        vad_v: list[float] = []
        vad_a: list[float] = []
        vad_d: list[float] = []
        emo_count = 0

        for w in lower_tokens:
            if w in EMOTION_LEXICON:
                emotion, intensity = EMOTION_LEXICON[w]
                plutchik[emotion] = max(plutchik[emotion], intensity)
                emo_count += 1
            if w in VAD_LEXICON:
                v, a, d = VAD_LEXICON[w]
                vad_v.append(v)
                vad_a.append(a)
                vad_d.append(d)

        valence = sum(vad_v) / len(vad_v) if vad_v else 0.5
        arousal = sum(vad_a) / len(vad_a) if vad_a else 0.5
        dominance = sum(vad_d) / len(vad_d) if vad_d else 0.5

        total_intensity = sum(plutchik.values())
        n_emotions = sum(1 for v in plutchik.values() if v > 0)
        emotional_intensity = total_intensity / max(n_emotions, 1)

        emo_ratio = emo_count / max(len(lower_tokens), 1)

        arc = self.emotion_arc(text)

        return LyricsEmotionProfile(
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            dominance=round(dominance, 4),
            plutchik_scores={k: round(v, 4) for k, v in plutchik.items()},
            emotional_intensity=round(emotional_intensity, 4),
            emotional_words_ratio=round(emo_ratio, 4),
            emotion_arc_text=arc,
        )

    def per_line_emotions(self, text: str) -> list[dict[str, float]]:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        results: list[dict[str, float]] = []
        for ln in lines:
            tokens = self._mlp.tokenize(ln, "en")
            lower = [t.lower() for t in tokens]
            scores: dict[str, float] = {e: 0.0 for e in PLUTCHIK_EMOTIONS}
            for w in lower:
                if w in EMOTION_LEXICON:
                    emotion, intensity = EMOTION_LEXICON[w]
                    scores[emotion] = max(scores[emotion], intensity)
            results.append(scores)
        return results

    def emotion_arc(self, text: str) -> list[float]:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        arc: list[float] = []
        for ln in lines:
            tokens = self._mlp.tokenize(ln, "en")
            lower = [t.lower() for t in tokens]
            vals = [VAD_LEXICON[w][0] for w in lower if w in VAD_LEXICON]
            if vals:
                arc.append(round(sum(vals) / len(vals), 4))
            else:
                arc.append(0.5)
        return arc

    def emotional_intensity_curve(self, text: str) -> list[float]:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        curve: list[float] = []
        for ln in lines:
            tokens = self._mlp.tokenize(ln, "en")
            lower = [t.lower() for t in tokens]
            intensities = []
            for w in lower:
                if w in EMOTION_LEXICON:
                    _, intensity = EMOTION_LEXICON[w]
                    intensities.append(intensity)
            if intensities:
                curve.append(round(sum(intensities) / len(intensities), 4))
            else:
                curve.append(0.0)
        return curve
