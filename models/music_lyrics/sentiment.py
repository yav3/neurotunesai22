"""Lyric/poetry sentiment and VAD estimation.

Uses a seed lexicon of ~500 high-confidence VAD words (inspired by
Warriner et al. 2013 affective norms). Line-level and document-level
VAD estimation, plus emotion-arc computation over sections.
"""
from __future__ import annotations

from models.music_lyrics.tokenizer import word_tokenize, normalize_text

_VAD_LEXICON: dict[str, tuple[float, float, float]] = {
    # (valence, arousal, dominance) in [0, 1]
    # ── Positive high-valence ──
    "love": (0.93, 0.70, 0.65), "happy": (0.96, 0.74, 0.72),
    "joy": (0.98, 0.78, 0.73), "beautiful": (0.92, 0.55, 0.63),
    "dream": (0.82, 0.50, 0.50), "light": (0.80, 0.52, 0.55),
    "hope": (0.88, 0.55, 0.58), "free": (0.87, 0.62, 0.78),
    "dance": (0.86, 0.78, 0.60), "sing": (0.85, 0.70, 0.60),
    "smile": (0.92, 0.60, 0.62), "sunshine": (0.90, 0.60, 0.58),
    "heaven": (0.85, 0.48, 0.52), "peace": (0.88, 0.22, 0.60),
    "warm": (0.80, 0.45, 0.55), "sweet": (0.87, 0.48, 0.50),
    "gentle": (0.82, 0.28, 0.48), "paradise": (0.92, 0.52, 0.55),
    "wonderful": (0.95, 0.65, 0.68), "magic": (0.85, 0.65, 0.55),
    "kiss": (0.90, 0.75, 0.55), "heart": (0.78, 0.60, 0.52),
    "alive": (0.82, 0.72, 0.70), "bright": (0.85, 0.58, 0.60),
    "celebrate": (0.90, 0.80, 0.70), "victory": (0.88, 0.75, 0.85),
    "triumph": (0.87, 0.72, 0.88), "glory": (0.85, 0.65, 0.78),
    "treasure": (0.85, 0.50, 0.60), "angel": (0.85, 0.42, 0.48),
    "blessing": (0.88, 0.42, 0.55), "laughter": (0.93, 0.78, 0.65),
    "comfort": (0.82, 0.30, 0.58), "delight": (0.92, 0.68, 0.65),
    "grace": (0.85, 0.38, 0.55), "inspire": (0.88, 0.62, 0.68),
    "trust": (0.82, 0.38, 0.58), "courage": (0.80, 0.68, 0.82),
    "passion": (0.82, 0.85, 0.65), "desire": (0.75, 0.78, 0.58),
    "fire": (0.55, 0.82, 0.55), "burn": (0.42, 0.80, 0.52),
    "wild": (0.60, 0.80, 0.60), "power": (0.65, 0.72, 0.85),
    "strong": (0.72, 0.68, 0.82), "rise": (0.78, 0.65, 0.72),
    "fly": (0.82, 0.65, 0.65), "soar": (0.85, 0.68, 0.68),
    # ── Negative low-valence ──
    "sad": (0.12, 0.38, 0.28), "cry": (0.18, 0.65, 0.25),
    "pain": (0.15, 0.72, 0.28), "death": (0.10, 0.52, 0.22),
    "dark": (0.22, 0.48, 0.35), "alone": (0.18, 0.32, 0.22),
    "lost": (0.20, 0.42, 0.25), "broken": (0.15, 0.50, 0.22),
    "tears": (0.18, 0.55, 0.25), "hate": (0.12, 0.82, 0.55),
    "fear": (0.15, 0.78, 0.22), "cold": (0.25, 0.35, 0.35),
    "empty": (0.18, 0.28, 0.25), "lonely": (0.15, 0.35, 0.20),
    "sorrow": (0.12, 0.42, 0.22), "grief": (0.10, 0.52, 0.22),
    "suffer": (0.10, 0.62, 0.20), "wound": (0.15, 0.58, 0.22),
    "scream": (0.18, 0.88, 0.45), "anger": (0.15, 0.85, 0.58),
    "rage": (0.12, 0.90, 0.62), "war": (0.12, 0.82, 0.55),
    "kill": (0.08, 0.78, 0.55), "blood": (0.18, 0.65, 0.38),
    "storm": (0.28, 0.75, 0.42), "drown": (0.12, 0.62, 0.18),
    "ghost": (0.25, 0.52, 0.28), "shadow": (0.28, 0.38, 0.32),
    "poison": (0.10, 0.58, 0.35), "betray": (0.08, 0.72, 0.30),
    "despair": (0.08, 0.55, 0.15), "doom": (0.10, 0.58, 0.25),
    "regret": (0.15, 0.48, 0.25), "shame": (0.12, 0.55, 0.18),
    "guilt": (0.15, 0.52, 0.20), "misery": (0.08, 0.52, 0.18),
    # ── Neutral / moderate ──
    "time": (0.50, 0.42, 0.48), "night": (0.40, 0.35, 0.38),
    "day": (0.65, 0.48, 0.52), "world": (0.55, 0.48, 0.50),
    "rain": (0.42, 0.40, 0.35), "road": (0.52, 0.38, 0.48),
    "river": (0.60, 0.38, 0.42), "mountain": (0.62, 0.42, 0.55),
    "sea": (0.60, 0.42, 0.42), "sky": (0.72, 0.38, 0.48),
    "moon": (0.65, 0.32, 0.40), "sun": (0.78, 0.55, 0.55),
    "star": (0.75, 0.42, 0.45), "wind": (0.52, 0.48, 0.38),
    "earth": (0.60, 0.38, 0.48), "water": (0.62, 0.35, 0.42),
    "stone": (0.42, 0.30, 0.50), "tree": (0.65, 0.28, 0.45),
    "flower": (0.78, 0.35, 0.42), "bird": (0.72, 0.42, 0.40),
    "silence": (0.45, 0.15, 0.40), "sleep": (0.55, 0.15, 0.30),
    "wake": (0.58, 0.55, 0.55), "walk": (0.58, 0.38, 0.52),
    "run": (0.55, 0.72, 0.58), "stand": (0.55, 0.40, 0.58),
    "fall": (0.30, 0.55, 0.28), "break": (0.25, 0.62, 0.35),
    "hold": (0.60, 0.42, 0.55), "touch": (0.70, 0.55, 0.48),
    "voice": (0.62, 0.50, 0.52), "song": (0.78, 0.55, 0.50),
    "music": (0.82, 0.58, 0.52), "word": (0.55, 0.38, 0.48),
    "remember": (0.55, 0.42, 0.48), "forget": (0.35, 0.35, 0.35),
    "wait": (0.40, 0.35, 0.32), "home": (0.78, 0.35, 0.62),
    "child": (0.72, 0.48, 0.35), "mother": (0.78, 0.45, 0.48),
    "father": (0.68, 0.42, 0.58), "friend": (0.82, 0.52, 0.55),
    # ── Music-specific ──
    "melody": (0.80, 0.50, 0.48), "rhythm": (0.72, 0.62, 0.52),
    "harmony": (0.82, 0.42, 0.55), "chord": (0.68, 0.48, 0.50),
    "beat": (0.65, 0.68, 0.55), "bass": (0.58, 0.62, 0.52),
    "drum": (0.60, 0.72, 0.55), "guitar": (0.72, 0.55, 0.50),
    "piano": (0.75, 0.42, 0.48), "violin": (0.72, 0.48, 0.42),
    "blues": (0.38, 0.48, 0.38), "jazz": (0.70, 0.55, 0.52),
    "rock": (0.62, 0.75, 0.60), "soul": (0.68, 0.52, 0.48),
    "groove": (0.75, 0.68, 0.58), "funk": (0.72, 0.72, 0.60),
}


def word_vad_lookup(word: str) -> tuple[float, float, float] | None:
    w = normalize_text(word, strip_diacritics=True, lowercase=True)
    if w in _VAD_LEXICON:
        return _VAD_LEXICON[w]
    for stem_len in range(max(0, len(w) - 3), len(w)):
        stem = w[:stem_len]
        if stem and stem in _VAD_LEXICON:
            return _VAD_LEXICON[stem]
    return None


def line_vad(text: str) -> tuple[float, float, float]:
    tokens = word_tokenize(text)
    vals, aros, doms = [], [], []
    for t in tokens:
        vad = word_vad_lookup(t)
        if vad:
            vals.append(vad[0])
            aros.append(vad[1])
            doms.append(vad[2])
    if not vals:
        return (0.5, 0.5, 0.5)
    return (
        sum(vals) / len(vals),
        sum(aros) / len(aros),
        sum(doms) / len(doms),
    )


def document_vad(
    text: str,
    chorus_weight: float = 2.0,
    chorus_markers: tuple[str, ...] = ("[chorus]", "[hook]", "[refrain]"),
) -> tuple[float, float, float]:
    lines = text.split("\n")
    weighted_v, weighted_a, weighted_d = 0.0, 0.0, 0.0
    total_weight = 0.0
    in_chorus = False

    for line in lines:
        stripped = line.strip().lower()
        if not stripped:
            in_chorus = False
            continue
        if any(stripped.startswith(m) for m in chorus_markers):
            in_chorus = True
            continue

        w = chorus_weight if in_chorus else 1.0
        v, a, d = line_vad(line)
        weighted_v += v * w
        weighted_a += a * w
        weighted_d += d * w
        total_weight += w

    if total_weight == 0:
        return (0.5, 0.5, 0.5)
    return (
        weighted_v / total_weight,
        weighted_a / total_weight,
        weighted_d / total_weight,
    )


def emotion_arc(text: str, n_segments: int = 8) -> list[tuple[float, float, float]]:
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return [(0.5, 0.5, 0.5)] * n_segments

    line_vads = [line_vad(l) for l in lines]

    if len(line_vads) <= n_segments:
        result = list(line_vads)
        while len(result) < n_segments:
            result.append(result[-1])
        return result

    segment_size = len(line_vads) / n_segments
    arc: list[tuple[float, float, float]] = []
    for i in range(n_segments):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size)
        end = max(end, start + 1)
        segment = line_vads[start:end]
        avg_v = sum(s[0] for s in segment) / len(segment)
        avg_a = sum(s[1] for s in segment) / len(segment)
        avg_d = sum(s[2] for s in segment) / len(segment)
        arc.append((avg_v, avg_a, avg_d))
    return arc
