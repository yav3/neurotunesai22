"""Heuristic auto-tagging from audio feature profiles."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AudioFeatureProfile:
    tempo_bpm: float
    key: int  # 0-11
    mode: int  # 0=minor, 1=major
    energy: float  # 0-1
    danceability: float  # 0-1
    loudness_lufs: float


_GENRE_RULES: list[tuple[str, dict]] = [
    ("classical", {"energy_max": 0.4, "tempo_max": 140, "loudness_max": -20}),
    ("electronic", {"energy_min": 0.6, "tempo_min": 115, "tempo_max": 145, "danceability_min": 0.6}),
    ("hip-hop", {"tempo_min": 70, "tempo_max": 110, "energy_min": 0.5}),
    ("rock", {"energy_min": 0.65, "tempo_min": 100, "tempo_max": 170, "loudness_min": -14}),
    ("jazz", {"energy_max": 0.6, "tempo_min": 80, "tempo_max": 200}),
    ("pop", {"danceability_min": 0.5, "tempo_min": 95, "tempo_max": 135, "energy_min": 0.4}),
    ("ambient", {"energy_max": 0.3, "tempo_max": 100, "loudness_max": -25}),
    ("metal", {"energy_min": 0.8, "tempo_min": 100, "loudness_min": -10}),
    ("folk", {"energy_max": 0.5, "tempo_min": 80, "tempo_max": 150}),
    ("r&b", {"danceability_min": 0.55, "tempo_min": 60, "tempo_max": 110, "energy_max": 0.7}),
]

_MOOD_MAP: list[tuple[str, dict]] = [
    ("energetic", {"energy_min": 0.7, "tempo_min": 120}),
    ("melancholic", {"mode": 0, "energy_max": 0.4, "tempo_max": 100}),
    ("uplifting", {"mode": 1, "energy_min": 0.5, "tempo_min": 110}),
    ("dark", {"mode": 0, "energy_min": 0.6, "loudness_min": -12}),
    ("calm", {"energy_max": 0.3, "tempo_max": 90}),
    ("aggressive", {"energy_min": 0.8, "loudness_min": -10}),
    ("romantic", {"mode": 1, "energy_max": 0.5, "tempo_min": 60, "tempo_max": 110}),
    ("dreamy", {"energy_max": 0.35, "mode": 1}),
]


def _match_rule(profile: AudioFeatureProfile, rule: dict) -> float:
    score = 1.0
    checks = 0
    for key, val in rule.items():
        attr, op = key.rsplit("_", 1)
        pval = getattr(profile, attr, None)
        if pval is None:
            continue
        checks += 1
        if op == "min" and pval < val:
            score *= max(0.0, 1.0 - (val - pval) / (val + 1e-6))
        elif op == "max" and pval > val:
            score *= max(0.0, 1.0 - (pval - val) / (val + 1e-6))
        elif key == "mode" and pval != val:
            score *= 0.3
    return score if checks > 0 else 0.0


def estimate_genre(profile: AudioFeatureProfile) -> list[tuple[str, float]]:
    scores = [(name, _match_rule(profile, rule)) for name, rule in _GENRE_RULES]
    scores.sort(key=lambda x: -x[1])
    return [(n, s) for n, s in scores if s > 0.2][:5]


def estimate_mood(profile: AudioFeatureProfile) -> list[tuple[str, float]]:
    scores = [(name, _match_rule(profile, rule)) for name, rule in _MOOD_MAP]
    scores.sort(key=lambda x: -x[1])
    return [(n, s) for n, s in scores if s > 0.2][:4]


def estimate_era(production_features: dict) -> tuple[str, float]:
    """Heuristic era estimation from spectral/production characteristics."""
    loudness = production_features.get("loudness_lufs", -14)
    bandwidth = production_features.get("spectral_bandwidth", 2000)
    stereo_width = production_features.get("stereo_width", 0.5)

    if loudness > -8 and bandwidth > 3000:
        return "2010s-present", 0.7
    if loudness > -12 and stereo_width > 0.6:
        return "2000s", 0.5
    if stereo_width < 0.3:
        return "pre-1970s", 0.6
    if bandwidth < 2000:
        return "1970s-1980s", 0.5
    return "1990s", 0.4


def suggest_tags(profile: AudioFeatureProfile, existing: dict) -> dict:
    suggestions = dict(existing)
    if "genre" not in suggestions:
        genres = estimate_genre(profile)
        if genres:
            suggestions["genre"] = genres[0][0]
    if "mood" not in suggestions:
        moods = estimate_mood(profile)
        if moods:
            suggestions["mood"] = moods[0][0]
    if "bpm" not in suggestions:
        suggestions["bpm"] = str(int(round(profile.tempo_bpm)))
    return suggestions


def tag_confidence(auto_tags: dict, existing_tags: dict) -> float:
    if not auto_tags or not existing_tags:
        return 0.0
    matches = sum(1 for k in auto_tags if k in existing_tags and auto_tags[k] == existing_tags[k])
    total = len(set(auto_tags) | set(existing_tags))
    return matches / total if total > 0 else 0.0
