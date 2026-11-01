"""Context-aware playlist filtering: time-of-day, activity, weather, social
setting, and listener state.  Pure numpy scoring — no external APIs."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class TimeOfDay(str, Enum):
    EARLY_MORNING = "early_morning"
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    LATE_NIGHT = "late_night"


class Activity(str, Enum):
    WORKOUT = "workout"
    STUDY = "study"
    COMMUTE = "commute"
    COOKING = "cooking"
    RELAXATION = "relaxation"
    PARTY = "party"
    MEDITATION = "meditation"
    SLEEP = "sleep"
    WORK_FOCUS = "work_focus"
    SOCIAL = "social"


class Weather(str, Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    SNOWY = "snowy"
    FOGGY = "foggy"


class SocialSetting(str, Enum):
    ALONE = "alone"
    COUPLE = "couple"
    SMALL_GROUP = "small_group"
    PARTY = "party"
    PUBLIC = "public"


@dataclass
class ListenerContext:
    time_of_day: TimeOfDay = TimeOfDay.AFTERNOON
    activity: Activity = Activity.RELAXATION
    weather: Optional[Weather] = None
    social: SocialSetting = SocialSetting.ALONE
    energy_level: float = 0.5
    stress_level: float = 0.3


@dataclass
class TrackFeatures:
    energy: float = 0.5
    valence: float = 0.5
    tempo_bpm: float = 120.0
    loudness_db: float = -8.0
    acousticness: float = 0.3
    instrumentalness: float = 0.2
    danceability: float = 0.5


TIME_ENERGY_TARGETS: Dict[TimeOfDay, Tuple[float, float]] = {
    TimeOfDay.EARLY_MORNING: (0.15, 0.45),
    TimeOfDay.MORNING: (0.35, 0.65),
    TimeOfDay.MIDDAY: (0.50, 0.80),
    TimeOfDay.AFTERNOON: (0.40, 0.70),
    TimeOfDay.EVENING: (0.30, 0.65),
    TimeOfDay.NIGHT: (0.15, 0.50),
    TimeOfDay.LATE_NIGHT: (0.05, 0.35),
}

ACTIVITY_PROFILES: Dict[Activity, Dict[str, Tuple[float, float]]] = {
    Activity.WORKOUT: {"energy": (0.7, 1.0), "tempo_bpm": (120, 180), "danceability": (0.6, 1.0)},
    Activity.STUDY: {"energy": (0.0, 0.4), "instrumentalness": (0.5, 1.0), "loudness_db": (-20, -10)},
    Activity.COMMUTE: {"energy": (0.3, 0.7), "valence": (0.3, 0.8)},
    Activity.COOKING: {"energy": (0.4, 0.7), "valence": (0.5, 0.9), "danceability": (0.4, 0.8)},
    Activity.RELAXATION: {"energy": (0.0, 0.4), "acousticness": (0.4, 1.0)},
    Activity.PARTY: {"energy": (0.7, 1.0), "danceability": (0.7, 1.0), "valence": (0.5, 1.0)},
    Activity.MEDITATION: {"energy": (0.0, 0.2), "instrumentalness": (0.7, 1.0), "tempo_bpm": (50, 90)},
    Activity.SLEEP: {"energy": (0.0, 0.15), "instrumentalness": (0.6, 1.0), "loudness_db": (-30, -15)},
    Activity.WORK_FOCUS: {"energy": (0.1, 0.45), "instrumentalness": (0.4, 1.0)},
    Activity.SOCIAL: {"energy": (0.4, 0.8), "valence": (0.4, 0.9)},
}

WEATHER_VALENCE_SHIFT: Dict[Weather, float] = {
    Weather.SUNNY: 0.10,
    Weather.CLOUDY: -0.05,
    Weather.RAINY: -0.10,
    Weather.STORMY: -0.15,
    Weather.SNOWY: 0.0,
    Weather.FOGGY: -0.08,
}


def _range_score(value: float, lo: float, hi: float) -> float:
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return max(0.0, 1.0 - (lo - value) / max(hi - lo, 0.01))
    return max(0.0, 1.0 - (value - hi) / max(hi - lo, 0.01))


def time_score(track: TrackFeatures, time: TimeOfDay) -> float:
    lo, hi = TIME_ENERGY_TARGETS[time]
    return _range_score(track.energy, lo, hi)


def activity_score(track: TrackFeatures, activity: Activity) -> float:
    profile = ACTIVITY_PROFILES.get(activity, {})
    if not profile:
        return 0.5
    scores = []
    for attr, (lo, hi) in profile.items():
        val = getattr(track, attr, None)
        if val is not None:
            scores.append(_range_score(val, lo, hi))
    return float(np.mean(scores)) if scores else 0.5


def weather_score(track: TrackFeatures, weather: Weather) -> float:
    shift = WEATHER_VALENCE_SHIFT.get(weather, 0.0)
    target = 0.5 + shift
    return max(0.0, 1.0 - abs(track.valence - target))


def social_score(track: TrackFeatures, social: SocialSetting) -> float:
    if social == SocialSetting.ALONE:
        return 1.0
    if social == SocialSetting.PARTY:
        return _range_score(track.energy, 0.6, 1.0) * _range_score(track.danceability, 0.6, 1.0)
    if social == SocialSetting.PUBLIC:
        valence_ok = _range_score(track.valence, 0.3, 0.8)
        loud_ok = _range_score(track.loudness_db, -15, -5)
        return 0.5 * valence_ok + 0.5 * loud_ok
    if social == SocialSetting.COUPLE:
        return _range_score(track.energy, 0.15, 0.55) * _range_score(track.acousticness, 0.3, 1.0)
    return _range_score(track.energy, 0.3, 0.7)


def energy_alignment(track: TrackFeatures, listener_energy: float) -> float:
    return max(0.0, 1.0 - abs(track.energy - listener_energy))


def stress_compensation(track: TrackFeatures, stress: float) -> float:
    if stress > 0.7:
        return _range_score(track.energy, 0.0, 0.4) * _range_score(track.valence, 0.3, 0.7)
    if stress < 0.3:
        return 1.0
    return _range_score(track.energy, 0.2, 0.6)


def context_score(track: TrackFeatures, ctx: ListenerContext) -> float:
    s_time = time_score(track, ctx.time_of_day)
    s_act = activity_score(track, ctx.activity)
    s_social = social_score(track, ctx.social)
    s_energy = energy_alignment(track, ctx.energy_level)
    s_stress = stress_compensation(track, ctx.stress_level)
    parts = [0.25 * s_act, 0.20 * s_time, 0.15 * s_social,
             0.20 * s_energy, 0.10 * s_stress]
    if ctx.weather is not None:
        s_w = weather_score(track, ctx.weather)
        parts.append(0.10 * s_w)
    else:
        parts[0] += 0.10 * s_act
    return float(np.sum(parts))


def rank_tracks(
    tracks: List[TrackFeatures], ctx: ListenerContext,
) -> List[Tuple[int, float]]:
    scored = [(i, context_score(t, ctx)) for i, t in enumerate(tracks)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def filter_tracks(
    tracks: List[TrackFeatures], ctx: ListenerContext, threshold: float = 0.5,
) -> List[int]:
    return [i for i, t in enumerate(tracks) if context_score(t, ctx) >= threshold]


def transition_smoothness(a: TrackFeatures, b: TrackFeatures) -> float:
    de = abs(a.energy - b.energy)
    dv = abs(a.valence - b.valence)
    dt = abs(a.tempo_bpm - b.tempo_bpm) / 200.0
    return max(0.0, 1.0 - 0.4 * de - 0.3 * dv - 0.3 * dt)


def optimal_context_order(
    tracks: List[TrackFeatures], ctx: ListenerContext,
) -> List[int]:
    scores = [context_score(t, ctx) for t in tracks]
    indices = list(range(len(tracks)))
    if not indices:
        return []
    ordered = [indices[int(np.argmax(scores))]]
    remaining = set(indices) - set(ordered)
    while remaining:
        last = ordered[-1]
        best_idx = -1
        best_val = -1.0
        for r in remaining:
            combined = 0.6 * scores[r] + 0.4 * transition_smoothness(tracks[last], tracks[r])
            if combined > best_val:
                best_val = combined
                best_idx = r
        ordered.append(best_idx)
        remaining.remove(best_idx)
    return ordered


def hour_to_time_of_day(hour: int) -> TimeOfDay:
    if hour < 5:
        return TimeOfDay.LATE_NIGHT
    if hour < 7:
        return TimeOfDay.EARLY_MORNING
    if hour < 10:
        return TimeOfDay.MORNING
    if hour < 13:
        return TimeOfDay.MIDDAY
    if hour < 17:
        return TimeOfDay.AFTERNOON
    if hour < 21:
        return TimeOfDay.EVENING
    if hour < 24:
        return TimeOfDay.NIGHT
    return TimeOfDay.LATE_NIGHT


def daily_energy_curve(hours: np.ndarray) -> np.ndarray:
    peak = 14.0
    sigma = 5.0
    return np.exp(-0.5 * ((hours - peak) / sigma) ** 2)
