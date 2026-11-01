"""Release strategy: rollout planning, single vs album economics, marketing windows."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from datetime import date
import math


class ReleaseType(Enum):
    SINGLE = "single"
    EP = "ep"
    ALBUM = "album"
    DELUXE = "deluxe"
    COMPILATION = "compilation"
    LIVE = "live"
    REMIX = "remix"


class RolloutPhase(Enum):
    PRE_SAVE = "pre_save"
    ANNOUNCE = "announce"
    SINGLE_DROP = "single_drop"
    MUSIC_VIDEO = "music_video"
    ALBUM_RELEASE = "album_release"
    DELUXE_REISSUE = "deluxe_reissue"
    TOUR_ANNOUNCE = "tour_announce"


@dataclass
class ReleaseWindow:
    phase: RolloutPhase
    days_before_release: int
    description: str


STANDARD_ROLLOUT: list[ReleaseWindow] = [
    ReleaseWindow(RolloutPhase.PRE_SAVE, 28, "Pre-save campaign on DSPs"),
    ReleaseWindow(RolloutPhase.ANNOUNCE, 21, "Social announcement + artwork reveal"),
    ReleaseWindow(RolloutPhase.SINGLE_DROP, 14, "Lead single release"),
    ReleaseWindow(RolloutPhase.MUSIC_VIDEO, 7, "Music video premiere"),
    ReleaseWindow(RolloutPhase.ALBUM_RELEASE, 0, "Full album drop"),
    ReleaseWindow(RolloutPhase.DELUXE_REISSUE, -30, "Deluxe edition / bonus tracks"),
]


@dataclass
class ReleasePlan:
    title: str
    release_type: ReleaseType
    target_date: date
    track_count: int
    rollout: list[ReleaseWindow] = field(default_factory=lambda: list(STANDARD_ROLLOUT))
    territory_stagger: bool = False

    def timeline(self) -> list[tuple[date, str]]:
        from datetime import timedelta
        return [
            (self.target_date + timedelta(days=w.days_before_release), w.description)
            for w in sorted(self.rollout, key=lambda w: -w.days_before_release)
        ]


def optimal_release_day() -> str:
    return "Friday"


def single_vs_album_roi(
    single_streams_per_track: int,
    album_streams_per_track: int,
    n_tracks: int,
    production_cost_single: float,
    production_cost_album: float,
    per_stream_rate: float = 0.004,
) -> dict[str, float]:
    single_rev = single_streams_per_track * per_stream_rate * n_tracks
    single_cost = production_cost_single * n_tracks
    album_rev = album_streams_per_track * per_stream_rate * n_tracks
    album_cost = production_cost_album
    return {
        "single_strategy_revenue": single_rev,
        "single_strategy_cost": single_cost,
        "single_strategy_roi": (single_rev - single_cost) / max(single_cost, 1),
        "album_strategy_revenue": album_rev,
        "album_strategy_cost": album_cost,
        "album_strategy_roi": (album_rev - album_cost) / max(album_cost, 1),
        "recommended": "singles" if (single_rev - single_cost) > (album_rev - album_cost) else "album",
    }


def playlist_pitch_score(
    track_energy: float,
    track_danceability: float,
    track_popularity: float,
    artist_monthly_listeners: int,
) -> float:
    base = 0.3 * track_energy + 0.3 * track_danceability + 0.2 * track_popularity
    listener_bonus = min(0.2, math.log10(max(artist_monthly_listeners, 1)) / 35)
    return min(1.0, base + listener_bonus)


def waterfall_schedule(
    release_date: date,
    n_singles: int = 3,
    weeks_apart: int = 4,
) -> list[tuple[date, str]]:
    from datetime import timedelta
    schedule = []
    for i in range(n_singles):
        d = release_date - timedelta(weeks=(n_singles - i) * weeks_apart)
        schedule.append((d, f"Single {i + 1}"))
    schedule.append((release_date, "Album release"))
    return schedule
