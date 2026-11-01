"""Music rights modeling: ownership splits, licensing, royalty structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class RightType(Enum):
    COMPOSITION = "composition"
    MASTER_RECORDING = "master_recording"
    PERFORMANCE = "performance"
    ARRANGEMENT = "arrangement"
    LYRICS = "lyrics"
    SYNC = "sync"
    MECHANICAL = "mechanical"
    PUBLIC_PERFORMANCE = "public_performance"
    DIGITAL_PERFORMANCE = "digital_performance"
    PRINT = "print"


class LicenseType(Enum):
    EXCLUSIVE = "exclusive"
    NON_EXCLUSIVE = "non_exclusive"
    SYNC = "sync"
    MECHANICAL = "mechanical"
    BLANKET = "blanket"
    COMPULSORY = "compulsory"
    CREATIVE_COMMONS = "creative_commons"
    PUBLIC_DOMAIN = "public_domain"


@dataclass(frozen=True)
class RightsHolder:
    name: str
    share: float
    right_type: RightType
    territory: str = "worldwide"
    pro: str | None = None

    def validate(self) -> bool:
        return 0.0 <= self.share <= 1.0


@dataclass
class RightsChain:
    holders: list[RightsHolder] = field(default_factory=list)

    def add_holder(self, holder: RightsHolder) -> None:
        self.holders.append(holder)

    def total_share(self, right_type: RightType | None = None) -> float:
        relevant = self.holders
        if right_type is not None:
            relevant = [h for h in self.holders if h.right_type == right_type]
        return sum(h.share for h in relevant)

    def is_complete(self, right_type: RightType) -> bool:
        return abs(self.total_share(right_type) - 1.0) < 0.001

    def validate(self) -> list[str]:
        errors: list[str] = []
        for h in self.holders:
            if not h.validate():
                errors.append(f"{h.name}: share {h.share} out of range")
        by_type: dict[RightType, float] = {}
        for h in self.holders:
            by_type[h.right_type] = by_type.get(h.right_type, 0) + h.share
        for rt, total in by_type.items():
            if total > 1.001:
                errors.append(f"{rt.value}: total shares {total:.3f} > 1.0")
        return errors

    def holder_royalty(self, holder_name: str, gross: float) -> float:
        total = 0.0
        for h in self.holders:
            if h.name == holder_name:
                total += h.share * gross
        return total


@dataclass(frozen=True)
class SampleClearance:
    source_track: str
    source_artist: str
    duration_seconds: float
    cleared: bool = False
    fee: float = 0.0
    royalty_share: float = 0.0
    notes: str = ""


def calculate_split(contributions: dict[str, float]) -> dict[str, float]:
    total = sum(contributions.values())
    if total <= 0:
        return {name: 0.0 for name in contributions}
    return {name: val / total for name, val in contributions.items()}


def mechanical_royalty_rate(year: int = 2024) -> float:
    if year >= 2023:
        return 0.12
    if year >= 2018:
        return 0.091
    return 0.091


def streaming_royalty_estimate(streams: int, per_stream: float = 0.004,
                               share: float = 1.0) -> float:
    return streams * per_stream * share


STANDARD_SPLITS: dict[str, dict[str, float]] = {
    "songwriter_publisher": {"songwriter": 0.5, "publisher": 0.5},
    "producer_artist": {"producer": 0.2, "artist": 0.8},
    "band_equal_4": {"member_1": 0.25, "member_2": 0.25, "member_3": 0.25, "member_4": 0.25},
    "lyrics_music": {"lyricist": 0.5, "composer": 0.5},
    "collab_3_way": {"writer_1": 0.34, "writer_2": 0.33, "writer_3": 0.33},
}
