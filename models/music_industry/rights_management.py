"""Rights management: ownership splits, licensing, royalty chains, conflict resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class RightType(Enum):
    COMPOSITION = "composition"
    MASTER = "master"
    PERFORMANCE = "performance"
    SYNC = "sync"
    MECHANICAL = "mechanical"
    PRINT = "print"


class LicenseType(Enum):
    EXCLUSIVE = "exclusive"
    NON_EXCLUSIVE = "non_exclusive"
    SYNC = "sync"
    MECHANICAL = "mechanical"
    BLANKET = "blanket"


@dataclass(frozen=True)
class OwnershipShare:
    holder: str
    share_pct: float
    right_type: RightType
    territory: str = "worldwide"

    def revenue(self, total: float) -> float:
        return total * self.share_pct / 100.0


@dataclass(frozen=True)
class License:
    license_type: LicenseType
    grantor: str
    grantee: str
    territory: str = "worldwide"
    duration_months: int = 12
    fee: float = 0.0
    revenue_share_pct: float = 0.0


@dataclass
class RightsBundle:
    work_id: str
    title: str
    shares: list[OwnershipShare] = field(default_factory=list)
    licenses: list[License] = field(default_factory=list)

    def total_share(self, right_type: RightType | None = None) -> float:
        if right_type is None:
            return sum(s.share_pct for s in self.shares)
        return sum(s.share_pct for s in self.shares if s.right_type == right_type)

    def is_fully_allocated(self, right_type: RightType) -> bool:
        return abs(self.total_share(right_type) - 100.0) < 0.01

    def holders(self) -> set[str]:
        return {s.holder for s in self.shares}


def validate_splits(shares: Sequence[OwnershipShare]) -> list[str]:
    issues: list[str] = []
    by_type: dict[RightType, float] = {}
    for s in shares:
        by_type[s.right_type] = by_type.get(s.right_type, 0.0) + s.share_pct
    for rt, total in by_type.items():
        if total > 100.01:
            issues.append(f"{rt.value}: over-allocated at {total:.2f}%")
        elif total < 99.99:
            issues.append(f"{rt.value}: under-allocated at {total:.2f}%")
    return issues


def detect_conflicts(bundles: Sequence[RightsBundle]) -> list[str]:
    conflicts: list[str] = []
    for b in bundles:
        for rt in RightType:
            total = b.total_share(rt)
            if total > 100.01:
                conflicts.append(f"{b.work_id}/{rt.value}: {total:.1f}% > 100%")
        exc_licenses = [l for l in b.licenses if l.license_type == LicenseType.EXCLUSIVE]
        if len(exc_licenses) > 1:
            conflicts.append(f"{b.work_id}: multiple exclusive licenses")
    return conflicts


@dataclass(frozen=True)
class RoyaltyWaterfall:
    stages: list[tuple[str, float]]

    def distribute(self, gross: float) -> dict[str, float]:
        result: dict[str, float] = {}
        remaining = gross
        for holder, pct in self.stages:
            amount = remaining * pct / 100.0
            result[holder] = result.get(holder, 0.0) + amount
            remaining -= amount
        return result


def compute_waterfall(
    shares: Sequence[OwnershipShare],
    admin_fee_pct: float = 15.0,
    admin_holder: str = "publisher",
) -> RoyaltyWaterfall:
    stages: list[tuple[str, float]] = [(admin_holder, admin_fee_pct)]
    remaining = 100.0 - admin_fee_pct
    total_share = sum(s.share_pct for s in shares)
    if total_share > 0:
        for s in shares:
            pct = (s.share_pct / total_share) * remaining
            stages.append((s.holder, pct))
    return RoyaltyWaterfall(stages=stages)


def license_value_estimate(
    license: License,
    monthly_streams: float,
    per_stream_rate: float = 0.004,
) -> float:
    if license.fee > 0:
        return license.fee
    monthly_rev = monthly_streams * per_stream_rate
    return monthly_rev * license.revenue_share_pct / 100.0 * license.duration_months


def territory_overlap(
    shares: Sequence[OwnershipShare],
) -> dict[str, list[str]]:
    by_territory: dict[str, list[str]] = {}
    for s in shares:
        by_territory.setdefault(s.territory, []).append(s.holder)
    return by_territory
