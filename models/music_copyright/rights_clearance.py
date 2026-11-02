"""Rights clearance: mechanical licensing, royalty splits, usage rights,
sync licensing, territory coverage, rights-chain audit.

Based on US Copyright Act §115 mechanical license provisions and
standard music industry royalty structures. Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum


class LicenseType(Enum):
    MECHANICAL = "mechanical"
    PERFORMANCE = "performance"
    SYNCHRONIZATION = "synchronization"
    MASTER = "master"
    PRINT = "print"
    GRAND_RIGHTS = "grand_rights"


class UsageMedium(Enum):
    FILM = "film"
    TELEVISION = "television"
    STREAMING = "streaming"
    ADVERTISING = "advertising"
    VIDEO_GAME = "video_game"
    PODCAST = "podcast"


@dataclass
class RoyaltyShare:
    role: str
    share_pct: float
    is_writer: bool = True


@dataclass
class UsageRight:
    license_type: LicenseType
    territories: Set[str] = field(default_factory=lambda: {"WW"})
    exclusive: bool = False
    duration_years: float = 1.0
    medium: Optional[UsageMedium] = None


@dataclass
class SyncLicense:
    medium: UsageMedium
    duration_seconds: float
    territory_count: int = 1
    exclusivity: bool = False
    term_years: float = 1.0


@dataclass
class ClearanceResult:
    cleared: bool
    conflicts: List[str] = field(default_factory=list)
    missing_rights: List[LicenseType] = field(default_factory=list)
    total_cost_estimate: float = 0.0


def compute_royalty_split(writers: List[RoyaltyShare],
                          publishers: List[RoyaltyShare],
                          performers: List[RoyaltyShare]) -> Dict[str, float]:
    w_total = sum(s.share_pct for s in writers) or 1.0
    p_total = sum(s.share_pct for s in publishers) or 1.0
    perf_total = sum(s.share_pct for s in performers) or 1.0
    result = {}
    for s in writers:
        result[f"writer:{s.role}"] = 0.5 * (s.share_pct / w_total)
    for s in publishers:
        result[f"publisher:{s.role}"] = 0.5 * (s.share_pct / p_total)
    for s in performers:
        result[f"performer:{s.role}"] = s.share_pct / perf_total
    return result


def mechanical_rate(duration_seconds: float,
                    is_digital: bool = True,
                    statutory_rate: float = 0.091) -> float:
    minutes = duration_seconds / 60.0
    if minutes <= 5.0:
        return statutory_rate
    extra_minutes = minutes - 5.0
    return statutory_rate + extra_minutes * 0.0175


def sync_license_value(license: SyncLicense,
                       base_rate: float = 1000.0) -> float:
    medium_mult = {
        UsageMedium.FILM: 5.0,
        UsageMedium.TELEVISION: 3.0,
        UsageMedium.STREAMING: 2.0,
        UsageMedium.ADVERTISING: 8.0,
        UsageMedium.VIDEO_GAME: 4.0,
        UsageMedium.PODCAST: 1.0,
    }
    val = base_rate * medium_mult.get(license.medium, 1.0)
    val *= np.clip(license.duration_seconds / 30.0, 0.5, 10.0)
    val *= np.sqrt(max(license.territory_count, 1))
    if license.exclusivity:
        val *= 3.0
    val *= np.sqrt(max(license.term_years, 0.25))
    return float(val)


def clearance_check(required: List[LicenseType],
                    granted: List[UsageRight]) -> ClearanceResult:
    granted_types = {r.license_type for r in granted}
    missing = [lt for lt in required if lt not in granted_types]
    conflicts = []
    exclusive_map: Dict[LicenseType, List[UsageRight]] = {}
    for r in granted:
        if r.exclusive:
            exclusive_map.setdefault(r.license_type, []).append(r)
    for lt, excl_list in exclusive_map.items():
        if len(excl_list) > 1:
            conflicts.append(f"Multiple exclusive grants for {lt.value}")
    return ClearanceResult(
        cleared=len(missing) == 0 and len(conflicts) == 0,
        conflicts=conflicts,
        missing_rights=missing,
    )


def territory_coverage(grants: List[UsageRight],
                       target_territories: Set[str]) -> float:
    covered = set()
    for g in grants:
        if "WW" in g.territories:
            return 1.0
        covered |= g.territories
    if not target_territories:
        return 1.0
    return len(covered & target_territories) / len(target_territories)


def usage_conflict(rights: List[UsageRight]) -> List[Tuple[int, int, str]]:
    conflicts = []
    for i in range(len(rights)):
        for j in range(i + 1, len(rights)):
            a, b = rights[i], rights[j]
            if a.license_type != b.license_type:
                continue
            if not (a.exclusive or b.exclusive):
                continue
            overlap = a.territories & b.territories
            if "WW" in a.territories or "WW" in b.territories:
                overlap = a.territories | b.territories
            if overlap:
                conflicts.append((i, j,
                    f"Exclusive conflict: {a.license_type.value} in {overlap}"))
    return conflicts


def aggregate_royalties(streams: int, downloads: int,
                        sync_fees: float,
                        per_stream: float = 0.004,
                        per_download: float = 0.70) -> Dict[str, float]:
    streaming = streams * per_stream
    download_rev = downloads * per_download
    mech = downloads * mechanical_rate(210.0)
    return {
        "streaming_revenue": float(streaming),
        "download_revenue": float(download_rev),
        "mechanical_royalties": float(mech),
        "sync_fees": float(sync_fees),
        "total": float(streaming + download_rev + mech + sync_fees),
    }


def license_compatibility(a: UsageRight, b: UsageRight) -> float:
    if a.license_type != b.license_type:
        return 1.0
    if a.exclusive and b.exclusive:
        territory_overlap = a.territories & b.territories
        if "WW" in a.territories or "WW" in b.territories or territory_overlap:
            return 0.0
    if a.exclusive or b.exclusive:
        territory_overlap = a.territories & b.territories
        if "WW" in a.territories or "WW" in b.territories or territory_overlap:
            return 0.5
    return 1.0


def rights_chain_audit(chain: List[Dict[str, str]]) -> Dict[str, object]:
    issues = []
    seen_grantors = set()
    for i, link in enumerate(chain):
        grantor = link.get("grantor", "")
        grantee = link.get("grantee", "")
        if not grantor or not grantee:
            issues.append(f"Link {i}: missing grantor or grantee")
        if grantor in seen_grantors:
            issues.append(f"Link {i}: {grantor} appears as grantor multiple times")
        seen_grantors.add(grantor)
        if i > 0:
            prev_grantee = chain[i - 1].get("grantee", "")
            if grantor != prev_grantee:
                issues.append(
                    f"Link {i}: grantor '{grantor}' != prior grantee '{prev_grantee}'")
    return {
        "chain_length": len(chain),
        "valid": len(issues) == 0,
        "issues": issues,
    }
