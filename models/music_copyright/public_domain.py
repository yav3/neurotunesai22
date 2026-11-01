"""Public domain determination algorithms.

Date-based rules for US, EU, and Berne Convention jurisdictions.
Determines whether a musical work's copyright has expired based on
publication date, author death date, jurisdiction, and work type.
Pure Python (no external deps).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Jurisdiction(Enum):
    US = "us"
    EU = "eu"
    UK = "uk"
    CANADA = "canada"
    JAPAN = "japan"
    BERNE_DEFAULT = "berne_default"


class WorkType(Enum):
    IDENTIFIED_AUTHOR = "identified_author"
    ANONYMOUS = "anonymous"
    PSEUDONYMOUS = "pseudonymous"
    WORK_FOR_HIRE = "work_for_hire"
    JOINT = "joint"
    SOUND_RECORDING = "sound_recording"
    COMPOSITION = "composition"


@dataclass
class CopyrightQuery:
    publication_year: int | None = None
    creation_year: int | None = None
    author_death_year: int | None = None
    jurisdiction: Jurisdiction = Jurisdiction.US
    work_type: WorkType = WorkType.IDENTIFIED_AUTHOR
    registered_us: bool = False
    renewed_us: bool = True
    current_year: int = 2026


@dataclass
class CopyrightStatus:
    is_public_domain: bool
    expiration_year: int | None
    rule_applied: str
    confidence: float


# ── US rules ───────────────────────────────────────────────────────

def _us_status(q: CopyrightQuery) -> CopyrightStatus:
    pub = q.publication_year
    cur = q.current_year

    if pub is not None and pub <= 1928:
        return CopyrightStatus(True, pub, "us_pre_1929", 1.0)

    if pub is not None and 1929 <= pub <= 1977:
        if not q.registered_us:
            return CopyrightStatus(True, pub, "us_1929_1977_not_registered", 0.8)
        if not q.renewed_us and pub + 28 < cur:
            return CopyrightStatus(True, pub + 28, "us_1929_1977_not_renewed", 0.85)
        expiry = pub + 95
        return CopyrightStatus(expiry <= cur, expiry, "us_1929_1977_renewed_95yr", 0.95)

    if pub is not None and 1978 <= pub:
        if q.work_type in (WorkType.WORK_FOR_HIRE, WorkType.ANONYMOUS, WorkType.PSEUDONYMOUS):
            expiry = pub + 95
            creation = q.creation_year or pub
            alt = creation + 120
            expiry = min(expiry, alt)
            return CopyrightStatus(expiry <= cur, expiry, "us_post1978_wfh_95_or_120", 0.9)
        if q.author_death_year is not None:
            expiry = q.author_death_year + 70
            return CopyrightStatus(expiry <= cur, expiry, "us_post1978_life_plus_70", 0.95)
        return CopyrightStatus(False, None, "us_post1978_author_alive_or_unknown", 0.5)

    if pub is None and q.creation_year is not None:
        if q.author_death_year is not None:
            expiry = q.author_death_year + 70
            return CopyrightStatus(expiry <= cur, expiry, "us_unpublished_life_plus_70", 0.7)
        return CopyrightStatus(False, None, "us_unpublished_unknown_death", 0.3)

    return CopyrightStatus(False, None, "us_insufficient_data", 0.1)


# ── EU rules (harmonised) ─────────────────────────────────────────

def _eu_status(q: CopyrightQuery) -> CopyrightStatus:
    cur = q.current_year

    if q.work_type == WorkType.SOUND_RECORDING:
        pub = q.publication_year
        if pub is not None:
            expiry = pub + 70
            return CopyrightStatus(expiry <= cur, expiry, "eu_sound_recording_70yr", 0.9)
        return CopyrightStatus(False, None, "eu_sound_recording_no_pub_date", 0.3)

    if q.author_death_year is not None:
        if q.work_type == WorkType.JOINT and q.author_death_year is not None:
            expiry = q.author_death_year + 70
            return CopyrightStatus(expiry <= cur, expiry, "eu_joint_last_survivor_plus_70", 0.85)
        expiry = q.author_death_year + 70
        return CopyrightStatus(expiry <= cur, expiry, "eu_life_plus_70", 0.95)

    if q.work_type in (WorkType.ANONYMOUS, WorkType.PSEUDONYMOUS):
        pub = q.publication_year or q.creation_year
        if pub is not None:
            expiry = pub + 70
            return CopyrightStatus(expiry <= cur, expiry, "eu_anonymous_70yr_from_pub", 0.8)

    return CopyrightStatus(False, None, "eu_insufficient_data", 0.2)


# ── Berne Convention default ───────────────────────────────────────

def _berne_status(q: CopyrightQuery) -> CopyrightStatus:
    cur = q.current_year
    if q.author_death_year is not None:
        expiry = q.author_death_year + 50
        return CopyrightStatus(expiry <= cur, expiry, "berne_life_plus_50", 0.85)
    if q.work_type in (WorkType.ANONYMOUS, WorkType.PSEUDONYMOUS):
        pub = q.publication_year or q.creation_year
        if pub is not None:
            expiry = pub + 50
            return CopyrightStatus(expiry <= cur, expiry, "berne_anonymous_50yr", 0.7)
    return CopyrightStatus(False, None, "berne_insufficient_data", 0.2)


# ── Canada ─────────────────────────────────────────────────────────

def _canada_status(q: CopyrightQuery) -> CopyrightStatus:
    cur = q.current_year
    if q.author_death_year is not None:
        expiry = q.author_death_year + 70
        return CopyrightStatus(expiry <= cur, expiry, "canada_life_plus_70", 0.9)
    if q.work_type == WorkType.SOUND_RECORDING and q.publication_year is not None:
        expiry = q.publication_year + 70
        return CopyrightStatus(expiry <= cur, expiry, "canada_recording_70yr", 0.85)
    return CopyrightStatus(False, None, "canada_insufficient_data", 0.2)


# ── Japan ──────────────────────────────────────────────────────────

def _japan_status(q: CopyrightQuery) -> CopyrightStatus:
    cur = q.current_year
    if q.author_death_year is not None:
        expiry = q.author_death_year + 70
        return CopyrightStatus(expiry <= cur, expiry, "japan_life_plus_70", 0.9)
    if q.work_type in (WorkType.ANONYMOUS, WorkType.WORK_FOR_HIRE):
        pub = q.publication_year or q.creation_year
        if pub is not None:
            expiry = pub + 70
            return CopyrightStatus(expiry <= cur, expiry, "japan_anonymous_wfh_70yr", 0.8)
    return CopyrightStatus(False, None, "japan_insufficient_data", 0.2)


# ── Dispatcher ─────────────────────────────────────────────────────

_JURISDICTION_FN = {
    Jurisdiction.US: _us_status,
    Jurisdiction.EU: _eu_status,
    Jurisdiction.UK: _eu_status,
    Jurisdiction.CANADA: _canada_status,
    Jurisdiction.JAPAN: _japan_status,
    Jurisdiction.BERNE_DEFAULT: _berne_status,
}


def determine_copyright_status(query: CopyrightQuery) -> CopyrightStatus:
    """Determine whether a work is in the public domain."""
    fn = _JURISDICTION_FN.get(query.jurisdiction, _berne_status)
    return fn(query)


def is_public_domain(
    publication_year: int | None = None,
    author_death_year: int | None = None,
    jurisdiction: str = "us",
    work_type: str = "identified_author",
    current_year: int = 2026,
) -> bool:
    """Convenience: returns True if the work is public domain."""
    q = CopyrightQuery(
        publication_year=publication_year,
        author_death_year=author_death_year,
        jurisdiction=Jurisdiction(jurisdiction),
        work_type=WorkType(work_type),
        current_year=current_year,
    )
    return determine_copyright_status(q).is_public_domain
