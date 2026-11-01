"""MusicBrainz entity schema for music metadata modeling."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re

MBID = str  # UUID format


class MBEntity(Enum):
    ARTIST = "artist"
    RELEASE_GROUP = "release-group"
    RELEASE = "release"
    RECORDING = "recording"
    WORK = "work"
    LABEL = "label"
    AREA = "area"
    SERIES = "series"


class MBRelationType(Enum):
    PERFORMANCE = "performance"
    COMPOSITION = "composition"
    ARRANGEMENT = "arrangement"
    COVER = "cover"
    REMIX = "remix"
    SAMPLE = "sample"
    COMPILATION = "compilation"
    TRANSLATION = "translation"
    RECORDING = "recording"
    MIX = "mix"
    MASTERING = "mastering"
    PRODUCTION = "production"


@dataclass
class MBArtistCredit:
    name: str
    sort_name: str
    mbid: MBID
    join_phrase: str = ""


@dataclass
class MBRecording:
    mbid: MBID
    title: str
    artist_credit: list[MBArtistCredit]
    duration_ms: int | None = None
    isrcs: list[str] = field(default_factory=list)


@dataclass
class MBRelease:
    mbid: MBID
    title: str
    artist_credit: list[MBArtistCredit]
    date: str | None = None
    country: str | None = None
    barcode: str | None = None
    status: str = "Official"
    media: list[dict] = field(default_factory=list)


@dataclass
class MBWork:
    mbid: MBID
    title: str
    composers: list[MBArtistCredit]
    iswc: str | None = None
    language: str | None = None
    work_type: str | None = None


@dataclass
class MBRelation:
    type: MBRelationType
    source_entity: MBEntity
    source_mbid: MBID
    target_entity: MBEntity
    target_mbid: MBID
    attributes: list[str] = field(default_factory=list)


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

_MB_BASE = "https://musicbrainz.org"


def entity_url(entity_type: MBEntity, mbid: MBID) -> str:
    if not _UUID_RE.match(mbid):
        raise ValueError(f"invalid MBID: {mbid}")
    return f"{_MB_BASE}/{entity_type.value}/{mbid}"


def parse_artist_credit(credit_string: str) -> list[MBArtistCredit]:
    parts = re.split(r"\s+(?:feat\.|ft\.|&|and|with)\s+", credit_string, flags=re.IGNORECASE)
    result = []
    for i, name in enumerate(parts):
        name = name.strip()
        if not name:
            continue
        sort_parts = name.split()
        sort_name = ", ".join(reversed(sort_parts)) if len(sort_parts) > 1 else name
        join = "" if i == len(parts) - 1 else " feat. "
        result.append(MBArtistCredit(name=name, sort_name=sort_name, mbid="", join_phrase=join))
    return result


def format_artist_credit(credits: list[MBArtistCredit]) -> str:
    return "".join(c.name + c.join_phrase for c in credits)
