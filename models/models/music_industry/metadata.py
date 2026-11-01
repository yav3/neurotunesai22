"""Music metadata standards and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class TrackMetadata:
    title: str = ""
    artist: str = ""
    album: str = ""
    isrc: str = ""
    iswc: str = ""
    duration_ms: int = 0
    genre: str = ""
    subgenre: str = ""
    bpm: float = 0.0
    key: str = ""
    release_date: str = ""
    label: str = ""
    upc: str = ""


_REQUIRED = {"title", "artist", "duration_ms"}
_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def validate_metadata(meta: TrackMetadata) -> list[str]:
    errors = []
    if not meta.title.strip():
        errors.append("title is empty")
    if not meta.artist.strip():
        errors.append("artist is empty")
    if meta.duration_ms <= 0:
        errors.append("duration_ms must be positive")
    if meta.bpm < 0:
        errors.append("bpm must be non-negative")
    if meta.release_date and not _DATE_RE.match(meta.release_date):
        errors.append("release_date must be YYYY, YYYY-MM, or YYYY-MM-DD")
    return errors


def metadata_completeness(meta: TrackMetadata) -> float:
    total = 0
    filled = 0
    for f in fields(meta):
        total += 1
        val = getattr(meta, f.name)
        if isinstance(val, str) and val.strip():
            filled += 1
        elif isinstance(val, (int, float)) and val > 0:
            filled += 1
    return filled / max(total, 1)


def normalize_artist_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    parts = name.split()
    return " ".join(p.capitalize() for p in parts)


def detect_metadata_conflicts(
    meta_list: list[TrackMetadata],
) -> list[tuple[int, int, str]]:
    conflicts = []
    for i in range(len(meta_list)):
        for j in range(i + 1, len(meta_list)):
            a, b = meta_list[i], meta_list[j]
            if a.isrc and b.isrc and a.isrc == b.isrc:
                if a.title.lower() != b.title.lower():
                    conflicts.append((i, j, f"Same ISRC {a.isrc} but different titles"))
                if a.artist.lower() != b.artist.lower():
                    conflicts.append((i, j, f"Same ISRC {a.isrc} but different artists"))
    return conflicts
