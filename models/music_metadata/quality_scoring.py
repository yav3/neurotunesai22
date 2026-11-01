"""Metadata quality scoring: completeness, consistency, accuracy estimation.

Assesses track/album metadata quality across multiple dimensions without
external API calls — works entirely from the metadata fields themselves
and optional audio-derived features.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class QualityDimension(Enum):
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    FRESHNESS = "freshness"
    CONFORMANCE = "conformance"


# --- field importance weights (sum=1 within each tier) ---

CORE_FIELDS = {
    "title": 0.20, "artist": 0.20, "album": 0.15,
    "duration": 0.10, "genre": 0.10, "year": 0.10,
    "track_number": 0.05, "isrc": 0.05, "label": 0.05,
}

EXTENDED_FIELDS = {
    "composer": 0.12, "lyricist": 0.10, "producer": 0.10,
    "bpm": 0.08, "key": 0.08, "mood": 0.08,
    "language": 0.06, "iswc": 0.06, "barcode": 0.06,
    "album_artist": 0.06, "disc_number": 0.05,
    "copyright": 0.05, "musicbrainz_id": 0.05,
    "acoustid": 0.05,
}


@dataclass
class FieldScore:
    name: str
    present: bool
    weight: float
    penalty: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def effective(self) -> float:
        return self.weight * (1.0 - self.penalty) if self.present else 0.0


@dataclass
class QualityReport:
    completeness: float
    consistency: float
    accuracy: float
    freshness: float
    conformance: float
    overall: float
    field_scores: list[FieldScore]
    issues: list[str]

    def dimension(self, dim: QualityDimension) -> float:
        return getattr(self, dim.value)


# ---- completeness ----

def completeness_score(
    metadata: dict[str, Any],
    *,
    core_weight: float = 0.7,
    extended_weight: float = 0.3,
) -> tuple[float, list[FieldScore]]:
    """Weighted completeness from core + extended field presence."""
    scores: list[FieldScore] = []
    core_total = 0.0
    for f, w in CORE_FIELDS.items():
        present = _field_present(metadata, f)
        fs = FieldScore(name=f, present=present, weight=w)
        if present:
            core_total += w
        scores.append(fs)

    ext_total = 0.0
    for f, w in EXTENDED_FIELDS.items():
        present = _field_present(metadata, f)
        fs = FieldScore(name=f, present=present, weight=w)
        if present:
            ext_total += w
        scores.append(fs)

    combined = core_weight * core_total + extended_weight * ext_total
    return float(np.clip(combined, 0.0, 1.0)), scores


def _field_present(meta: dict[str, Any], key: str) -> bool:
    v = meta.get(key)
    if v is None:
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    return True


# ---- consistency ----

_TITLE_JUNK = re.compile(
    r"\b(remaster(ed)?|bonus\s*track|deluxe|explicit|clean|radio\s*edit"
    r"|single\s*version|album\s*version|live|demo|acoustic)\b",
    re.IGNORECASE,
)

_ARTIST_SEP = re.compile(r"\s*[,;&/]\s*|\s+feat\.?\s+|\s+ft\.?\s+|\s+and\s+|\s+vs\.?\s+", re.IGNORECASE)


def consistency_score(metadata: dict[str, Any]) -> tuple[float, list[str]]:
    """Check internal consistency of metadata fields.

    Returns (score ∈ [0,1], list of issues found).
    """
    issues: list[str] = []
    penalties = 0.0
    checks = 0

    title = metadata.get("title", "")
    artist = metadata.get("artist", "")
    album_artist = metadata.get("album_artist", "")

    # title should not embed artist name
    checks += 1
    if title and artist and artist.lower() in title.lower() and len(artist) > 3:
        issues.append("title embeds artist name")
        penalties += 1

    # title junk tags
    checks += 1
    if _TITLE_JUNK.search(title):
        issues.append("title contains version/remaster tags — should be in separate field")
        penalties += 0.5

    # artist vs album_artist consistency
    checks += 1
    if artist and album_artist:
        if artist.lower() == album_artist.lower():
            pass  # fine
        elif album_artist.lower() == "various artists":
            pass  # compilation
        else:
            parts = _ARTIST_SEP.split(artist)
            if not any(p.strip().lower() == album_artist.lower() for p in parts):
                issues.append("artist and album_artist have no overlap")
                penalties += 0.5

    # track number sanity
    checks += 1
    tn = metadata.get("track_number")
    if tn is not None:
        try:
            n = int(tn)
            if n < 1 or n > 999:
                issues.append(f"track_number {n} out of plausible range")
                penalties += 1
        except (ValueError, TypeError):
            issues.append("track_number not a valid integer")
            penalties += 1

    # year sanity
    checks += 1
    yr = metadata.get("year")
    if yr is not None:
        try:
            y = int(yr)
            if y < 1900 or y > 2100:
                issues.append(f"year {y} out of plausible range")
                penalties += 1
        except (ValueError, TypeError):
            issues.append("year not a valid integer")
            penalties += 1

    # duration sanity
    checks += 1
    dur = metadata.get("duration")
    if dur is not None:
        try:
            d = float(dur)
            if d <= 0:
                issues.append("duration is non-positive")
                penalties += 1
            elif d > 7200:
                issues.append("duration exceeds 2 hours — likely error")
                penalties += 0.5
        except (ValueError, TypeError):
            issues.append("duration not numeric")
            penalties += 1

    score = max(0.0, 1.0 - penalties / max(checks, 1))
    return score, issues


# ---- accuracy estimation ----

def accuracy_score(
    metadata: dict[str, Any],
    *,
    audio_bpm: float | None = None,
    audio_key: str | None = None,
    audio_duration: float | None = None,
) -> tuple[float, list[str]]:
    """Estimate accuracy by cross-checking metadata against audio-derived features.

    Returns (score ∈ [0,1], list of discrepancies).
    """
    issues: list[str] = []
    checks = 0
    matches = 0

    # BPM cross-check
    if audio_bpm is not None and "bpm" in metadata:
        checks += 1
        try:
            meta_bpm = float(metadata["bpm"])
            ratio = meta_bpm / audio_bpm if audio_bpm > 0 else 0
            # allow octave errors (half/double tempo) and ±5% tolerance
            if abs(ratio - 1.0) < 0.05 or abs(ratio - 2.0) < 0.05 or abs(ratio - 0.5) < 0.05:
                matches += 1
            else:
                issues.append(f"bpm mismatch: metadata={meta_bpm:.1f}, audio={audio_bpm:.1f}")
        except (ValueError, TypeError):
            issues.append("bpm field not numeric")

    # key cross-check
    if audio_key is not None and "key" in metadata:
        checks += 1
        mk = _normalize_key(str(metadata["key"]))
        ak = _normalize_key(audio_key)
        if mk == ak:
            matches += 1
        else:
            # check relative major/minor equivalence
            rel = _relative_key(ak)
            if mk == rel:
                matches += 0.5
                issues.append(f"key is relative major/minor: metadata={mk}, audio={ak}")
            else:
                issues.append(f"key mismatch: metadata={mk}, audio={ak}")

    # duration cross-check
    if audio_duration is not None and "duration" in metadata:
        checks += 1
        try:
            meta_dur = float(metadata["duration"])
            if audio_duration > 0 and abs(meta_dur - audio_duration) / audio_duration < 0.02:
                matches += 1
            else:
                issues.append(
                    f"duration mismatch: metadata={meta_dur:.1f}s, audio={audio_duration:.1f}s"
                )
        except (ValueError, TypeError):
            issues.append("duration field not numeric")

    # ISRC format check
    isrc = metadata.get("isrc")
    if isrc:
        checks += 1
        if re.match(r"^[A-Z]{2}[A-Z0-9]{3}\d{7}$", str(isrc).replace("-", "")):
            matches += 1
        else:
            issues.append(f"ISRC format invalid: {isrc}")

    score = matches / checks if checks > 0 else 1.0
    return float(np.clip(score, 0.0, 1.0)), issues


_KEY_MAP = {
    "c": "C", "c#": "C#", "db": "Db", "d": "D", "d#": "D#", "eb": "Eb",
    "e": "E", "f": "F", "f#": "F#", "gb": "Gb", "g": "G", "g#": "G#",
    "ab": "Ab", "a": "A", "a#": "A#", "bb": "Bb", "b": "B",
}

_RELATIVE_MINOR = {
    "C": "Am", "G": "Em", "D": "Bm", "A": "F#m", "E": "C#m", "B": "G#m",
    "F#": "D#m", "Gb": "Ebm", "Db": "Bbm", "Ab": "Fm", "Eb": "Cm",
    "Bb": "Gm", "F": "Dm",
}


def _normalize_key(k: str) -> str:
    k = k.strip()
    lower = k.lower().replace("minor", "m").replace("major", "").replace(" ", "")
    is_minor = lower.endswith("m")
    root = lower.rstrip("m")
    root = _KEY_MAP.get(root, root.capitalize())
    return root + ("m" if is_minor else "")


def _relative_key(k: str) -> str:
    if k.endswith("m"):
        root = k[:-1]
        for maj, mn in _RELATIVE_MINOR.items():
            if mn == k or mn[:-1] == root:
                return maj
        return k
    return _RELATIVE_MINOR.get(k, k)


# ---- freshness ----

def freshness_score(
    metadata: dict[str, Any],
    *,
    reference_year: int = 2026,
    schema_version: str | None = None,
) -> float:
    """Penalise stale metadata: old schema versions, missing modern fields."""
    score = 1.0

    yr = metadata.get("year")
    if yr is not None:
        try:
            age = reference_year - int(yr)
            if age > 50:
                score -= 0.1
        except (ValueError, TypeError):
            pass

    # missing modern identifiers
    if not metadata.get("isrc"):
        score -= 0.1
    if not metadata.get("musicbrainz_id") and not metadata.get("acoustid"):
        score -= 0.1

    # schema version penalty
    if schema_version and schema_version < "2.4":
        score -= 0.15

    return float(np.clip(score, 0.0, 1.0))


# ---- conformance ----

_GENRE_CANONICAL = {
    "hiphop": "hip-hop", "hip hop": "hip-hop", "r&b": "rnb",
    "rhythm and blues": "rnb", "drum and bass": "drum-and-bass",
    "dnb": "drum-and-bass", "electronic dance music": "edm",
    "electronica": "electronic",
}


def conformance_score(metadata: dict[str, Any]) -> tuple[float, list[str]]:
    """Check field values against canonical forms and controlled vocabularies.

    Returns (score ∈ [0,1], list of non-conformances).
    """
    issues: list[str] = []
    score = 1.0

    genre = metadata.get("genre", "")
    if genre:
        g_lower = genre.lower().strip()
        if g_lower in _GENRE_CANONICAL:
            issues.append(
                f"genre '{genre}' should be '{_GENRE_CANONICAL[g_lower]}'"
            )
            score -= 0.1

    # title case check
    title = metadata.get("title", "")
    if title and title == title.upper() and len(title) > 4:
        issues.append("title is ALL CAPS — should be title case")
        score -= 0.1

    artist = metadata.get("artist", "")
    if artist and artist == artist.upper() and len(artist) > 3:
        issues.append("artist is ALL CAPS")
        score -= 0.1

    # language code ISO 639
    lang = metadata.get("language", "")
    if lang and len(lang) not in (2, 3):
        issues.append(f"language '{lang}' not ISO 639-1/639-3 code")
        score -= 0.1

    return float(np.clip(score, 0.0, 1.0)), issues


# ---- overall quality ----

DIMENSION_WEIGHTS = {
    QualityDimension.COMPLETENESS: 0.30,
    QualityDimension.CONSISTENCY: 0.25,
    QualityDimension.ACCURACY: 0.20,
    QualityDimension.FRESHNESS: 0.10,
    QualityDimension.CONFORMANCE: 0.15,
}


def quality_report(
    metadata: dict[str, Any],
    *,
    audio_bpm: float | None = None,
    audio_key: str | None = None,
    audio_duration: float | None = None,
    reference_year: int = 2026,
    weights: dict[QualityDimension, float] | None = None,
) -> QualityReport:
    """Full multi-dimensional quality assessment."""
    w = weights or DIMENSION_WEIGHTS

    comp, field_scores = completeness_score(metadata)
    cons, cons_issues = consistency_score(metadata)
    acc, acc_issues = accuracy_score(
        metadata, audio_bpm=audio_bpm, audio_key=audio_key, audio_duration=audio_duration,
    )
    fresh = freshness_score(metadata, reference_year=reference_year)
    conf, conf_issues = conformance_score(metadata)

    all_issues = cons_issues + acc_issues + conf_issues

    overall = (
        w.get(QualityDimension.COMPLETENESS, 0.3) * comp
        + w.get(QualityDimension.CONSISTENCY, 0.25) * cons
        + w.get(QualityDimension.ACCURACY, 0.2) * acc
        + w.get(QualityDimension.FRESHNESS, 0.1) * fresh
        + w.get(QualityDimension.CONFORMANCE, 0.15) * conf
    )

    return QualityReport(
        completeness=comp,
        consistency=cons,
        accuracy=acc,
        freshness=fresh,
        conformance=conf,
        overall=float(np.clip(overall, 0.0, 1.0)),
        field_scores=field_scores,
        issues=all_issues,
    )


def batch_quality_scores(
    records: list[dict[str, Any]],
    **kwargs: Any,
) -> np.ndarray:
    """Compute overall quality for a batch of metadata records.

    Returns shape (n,) array of overall scores.
    """
    return np.array([quality_report(r, **kwargs).overall for r in records])


def quality_histogram(
    records: list[dict[str, Any]],
    bins: int = 10,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of quality scores across a catalogue.

    Returns (counts, bin_edges) like numpy.histogram.
    """
    scores = batch_quality_scores(records, **kwargs)
    return np.histogram(scores, bins=bins, range=(0.0, 1.0))
