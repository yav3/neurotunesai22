"""Normalize and validate music metadata fields."""
from __future__ import annotations

import re
from typing import Any

_KEY_ALIASES: dict[str, str] = {
    "cb": "B", "c": "C", "c#": "C#", "db": "Db", "d": "D",
    "d#": "D#", "eb": "Eb", "e": "E", "fb": "E", "f": "F",
    "f#": "F#", "gb": "Gb", "g": "G", "g#": "G#", "ab": "Ab",
    "a": "A", "a#": "A#", "bb": "Bb", "b": "B",
}

_KEY_PATTERN = re.compile(
    r"^([A-Ga-g])([#b♯♭]?)\s*(maj(?:or)?|min(?:or)?|m|M)?$"
)

_SHARP_MAP = {"♯": "#", "♭": "b"}


def normalize_key_signature(key_str: str) -> str:
    """Normalize a key string to canonical form (e.g. 'eb minor' → 'Eb minor').

    Returns the input unchanged if it cannot be parsed.
    """
    s = key_str.strip()
    if not s:
        return s
    for symbol, ascii_char in _SHARP_MAP.items():
        s = s.replace(symbol, ascii_char)
    m = _KEY_PATTERN.match(s)
    if not m:
        return key_str
    root = m.group(1).upper()
    accidental = m.group(2) if m.group(2) else ""
    mode_raw = (m.group(3) or "").lower()
    if mode_raw in ("m", "min", "minor"):
        mode = "minor"
    elif mode_raw in ("", "maj", "major", "M"):
        mode = "major"
    else:
        mode = "major"
    return f"{root}{accidental} {mode}"


def normalize_tempo(bpm: float, confidence: float = 1.0) -> float:
    """Clamp and optionally halve/double tempo into the musical range [30, 300].

    If *confidence* < 0.5 and the tempo is an octave error candidate,
    move it towards the centre of [60, 180].
    """
    if bpm <= 0:
        return 0.0
    result = float(bpm)
    if confidence < 0.5:
        if result > 180:
            result /= 2.0
        elif result < 60:
            result *= 2.0
    return max(30.0, min(result, 300.0))


def normalize_duration(
    seconds: float | None = None,
    sr: int | None = None,
    n_samples: int | None = None,
) -> float:
    """Return a non-negative duration in seconds.

    Accepts either *seconds* directly or (*sr*, *n_samples*).
    """
    if seconds is not None:
        return max(0.0, float(seconds))
    if sr is not None and n_samples is not None and sr > 0:
        return max(0.0, float(n_samples) / float(sr))
    return 0.0


def validate_metadata_schema(
    metadata: dict[str, Any],
    required_fields: list[str],
) -> list[str]:
    """Return a list of missing or invalid required fields."""
    errors: list[str] = []
    for field in required_fields:
        if field not in metadata:
            errors.append(f"missing:{field}")
        elif metadata[field] is None:
            errors.append(f"null:{field}")
        elif isinstance(metadata[field], str) and not metadata[field].strip():
            errors.append(f"empty:{field}")
    return errors


def merge_metadata(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    conflict_strategy: str = "primary_wins",
) -> dict[str, Any]:
    """Merge two metadata dicts.

    *conflict_strategy*:
        ``'primary_wins'`` — keep primary on key conflict (default).
        ``'secondary_wins'`` — keep secondary on key conflict.
        ``'longest'`` — keep the longer string; for non-strings, primary wins.
    """
    merged: dict[str, Any] = {}
    all_keys = set(primary) | set(secondary)
    for k in all_keys:
        p_val = primary.get(k)
        s_val = secondary.get(k)
        if p_val is not None and s_val is None:
            merged[k] = p_val
        elif p_val is None and s_val is not None:
            merged[k] = s_val
        elif p_val is not None and s_val is not None:
            if conflict_strategy == "secondary_wins":
                merged[k] = s_val
            elif conflict_strategy == "longest":
                if isinstance(p_val, str) and isinstance(s_val, str):
                    merged[k] = p_val if len(p_val) >= len(s_val) else s_val
                else:
                    merged[k] = p_val
            else:
                merged[k] = p_val
    return merged
