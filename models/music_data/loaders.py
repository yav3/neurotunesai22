"""Loaders for music track data from CSV, JSON, and dict sources."""
from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from models.music_data.schema import TrackRecord


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _safe_list(val: Any) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("["):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                pass
        if val:
            return [v.strip() for v in val.split(",")]
    return []


def _safe_dict(val: Any) -> dict:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _row_to_track(row: dict[str, Any]) -> TrackRecord:
    return TrackRecord(
        id=str(row.get("id", "")),
        path=str(row.get("path", "")),
        title=str(row.get("title", "")),
        artist=str(row.get("artist", "")),
        album=str(row.get("album", "")),
        genre=str(row.get("genre", "")),
        duration_s=_safe_float(row.get("duration_s") or row.get("duration")),
        sr=_safe_int(row.get("sr"), 22050),
        bpm=_safe_float(row.get("bpm") or row.get("tempo_bpm")),
        key_str=str(row.get("key_str") or row.get("key", "")),
        camelot=str(row.get("camelot", "")),
        scale=str(row.get("scale", "")),
        mode=str(row.get("mode", "")),
        key_confidence=_safe_float(row.get("key_confidence")),
        time_signature=str(row.get("time_signature", "4/4")),
        valence=_safe_float(row.get("valence")),
        arousal=_safe_float(row.get("arousal")),
        dominance=_safe_float(row.get("dominance")),
        emotion_tags=_safe_list(row.get("emotion_tags")),
        mood_scores=_safe_dict(row.get("mood_scores")),
        spectral_centroid=_safe_float(row.get("spectral_centroid")),
        spectral_rolloff=_safe_float(row.get("spectral_rolloff")),
        spectral_bandwidth=_safe_float(row.get("spectral_bandwidth")),
        brightness=_safe_float(row.get("brightness")),
        zero_crossing_rate=_safe_float(row.get("zero_crossing_rate")),
        rms_energy=_safe_float(row.get("rms_energy")),
        dynamic_range=_safe_float(row.get("dynamic_range")),
        mfcc_coefficients=_safe_list(row.get("mfcc_coefficients")),
        bass_energy=_safe_float(row.get("bass_energy")),
        mid_energy=_safe_float(row.get("mid_energy")),
        treble_energy=_safe_float(row.get("treble_energy")),
        loudness_db=_safe_float(row.get("loudness_db"), -14.0),
        dynamic_complexity=_safe_float(row.get("dynamic_complexity")),
        attack_density=_safe_float(row.get("attack_density")),
        emotion_arc=[_safe_float(v) for v in _safe_list(row.get("emotion_arc"))],
        has_lyrics=bool(row.get("has_lyrics")),
        lyric_language=str(row.get("lyric_language", "")),
        lyric_sentiment=_safe_float(row.get("lyric_sentiment")),
        danceability=_safe_float(row.get("danceability")),
        energy=_safe_float(row.get("energy")),
        acousticness=_safe_float(row.get("acousticness")),
        instrumentalness=_safe_float(row.get("instrumentalness")),
        speechiness=_safe_float(row.get("speechiness")),
        liveness=_safe_float(row.get("liveness")),
        tags=_safe_list(row.get("tags")),
        checksum=str(row.get("checksum", "")),
        source_bucket=str(row.get("source_bucket", "")),
    )


class CSVTrackLoader:
    def load(self, path: str | Path) -> list[TrackRecord]:
        path = Path(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            return [_row_to_track(row) for row in reader]

    def loads(self, csv_text: str) -> list[TrackRecord]:
        reader = csv.DictReader(StringIO(csv_text))
        return [_row_to_track(row) for row in reader]


class JSONTrackLoader:
    def load(self, path: str | Path) -> list[TrackRecord]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [_row_to_track(row) for row in data]
        if isinstance(data, dict) and "tracks" in data:
            return [_row_to_track(row) for row in data["tracks"]]
        return []

    def loads(self, json_text: str) -> list[TrackRecord]:
        data = json.loads(json_text)
        if isinstance(data, list):
            return [_row_to_track(row) for row in data]
        if isinstance(data, dict) and "tracks" in data:
            return [_row_to_track(row) for row in data["tracks"]]
        return []
