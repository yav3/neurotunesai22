"""Parsers and adapters for open-source music datasets.

Supported formats:
  - Million Song Dataset (HDF5 summary → flat dict)
  - MusicBrainz metadata (JSON API response)
  - LAKH MIDI Dataset (MIDI file metadata extraction)
  - FMA (Free Music Archive) metadata CSV
  - Jamendo metadata CSV
  - AcousticBrainz features JSON
  - Common format: any CSV/JSON with recognized column names
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

from models.music_data.schema import TrackRecord


@dataclass
class DatasetMeta:
    name: str
    source_url: str
    licence: str
    record_count: int = 0
    languages: list[str] = field(default_factory=list)
    description: str = ""


KNOWN_DATASETS: dict[str, DatasetMeta] = {
    "million_song": DatasetMeta(
        name="Million Song Dataset",
        source_url="http://millionsongdataset.com/",
        licence="CC0 (metadata); audio via 7digital",
        description="1M track metadata + audio features from The Echo Nest",
    ),
    "fma": DatasetMeta(
        name="Free Music Archive",
        source_url="https://github.com/mdeff/fma",
        licence="CC-BY / CC-BY-NC / CC-BY-SA per track",
        description="106K tracks with genre labels and audio features",
    ),
    "musicbrainz": DatasetMeta(
        name="MusicBrainz",
        source_url="https://musicbrainz.org/",
        licence="CC0 (core data)",
        description="Open music encyclopedia with metadata for 30M+ recordings",
    ),
    "lakh_midi": DatasetMeta(
        name="LAKH MIDI Dataset",
        source_url="https://colinraffel.com/projects/lmd/",
        licence="Various (MIDI files from the web)",
        description="176K MIDI files matched to Million Song Dataset entries",
    ),
    "acousticbrainz": DatasetMeta(
        name="AcousticBrainz",
        source_url="https://acousticbrainz.org/",
        licence="CC0",
        description="Community-contributed audio features (Essentia-based)",
    ),
    "jamendo": DatasetMeta(
        name="Jamendo",
        source_url="https://www.jamendo.com/",
        licence="CC-BY / CC-BY-SA / CC-BY-NC per track",
        description="600K+ CC-licensed tracks with mood/instrument tags",
    ),
    "gtzan": DatasetMeta(
        name="GTZAN Genre Collection",
        source_url="http://marsyas.info/downloads/datasets.html",
        licence="Research use",
        description="1000 tracks, 10 genres, 30s clips — seminal genre benchmark",
    ),
    "magnatagatune": DatasetMeta(
        name="MagnaTagATune",
        source_url="https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset",
        licence="Research use",
        description="25.8K clips with 188 binary tags from human annotation",
    ),
    "musdb18": DatasetMeta(
        name="MUSDB18",
        source_url="https://sigsep.github.io/datasets/musdb.html",
        licence="Research use",
        description="150 full-length tracks with isolated stems (vocals/drums/bass/other)",
    ),
    "gutenberg_poetry": DatasetMeta(
        name="Project Gutenberg Poetry",
        source_url="https://www.gutenberg.org/",
        licence="Public domain",
        languages=["en", "fr", "de", "es", "it", "pt", "la", "el", "zh", "ja", "ar"],
        description="Public-domain poetry and song lyrics from 70K+ ebooks",
    ),
}


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", s.lower()).strip("_")


# ── Million Song Dataset adapter ────────────────────────────────────

_MSD_FIELD_MAP: dict[str, str] = {
    "track_id": "id",
    "song_id": "id",
    "title": "title",
    "artist_name": "artist",
    "release": "album",
    "duration": "duration_s",
    "tempo": "bpm",
    "key": "key_str",
    "mode": "mode",
    "key_confidence": "key_confidence",
    "time_signature": "time_signature",
    "loudness": "loudness_db",
    "energy": "energy",
    "danceability": "danceability",
    "speechiness": "speechiness",
    "acousticness": "acousticness",
    "instrumentalness": "instrumentalness",
    "liveness": "liveness",
    "valence": "valence",
    "artist_terms": "tags",
}

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse_msd_record(raw: dict[str, Any]) -> TrackRecord:
    mapped: dict[str, Any] = {}
    for src, dst in _MSD_FIELD_MAP.items():
        if src in raw and raw[src] is not None:
            mapped[dst] = raw[src]
    key_idx = raw.get("key")
    if isinstance(key_idx, (int, float)) and 0 <= int(key_idx) < 12:
        mapped["key_str"] = KEY_NAMES[int(key_idx)]
    mode_val = raw.get("mode")
    if isinstance(mode_val, (int, float)):
        mapped["mode"] = "major" if int(mode_val) == 1 else "minor"
    ts = raw.get("time_signature")
    if isinstance(ts, (int, float)):
        mapped["time_signature"] = f"{int(ts)}/4"
    mapped["source_bucket"] = "million_song"
    return TrackRecord.from_dict(mapped)


def parse_msd_csv(csv_text: str) -> list[TrackRecord]:
    reader = csv.DictReader(StringIO(csv_text))
    return [parse_msd_record(dict(row)) for row in reader]


# ── MusicBrainz adapter ─────────────────────────────────────────────

def parse_musicbrainz_recording(raw: dict[str, Any]) -> TrackRecord:
    d: dict[str, Any] = {
        "id": raw.get("id", ""),
        "title": raw.get("title", ""),
        "source_bucket": "musicbrainz",
    }
    if "length" in raw and raw["length"]:
        d["duration_s"] = float(raw["length"]) / 1000.0
    if "artist-credit" in raw:
        artists = raw["artist-credit"]
        if isinstance(artists, list) and artists:
            names = []
            for ac in artists:
                if isinstance(ac, dict) and "artist" in ac:
                    names.append(ac["artist"].get("name", ""))
                elif isinstance(ac, dict) and "name" in ac:
                    names.append(ac["name"])
            d["artist"] = "; ".join(n for n in names if n)
    releases = raw.get("releases") or raw.get("release-list", [])
    if isinstance(releases, list) and releases:
        d["album"] = releases[0].get("title", "")
    tags_raw = raw.get("tags") or raw.get("tag-list", [])
    if isinstance(tags_raw, list):
        d["tags"] = [t["name"] if isinstance(t, dict) else str(t) for t in tags_raw if t]
    return TrackRecord.from_dict(d)


def parse_musicbrainz_json(json_text: str) -> list[TrackRecord]:
    data = json.loads(json_text)
    recordings = []
    if isinstance(data, list):
        recordings = data
    elif isinstance(data, dict):
        recordings = data.get("recordings", data.get("recording-list", [data]))
    return [parse_musicbrainz_recording(r) for r in recordings if isinstance(r, dict)]


# ── FMA (Free Music Archive) adapter ────────────────────────────────

_FMA_FIELD_MAP: dict[str, str] = {
    "track_id": "id",
    "track_title": "title",
    "artist_name": "artist",
    "album_title": "album",
    "track_genre_top": "genre",
    "track_duration": "duration_s",
}


def parse_fma_record(raw: dict[str, Any]) -> TrackRecord:
    mapped: dict[str, Any] = {"source_bucket": "fma"}
    for src, dst in _FMA_FIELD_MAP.items():
        if src in raw and raw[src] is not None and raw[src] != "":
            mapped[dst] = raw[src]
    tags_raw = raw.get("track_genres")
    if tags_raw and isinstance(tags_raw, str):
        try:
            parsed = json.loads(tags_raw.replace("'", '"'))
            if isinstance(parsed, list):
                mapped["tags"] = [
                    g.get("genre_title", str(g)) if isinstance(g, dict) else str(g)
                    for g in parsed
                ]
        except (json.JSONDecodeError, ValueError):
            mapped["tags"] = [tags_raw]
    return TrackRecord.from_dict(mapped)


def parse_fma_csv(csv_text: str) -> list[TrackRecord]:
    reader = csv.DictReader(StringIO(csv_text))
    return [parse_fma_record(dict(row)) for row in reader]


# ── AcousticBrainz features adapter ─────────────────────────────────

def parse_acousticbrainz_features(raw: dict[str, Any]) -> TrackRecord:
    d: dict[str, Any] = {"source_bucket": "acousticbrainz"}
    meta = raw.get("metadata", {})
    if "tags" in meta:
        tags = meta["tags"]
        if isinstance(tags, dict):
            d["tags"] = list(tags.keys())
        elif isinstance(tags, list):
            d["tags"] = tags

    low = raw.get("lowlevel", {})
    d["spectral_centroid"] = low.get("spectral_centroid", {}).get("mean", 0.0)
    d["spectral_rolloff"] = low.get("spectral_rolloff", {}).get("mean", 0.0)
    d["loudness_db"] = low.get("average_loudness", -14.0)
    d["zero_crossing_rate"] = low.get("zerocrossingrate", {}).get("mean", 0.0)

    mfcc_raw = low.get("mfcc", {}).get("mean", [])
    if isinstance(mfcc_raw, list):
        d["mfcc_coefficients"] = [float(v) for v in mfcc_raw[:13]]

    rhythm = raw.get("rhythm", {})
    d["bpm"] = rhythm.get("bpm", 0.0)
    d["danceability"] = rhythm.get("danceability", 0.0)

    tonal = raw.get("tonal", {})
    key_info = tonal.get("key_edma", tonal.get("key_krumhansl", {}))
    if isinstance(key_info, dict):
        d["key_str"] = key_info.get("key", "")
        d["mode"] = key_info.get("scale", "")
        d["key_confidence"] = key_info.get("strength", 0.0)

    return TrackRecord.from_dict(d)


def parse_acousticbrainz_json(json_text: str) -> list[TrackRecord]:
    data = json.loads(json_text)
    if isinstance(data, list):
        return [parse_acousticbrainz_features(r) for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and "lowlevel" in data:
        return [parse_acousticbrainz_features(data)]
    return []


# ── Jamendo adapter ──────────────────────────────────────────────────

def parse_jamendo_record(raw: dict[str, Any]) -> TrackRecord:
    d: dict[str, Any] = {
        "id": str(raw.get("id", "")),
        "title": raw.get("name", raw.get("title", "")),
        "artist": raw.get("artist_name", ""),
        "album": raw.get("album_name", ""),
        "duration_s": float(raw.get("duration", 0)),
        "source_bucket": "jamendo",
    }
    tags = raw.get("musicinfo", {})
    if isinstance(tags, dict):
        tag_list = []
        for group in ["vocalinstrumental", "gender", "speed", "tags"]:
            val = tags.get(group)
            if isinstance(val, dict):
                tag_list.extend(val.get("nodes", []))
            elif isinstance(val, list):
                tag_list.extend(val)
            elif isinstance(val, str) and val:
                tag_list.append(val)
        d["tags"] = [t.get("id", str(t)) if isinstance(t, dict) else str(t) for t in tag_list if t]
    raw_tags = raw.get("tags") or raw.get("mood_tags")
    if raw_tags:
        if isinstance(raw_tags, list):
            d.setdefault("tags", []).extend(str(t) for t in raw_tags)
        elif isinstance(raw_tags, str):
            d.setdefault("tags", []).extend(raw_tags.split(","))
    return TrackRecord.from_dict(d)


def parse_jamendo_json(json_text: str) -> list[TrackRecord]:
    data = json.loads(json_text)
    if isinstance(data, list):
        return [parse_jamendo_record(r) for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        results = data.get("results", data.get("tracks", [data]))
        return [parse_jamendo_record(r) for r in results if isinstance(r, dict)]
    return []


# ── MIDI metadata adapter (for LAKH MIDI Dataset) ───────────────────

@dataclass
class MIDIMeta:
    filename: str = ""
    n_tracks: int = 0
    ticks_per_beat: int = 480
    tempo_bpm: float = 120.0
    time_signature: str = "4/4"
    key_str: str = ""
    duration_s: float = 0.0
    n_notes: int = 0
    pitch_range: tuple[int, int] = (0, 127)
    instruments: list[str] = field(default_factory=list)

    def to_track_record(self) -> TrackRecord:
        return TrackRecord(
            id=self.filename,
            path=self.filename,
            bpm=self.tempo_bpm,
            time_signature=self.time_signature,
            key_str=self.key_str,
            duration_s=self.duration_s,
            instrumentalness=1.0,
            source_bucket="lakh_midi",
            tags=self.instruments,
        )


GM_INSTRUMENT_NAMES: list[str] = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord",
    "Clavi", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
    "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ",
    "Rock Organ", "Church Organ", "Reed Organ", "Accordion", "Harmonica",
    "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)",
    "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)",
    "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings",
    "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensemble 1",
    "String Ensemble 2", "SynthStrings 1", "SynthStrings 2", "Choir Aahs",
    "Voice Oohs", "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba",
    "Muted Trumpet", "French Horn", "Brass Section", "SynthBrass 1", "SynthBrass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe",
    "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder",
    "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass+lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bag pipe", "Fiddle",
    "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]


def parse_midi_metadata(raw: dict[str, Any]) -> MIDIMeta:
    meta = MIDIMeta()
    meta.filename = str(raw.get("filename", raw.get("path", "")))
    meta.n_tracks = int(raw.get("n_tracks", raw.get("num_tracks", 0)))
    meta.ticks_per_beat = int(raw.get("ticks_per_beat", raw.get("resolution", 480)))

    tempo = raw.get("tempo_bpm", raw.get("tempo", raw.get("estimated_tempo", 120.0)))
    if isinstance(tempo, list):
        tempo = tempo[0] if tempo else 120.0
    meta.tempo_bpm = float(tempo)

    ts = raw.get("time_signature", "4/4")
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        meta.time_signature = f"{ts[0]}/{ts[1]}"
    else:
        meta.time_signature = str(ts)

    meta.key_str = str(raw.get("key", raw.get("estimated_key", "")))
    meta.duration_s = float(raw.get("duration", raw.get("duration_s", 0.0)))
    meta.n_notes = int(raw.get("n_notes", raw.get("total_notes", 0)))

    pr = raw.get("pitch_range", raw.get("note_range"))
    if isinstance(pr, (list, tuple)) and len(pr) >= 2:
        meta.pitch_range = (int(pr[0]), int(pr[1]))

    instruments = raw.get("instruments", raw.get("programs", []))
    if isinstance(instruments, list):
        resolved = []
        for inst in instruments:
            if isinstance(inst, int) and 0 <= inst < 128:
                resolved.append(GM_INSTRUMENT_NAMES[inst])
            elif isinstance(inst, str):
                resolved.append(inst)
        meta.instruments = resolved

    return meta


# ── Unified multi-source ingestion ───────────────────────────────────

def detect_source_format(raw: dict[str, Any] | str) -> str:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            if "track_id" in raw and "artist_name" in raw:
                return "fma_csv"
            if "song_id" in raw or "track_id" in raw:
                return "msd_csv"
            return "unknown"

    if not isinstance(raw, dict):
        return "unknown"

    if "lowlevel" in raw or "rhythm" in raw and "tonal" in raw:
        return "acousticbrainz"
    if "artist-credit" in raw or "recording-list" in raw:
        return "musicbrainz"
    if "musicinfo" in raw or ("name" in raw and "artist_name" in raw):
        return "jamendo"
    if "ticks_per_beat" in raw or "n_tracks" in raw or "resolution" in raw:
        return "lakh_midi"
    if "track_genre_top" in raw or "track_title" in raw:
        return "fma"
    if "song_id" in raw or ("tempo" in raw and "key" in raw and "loudness" in raw):
        return "msd"
    return "unknown"


def parse_record(raw: dict[str, Any], source: str | None = None) -> TrackRecord | None:
    if source is None:
        source = detect_source_format(raw)
    parsers = {
        "msd": parse_msd_record,
        "fma": parse_fma_record,
        "acousticbrainz": parse_acousticbrainz_features,
        "musicbrainz": parse_musicbrainz_recording,
        "jamendo": parse_jamendo_record,
    }
    parser = parsers.get(source)
    if parser is None:
        return TrackRecord.from_dict(raw)
    return parser(raw)


class MultiSourceIngester:
    def __init__(self) -> None:
        self.records: list[TrackRecord] = []
        self.errors: list[tuple[str, str]] = []
        self._seen_ids: set[str] = set()

    @property
    def count(self) -> int:
        return len(self.records)

    def ingest_json(self, json_text: str, source: str | None = None) -> int:
        added = 0
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            self.errors.append(("json_parse", str(e)))
            return 0

        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            src = source or detect_source_format(item)
            rec = parse_record(item, src)
            if rec and rec.id and rec.id not in self._seen_ids:
                self.records.append(rec)
                self._seen_ids.add(rec.id)
                added += 1
        return added

    def ingest_csv(self, csv_text: str, source: str | None = None) -> int:
        added = 0
        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            src = source or detect_source_format(dict(row))
            rec = parse_record(dict(row), src)
            if rec and rec.id and rec.id not in self._seen_ids:
                self.records.append(rec)
                self._seen_ids.add(rec.id)
                added += 1
        return added

    def ingest_file(self, path: str | Path, source: str | None = None) -> int:
        p = Path(path)
        if not p.exists():
            self.errors.append(("file_not_found", str(p)))
            return 0
        text = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix.lower() == ".json":
            return self.ingest_json(text, source)
        elif p.suffix.lower() == ".csv":
            return self.ingest_csv(text, source)
        else:
            self.errors.append(("unsupported_format", str(p.suffix)))
            return 0

    def deduplicate(self) -> int:
        seen: dict[str, int] = {}
        unique: list[TrackRecord] = []
        removed = 0
        for rec in self.records:
            key = f"{rec.title.lower().strip()}||{rec.artist.lower().strip()}"
            if key not in seen or key == "||":
                seen[key] = len(unique)
                unique.append(rec)
            else:
                removed += 1
        self.records = unique
        self._seen_ids = {r.id for r in unique if r.id}
        return removed

    def statistics(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        by_genre: dict[str, int] = {}
        has_bpm = has_key = has_vad = 0
        for rec in self.records:
            by_source[rec.source_bucket] = by_source.get(rec.source_bucket, 0) + 1
            if rec.genre:
                by_genre[rec.genre] = by_genre.get(rec.genre, 0) + 1
            if rec.bpm > 0:
                has_bpm += 1
            if rec.key_str:
                has_key += 1
            if rec.valence != 0 or rec.arousal != 0:
                has_vad += 1
        return {
            "total": len(self.records),
            "by_source": by_source,
            "by_genre": dict(sorted(by_genre.items(), key=lambda x: -x[1])[:20]),
            "coverage": {
                "bpm": has_bpm,
                "key": has_key,
                "vad": has_vad,
            },
            "errors": len(self.errors),
        }
