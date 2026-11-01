"""Audio file metadata extraction, normalization, and validation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class AudioMetadata:
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    track_number: Optional[int] = None
    disc_number: Optional[int] = None
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None
    bit_depth: Optional[int] = None
    channels: Optional[int] = None
    codec: Optional[str] = None
    bitrate_kbps: Optional[int] = None
    replay_gain_db: Optional[float] = None
    isrc: Optional[str] = None
    upc: Optional[str] = None
    musicbrainz_id: Optional[str] = None


@dataclass
class TagMapping:
    id3v2: str
    vorbis: str
    mp4: str
    ape: str


STANDARD_TAGS: dict[str, TagMapping] = {
    "title": TagMapping("TIT2", "TITLE", "\xa9nam", "Title"),
    "artist": TagMapping("TPE1", "ARTIST", "\xa9ART", "Artist"),
    "album": TagMapping("TALB", "ALBUM", "\xa9alb", "Album"),
    "year": TagMapping("TDRC", "DATE", "\xa9day", "Year"),
    "genre": TagMapping("TCON", "GENRE", "\xa9gen", "Genre"),
    "track": TagMapping("TRCK", "TRACKNUMBER", "trkn", "Track"),
    "disc": TagMapping("TPOS", "DISCNUMBER", "disk", "Disc"),
    "isrc": TagMapping("TSRC", "ISRC", "----:com.apple.iTunes:ISRC", "ISRC"),
    "upc": TagMapping("TXXX:UPC", "UPC", "----:com.apple.iTunes:UPC", "UPC"),
    "musicbrainz_recording_id": TagMapping(
        "TXXX:MusicBrainz Recording Id", "MUSICBRAINZ_TRACKID",
        "----:com.apple.iTunes:MusicBrainz Track Id", "MUSICBRAINZ_TRACKID",
    ),
    "replay_gain_track": TagMapping(
        "TXXX:REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_TRACK_GAIN",
        "----:com.apple.iTunes:replaygain_track_gain", "REPLAYGAIN_TRACK_GAIN",
    ),
    "replay_gain_album": TagMapping(
        "TXXX:REPLAYGAIN_ALBUM_GAIN", "REPLAYGAIN_ALBUM_GAIN",
        "----:com.apple.iTunes:replaygain_album_gain", "REPLAYGAIN_ALBUM_GAIN",
    ),
}


def _safe_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(str(val).split("/")[0].strip())
    except (ValueError, IndexError):
        return None


def _safe_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        cleaned = re.sub(r"[^\d.\-+eE]", "", str(val))
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def parse_id3v2_tags(tag_dict: dict[str, str]) -> AudioMetadata:
    return AudioMetadata(
        title=tag_dict.get("TIT2"),
        artist=tag_dict.get("TPE1"),
        album=tag_dict.get("TALB"),
        year=_safe_int(tag_dict.get("TDRC")),
        genre=tag_dict.get("TCON"),
        track_number=_safe_int(tag_dict.get("TRCK")),
        disc_number=_safe_int(tag_dict.get("TPOS")),
        isrc=tag_dict.get("TSRC"),
        replay_gain_db=_safe_float(tag_dict.get("TXXX:REPLAYGAIN_TRACK_GAIN")),
        musicbrainz_id=tag_dict.get("TXXX:MusicBrainz Recording Id"),
    )


def parse_vorbis_tags(tag_dict: dict[str, str]) -> AudioMetadata:
    return AudioMetadata(
        title=tag_dict.get("TITLE"),
        artist=tag_dict.get("ARTIST"),
        album=tag_dict.get("ALBUM"),
        year=_safe_int(tag_dict.get("DATE")),
        genre=tag_dict.get("GENRE"),
        track_number=_safe_int(tag_dict.get("TRACKNUMBER")),
        disc_number=_safe_int(tag_dict.get("DISCNUMBER")),
        isrc=tag_dict.get("ISRC"),
        replay_gain_db=_safe_float(tag_dict.get("REPLAYGAIN_TRACK_GAIN")),
        musicbrainz_id=tag_dict.get("MUSICBRAINZ_TRACKID"),
    )


def normalize_metadata(meta: AudioMetadata) -> AudioMetadata:
    title = meta.title.strip() if meta.title else None
    artist = meta.artist.strip() if meta.artist else None
    album = meta.album.strip() if meta.album else None
    genre = meta.genre
    if genre:
        parts = [g.strip() for g in re.split(r"[;/,]", genre)]
        seen: set[str] = set()
        deduped = []
        for g in parts:
            low = g.lower()
            if low and low not in seen:
                seen.add(low)
                deduped.append(g)
        genre = "; ".join(deduped) if deduped else None
    year = meta.year
    if year is not None and (year < 1000 or year > 2100):
        year = None
    return AudioMetadata(
        title=title, artist=artist, album=album, year=year, genre=genre,
        track_number=meta.track_number, disc_number=meta.disc_number,
        duration_s=meta.duration_s, sample_rate=meta.sample_rate,
        bit_depth=meta.bit_depth, channels=meta.channels,
        codec=meta.codec, bitrate_kbps=meta.bitrate_kbps,
        replay_gain_db=meta.replay_gain_db, isrc=meta.isrc,
        upc=meta.upc, musicbrainz_id=meta.musicbrainz_id,
    )


def metadata_completeness(meta: AudioMetadata) -> float:
    all_fields = fields(meta)
    filled = sum(1 for f in all_fields if getattr(meta, f.name) is not None)
    return filled / len(all_fields) if all_fields else 0.0


def merge_metadata(sources: list[AudioMetadata]) -> AudioMetadata:
    if not sources:
        return AudioMetadata()
    merged = AudioMetadata()
    for f in fields(merged):
        for src in sources:
            val = getattr(src, f.name)
            if val is not None:
                object.__setattr__(merged, f.name, val)
                break
    return merged


_ISRC_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}\d{2}\d{5}$")


def validate_isrc(isrc: str) -> bool:
    return bool(_ISRC_PATTERN.match(isrc.replace("-", "").replace(" ", "").upper()))


_ISWC_PATTERN = re.compile(r"^T\d{3}\.\d{3}\.\d{3}.\d$")


def validate_iswc(iswc: str) -> bool:
    cleaned = iswc.strip()
    if _ISWC_PATTERN.match(cleaned):
        return True
    no_sep = cleaned.replace("-", "").replace(".", "")
    return bool(re.match(r"^T\d{10}$", no_sep))
