"""Metadata tag schemas: ID3v2, Vorbis, APE, MP4, MusicBrainz."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class TagStandard(Enum):
    ID3V2 = "id3v2"
    VORBIS_COMMENT = "vorbis"
    APE = "ape"
    MP4_ATOM = "mp4"
    MUSICBRAINZ = "musicbrainz"


@dataclass
class TagField:
    name: str
    standard: TagStandard
    data_type: str  # "text", "integer", "date", "binary"
    max_length: int | None = None
    required: bool = False


ID3_FIELDS: list[TagField] = [
    TagField("TIT2", TagStandard.ID3V2, "text", 256, True),   # title
    TagField("TPE1", TagStandard.ID3V2, "text", 256, True),   # artist
    TagField("TALB", TagStandard.ID3V2, "text", 256),          # album
    TagField("TDRC", TagStandard.ID3V2, "date", 10),           # year
    TagField("TCON", TagStandard.ID3V2, "text", 128),          # genre
    TagField("TRCK", TagStandard.ID3V2, "text", 10),           # track
    TagField("TPOS", TagStandard.ID3V2, "text", 10),           # disc
    TagField("TBPM", TagStandard.ID3V2, "integer", 5),         # BPM
    TagField("TKEY", TagStandard.ID3V2, "text", 3),            # key
    TagField("TSRC", TagStandard.ID3V2, "text", 12),           # ISRC
    TagField("TCMP", TagStandard.ID3V2, "text", 1),            # compilation
    TagField("TPE2", TagStandard.ID3V2, "text", 256),          # album artist
    TagField("TCOM", TagStandard.ID3V2, "text", 256),          # composer
    TagField("TPUB", TagStandard.ID3V2, "text", 256),          # publisher
]

VORBIS_FIELDS: list[TagField] = [
    TagField("TITLE", TagStandard.VORBIS_COMMENT, "text", None, True),
    TagField("ARTIST", TagStandard.VORBIS_COMMENT, "text"),
    TagField("ALBUM", TagStandard.VORBIS_COMMENT, "text"),
    TagField("DATE", TagStandard.VORBIS_COMMENT, "date"),
    TagField("GENRE", TagStandard.VORBIS_COMMENT, "text"),
    TagField("TRACKNUMBER", TagStandard.VORBIS_COMMENT, "text"),
    TagField("DISCNUMBER", TagStandard.VORBIS_COMMENT, "text"),
    TagField("ISRC", TagStandard.VORBIS_COMMENT, "text", 12),
    TagField("BPM", TagStandard.VORBIS_COMMENT, "integer"),
    TagField("KEY", TagStandard.VORBIS_COMMENT, "text", 3),
    TagField("ALBUMARTIST", TagStandard.VORBIS_COMMENT, "text"),
    TagField("COMPOSER", TagStandard.VORBIS_COMMENT, "text"),
]

_FIELD_MAP: dict[tuple[str, TagStandard], str] = {
    ("TIT2", TagStandard.VORBIS_COMMENT): "TITLE",
    ("TITLE", TagStandard.ID3V2): "TIT2",
    ("TPE1", TagStandard.VORBIS_COMMENT): "ARTIST",
    ("ARTIST", TagStandard.ID3V2): "TPE1",
    ("TALB", TagStandard.VORBIS_COMMENT): "ALBUM",
    ("ALBUM", TagStandard.ID3V2): "TALB",
    ("TDRC", TagStandard.VORBIS_COMMENT): "DATE",
    ("DATE", TagStandard.ID3V2): "TDRC",
    ("TCON", TagStandard.VORBIS_COMMENT): "GENRE",
    ("GENRE", TagStandard.ID3V2): "TCON",
    ("TRCK", TagStandard.VORBIS_COMMENT): "TRACKNUMBER",
    ("TRACKNUMBER", TagStandard.ID3V2): "TRCK",
    ("TBPM", TagStandard.VORBIS_COMMENT): "BPM",
    ("BPM", TagStandard.ID3V2): "TBPM",
    ("TSRC", TagStandard.VORBIS_COMMENT): "ISRC",
    ("ISRC", TagStandard.ID3V2): "TSRC",
    ("TPE2", TagStandard.VORBIS_COMMENT): "ALBUMARTIST",
    ("ALBUMARTIST", TagStandard.ID3V2): "TPE2",
    ("TCOM", TagStandard.VORBIS_COMMENT): "COMPOSER",
    ("COMPOSER", TagStandard.ID3V2): "TCOM",
}


def normalize_field(field_name: str, source: TagStandard, target: TagStandard) -> str:
    if source == target:
        return field_name
    return _FIELD_MAP.get((field_name, target), field_name)


def validate_tag(field: TagField, value: str) -> tuple[bool, str | None]:
    if field.max_length and len(value) > field.max_length:
        return False, f"exceeds max length {field.max_length}"
    if field.data_type == "integer":
        if not value.strip().isdigit():
            return False, "expected integer"
    if field.data_type == "date":
        if not re.match(r"^\d{4}(-\d{2}(-\d{2})?)?$", value.strip()):
            return False, "expected YYYY or YYYY-MM-DD"
    if field.name == "TSRC" or field.name == "ISRC":
        if not re.match(r"^[A-Z]{2}[A-Z0-9]{3}\d{7}$", value.replace("-", "")):
            return False, "invalid ISRC format"
    return True, None
