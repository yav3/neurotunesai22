"""Parsers for open-source lyric and poetry corpus formats.

Supported:
  - Project Gutenberg plain-text poems (stanza-separated)
  - LRC timestamped lyrics (.lrc files)
  - SubRip subtitles (.srt — for subtitle-sourced lyrics)
  - Plain text lyrics (with optional section markers like [Verse], [Chorus])
"""
from __future__ import annotations

import re
from typing import Any

from models.music_lyrics.schema import LyricRecord, PoemRecord


def parse_gutenberg_poem(
    text: str,
    title: str = "",
    author: str = "",
    language: str = "en",
) -> PoemRecord:
    clean = text.strip()
    lines = clean.split("\n")
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("***") and "START" in stripped.upper():
            header_end = i + 1
            break

    footer_start = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("***") and "END" in stripped.upper():
            footer_start = i
            break

    body = "\n".join(lines[header_end:footer_start]).strip()

    if not title:
        for line in lines[header_end:min(header_end + 5, len(lines))]:
            stripped = line.strip()
            if stripped and len(stripped) < 100 and not stripped.startswith("by "):
                title = stripped
                break

    if not author:
        for line in lines[header_end:min(header_end + 10, len(lines))]:
            stripped = line.strip().lower()
            if stripped.startswith("by "):
                author = line.strip()[3:].strip()
                break

    return PoemRecord(
        text=body,
        title=title,
        author=author,
        language=language,
        source="gutenberg",
        licence="public domain",
    )


_LRC_TIMESTAMP = re.compile(r"\[(\d{1,3}):(\d{2})(?:\.(\d{1,3}))?\]")


def parse_lrc_file(text: str, title: str = "", artist: str = "") -> LyricRecord:
    lines = text.strip().split("\n")
    timestamps: list[tuple[float, str]] = []
    metadata: dict[str, str] = {}

    for line in lines:
        stripped = line.strip()
        meta_match = re.match(r"\[(ti|ar|al|by|offset):(.+)\]", stripped, re.I)
        if meta_match:
            key = meta_match.group(1).lower()
            val = meta_match.group(2).strip()
            metadata[key] = val
            continue

        matches = list(_LRC_TIMESTAMP.finditer(stripped))
        if matches:
            lyric_text = _LRC_TIMESTAMP.sub("", stripped).strip()
            for m in matches:
                minutes = int(m.group(1))
                seconds = int(m.group(2))
                centiseconds = int(m.group(3) or "0")
                t = minutes * 60 + seconds + centiseconds / 100.0
                timestamps.append((t, lyric_text))

    timestamps.sort(key=lambda x: x[0])
    full_text = "\n".join(t[1] for t in timestamps if t[1])

    return LyricRecord(
        text=full_text,
        title=title or metadata.get("ti", ""),
        artist=artist or metadata.get("ar", ""),
        album=metadata.get("al", ""),
        timestamps=timestamps,
        source="lrc",
    )


_SRT_INDEX = re.compile(r"^\d+$")
_SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt_subtitle(text: str, title: str = "", artist: str = "") -> LyricRecord:
    blocks = re.split(r"\n\s*\n", text.strip())
    timestamps: list[tuple[float, str]] = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        time_line = None
        text_lines = []
        for line in lines:
            if _SRT_INDEX.match(line.strip()):
                continue
            m = _SRT_TIME.match(line.strip())
            if m:
                h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                time_line = h * 3600 + mi * 60 + s + ms / 1000.0
            elif line.strip():
                clean = re.sub(r"<[^>]+>", "", line.strip())
                if clean:
                    text_lines.append(clean)

        if time_line is not None and text_lines:
            combined = " ".join(text_lines)
            timestamps.append((time_line, combined))

    timestamps.sort(key=lambda x: x[0])
    full_text = "\n".join(t[1] for t in timestamps if t[1])

    return LyricRecord(
        text=full_text,
        title=title,
        artist=artist,
        timestamps=timestamps,
        source="srt",
    )


_SECTION_MARKER = re.compile(r"^\[(verse|chorus|bridge|hook|intro|outro|pre-chorus|refrain|interlude)(?:\s*\d*)?\]", re.I)


def parse_plain_lyrics(
    text: str,
    title: str = "",
    artist: str = "",
    album: str = "",
    genre: str = "",
    language: str = "en",
) -> LyricRecord:
    lines = text.strip().split("\n")
    sections: list[dict[str, Any]] = []
    current_section = "verse"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        m = _SECTION_MARKER.match(stripped)
        if m:
            if current_lines:
                sections.append({
                    "type": current_section,
                    "text": "\n".join(current_lines),
                })
                current_lines = []
            current_section = m.group(1).lower()
            continue
        if stripped:
            current_lines.append(stripped)
        elif current_lines:
            sections.append({
                "type": current_section,
                "text": "\n".join(current_lines),
            })
            current_lines = []
            current_section = "verse"

    if current_lines:
        sections.append({
            "type": current_section,
            "text": "\n".join(current_lines),
        })

    full_text = "\n".join(l.strip() for l in lines if l.strip() and not _SECTION_MARKER.match(l.strip()))

    return LyricRecord(
        text=full_text,
        title=title,
        artist=artist,
        album=album,
        genre=genre,
        language=language,
        sections=sections,
        source="plain",
    )
