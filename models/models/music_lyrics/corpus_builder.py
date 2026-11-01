"""Corpus builder: aggregate lyrics/poetry records, deduplicate, export."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from models.music_lyrics.schema import LyricRecord, PoemRecord, CorpusStats
from models.music_lyrics.tokenizer import detect_language, normalize_text
from models.music_lyrics.sentiment import document_vad
from models.music_lyrics.rhyme import detect_rhyme_scheme
from models.music_lyrics.meter import detect_meter


class CorpusBuilder:
    def __init__(self) -> None:
        self.lyrics: list[LyricRecord] = []
        self.poems: list[PoemRecord] = []
        self._seen_hashes: set[str] = set()

    @property
    def total(self) -> int:
        return len(self.lyrics) + len(self.poems)

    def _text_hash(self, text: str) -> str:
        normed = normalize_text(text, strip_diacritics=True, lowercase=True)
        normed = " ".join(normed.split())
        return hashlib.sha256(normed.encode("utf-8")).hexdigest()[:16]

    def _is_duplicate(self, text: str) -> bool:
        h = self._text_hash(text)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def add_lyric(self, record: LyricRecord, enrich: bool = True) -> bool:
        if not record.text or len(record.text.strip()) < 10:
            return False
        if self._is_duplicate(record.text):
            return False
        if enrich:
            self._enrich_lyric(record)
        self.lyrics.append(record)
        return True

    def add_poem(self, record: PoemRecord, enrich: bool = True) -> bool:
        if not record.text or len(record.text.strip()) < 10:
            return False
        if self._is_duplicate(record.text):
            return False
        if enrich:
            self._enrich_poem(record)
        self.poems.append(record)
        return True

    def _enrich_lyric(self, rec: LyricRecord) -> None:
        if not rec.language or rec.language == "en":
            rec.language = detect_language(rec.text)
        if rec.vad_valence == 0 and rec.vad_arousal == 0:
            v, a, d = document_vad(rec.text)
            rec.vad_valence = v
            rec.vad_arousal = a
            rec.vad_dominance = d
        if not rec.rhyme_scheme:
            rec.rhyme_scheme = detect_rhyme_scheme(rec.text)
        if not rec.detected_meter:
            rec.detected_meter = detect_meter(rec.text)

    def _enrich_poem(self, rec: PoemRecord) -> None:
        if not rec.language or rec.language == "en":
            rec.language = detect_language(rec.text)
        if rec.vad_valence == 0 and rec.vad_arousal == 0:
            v, a, d = document_vad(rec.text)
            rec.vad_valence = v
            rec.vad_arousal = a
            rec.vad_dominance = d
        if not rec.rhyme_scheme:
            rec.rhyme_scheme = detect_rhyme_scheme(rec.text)
        if not rec.detected_meter:
            rec.detected_meter = detect_meter(rec.text)

    def add_from_records(self, records: list[dict[str, Any]], record_type: str = "lyric") -> int:
        added = 0
        for raw in records:
            text = raw.get("text", raw.get("lyrics", raw.get("content", "")))
            if not text:
                continue
            if record_type == "poem":
                rec = PoemRecord(
                    text=text,
                    title=raw.get("title", ""),
                    author=raw.get("author", raw.get("artist", "")),
                    language=raw.get("language", "en"),
                    source=raw.get("source", ""),
                    licence=raw.get("licence", raw.get("license", "")),
                    form_type=raw.get("form_type", raw.get("form", "")),
                    era=raw.get("era", ""),
                    tradition=raw.get("tradition", ""),
                )
                if self.add_poem(rec):
                    added += 1
            else:
                rec = LyricRecord(
                    text=text,
                    title=raw.get("title", ""),
                    artist=raw.get("artist", ""),
                    album=raw.get("album", ""),
                    genre=raw.get("genre", ""),
                    language=raw.get("language", "en"),
                    source=raw.get("source", ""),
                    licence=raw.get("licence", raw.get("license", "")),
                )
                if self.add_lyric(rec):
                    added += 1
        return added

    def deduplicate(self) -> int:
        self._seen_hashes.clear()
        removed = 0

        unique_lyrics: list[LyricRecord] = []
        for rec in self.lyrics:
            if not self._is_duplicate(rec.text):
                unique_lyrics.append(rec)
            else:
                removed += 1
        self.lyrics = unique_lyrics

        unique_poems: list[PoemRecord] = []
        for rec in self.poems:
            if not self._is_duplicate(rec.text):
                unique_poems.append(rec)
            else:
                removed += 1
        self.poems = unique_poems

        return removed

    def filter_by_language(self, lang: str) -> tuple[list[LyricRecord], list[PoemRecord]]:
        return (
            [r for r in self.lyrics if r.language == lang],
            [r for r in self.poems if r.language == lang],
        )

    def filter_by_source(self, source: str) -> tuple[list[LyricRecord], list[PoemRecord]]:
        return (
            [r for r in self.lyrics if r.source == source],
            [r for r in self.poems if r.source == source],
        )

    def statistics(self) -> CorpusStats:
        stats = CorpusStats()
        stats.total_records = self.total

        by_lang: dict[str, int] = {}
        by_source: dict[str, int] = {}
        by_genre: dict[str, int] = {}

        for rec in self.lyrics:
            by_lang[rec.language] = by_lang.get(rec.language, 0) + 1
            if rec.source:
                by_source[rec.source] = by_source.get(rec.source, 0) + 1
            if rec.genre:
                by_genre[rec.genre] = by_genre.get(rec.genre, 0) + 1
            stats.total_words += rec.word_count
            stats.total_lines += rec.line_count

        for rec in self.poems:
            by_lang[rec.language] = by_lang.get(rec.language, 0) + 1
            if rec.source:
                by_source[rec.source] = by_source.get(rec.source, 0) + 1
            stats.total_words += rec.word_count
            stats.total_lines += rec.line_count

        stats.by_language = by_lang
        stats.by_source = by_source
        stats.by_genre = by_genre
        stats.languages_count = len(by_lang)
        return stats

    def export_jsonl(self, path: str | Path) -> int:
        p = Path(path)
        count = 0
        with open(p, "w", encoding="utf-8") as f:
            for rec in self.lyrics:
                line = {
                    "type": "lyric",
                    "id": rec.id,
                    "text": rec.text,
                    "language": rec.language,
                    "title": rec.title,
                    "artist": rec.artist,
                    "genre": rec.genre,
                    "source": rec.source,
                    "vad": [rec.vad_valence, rec.vad_arousal, rec.vad_dominance],
                    "rhyme_scheme": rec.rhyme_scheme,
                    "meter": rec.detected_meter,
                    "word_count": rec.word_count,
                    "line_count": rec.line_count,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
            for rec in self.poems:
                line = {
                    "type": "poem",
                    "id": rec.id,
                    "text": rec.text,
                    "language": rec.language,
                    "title": rec.title,
                    "author": rec.author,
                    "source": rec.source,
                    "form_type": rec.form_type,
                    "era": rec.era,
                    "tradition": rec.tradition,
                    "vad": [rec.vad_valence, rec.vad_arousal, rec.vad_dominance],
                    "rhyme_scheme": rec.rhyme_scheme,
                    "meter": rec.detected_meter,
                    "word_count": rec.word_count,
                    "line_count": rec.line_count,
                    "stanza_count": rec.stanza_count,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
        return count
