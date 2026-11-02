"""Linked open data vocabularies and RDF-style triples for music metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Namespace(Enum):
    """Common music-metadata RDF namespaces."""
    MUSIC_ONTOLOGY = "mo"
    DUBLIN_CORE = "dc"
    SCHEMA_ORG = "schema"
    WIKIDATA = "wd"
    MUSICBRAINZ = "mb"
    AUDIO_FEATURES = "af"


@dataclass
class Triple:
    """An RDF-style (subject, predicate, object) triple."""
    subject: str
    predicate: str
    obj: str
    namespace: Namespace = Namespace.MUSIC_ONTOLOGY

    @property
    def uri(self) -> str:
        return f"{self.namespace.value}:{self.predicate}"


@dataclass
class LinkedRecord:
    """A music entity with linked-data triples."""
    entity_id: str
    entity_type: str
    triples: list[Triple] = field(default_factory=list)
    external_ids: dict[str, str] = field(default_factory=dict)

    @property
    def predicates(self) -> list[str]:
        return [t.predicate for t in self.triples]


def build_track_record(
    track_id: str,
    title: str,
    artist: str,
    duration_s: float,
    isrc: str | None = None,
    mbid: str | None = None,
) -> LinkedRecord:
    """Build a linked-data record for a track."""
    triples = [
        Triple(track_id, "title", title, Namespace.DUBLIN_CORE),
        Triple(track_id, "creator", artist, Namespace.DUBLIN_CORE),
        Triple(track_id, "duration", str(duration_s), Namespace.MUSIC_ONTOLOGY),
        Triple(track_id, "type", "Track", Namespace.SCHEMA_ORG),
    ]
    ext: dict[str, str] = {}
    if isrc:
        triples.append(Triple(track_id, "isrc", isrc, Namespace.MUSIC_ONTOLOGY))
        ext["isrc"] = isrc
    if mbid:
        triples.append(Triple(track_id, "musicbrainz_id", mbid, Namespace.MUSICBRAINZ))
        ext["musicbrainz"] = mbid
    return LinkedRecord(entity_id=track_id, entity_type="Track", triples=triples, external_ids=ext)


def build_artist_record(
    artist_id: str,
    name: str,
    genres: list[str] | None = None,
    wikidata_id: str | None = None,
) -> LinkedRecord:
    """Build a linked-data record for an artist."""
    triples = [
        Triple(artist_id, "name", name, Namespace.SCHEMA_ORG),
        Triple(artist_id, "type", "MusicArtist", Namespace.SCHEMA_ORG),
    ]
    for g in (genres or []):
        triples.append(Triple(artist_id, "genre", g, Namespace.MUSIC_ONTOLOGY))
    ext: dict[str, str] = {}
    if wikidata_id:
        triples.append(Triple(artist_id, "sameAs", wikidata_id, Namespace.WIKIDATA))
        ext["wikidata"] = wikidata_id
    return LinkedRecord(entity_id=artist_id, entity_type="Artist", triples=triples, external_ids=ext)


def resolve_same_as(records: list[LinkedRecord]) -> dict[str, list[str]]:
    """Find sameAs links across records, returning {ext_id: [entity_ids]}."""
    mapping: dict[str, list[str]] = {}
    for rec in records:
        for ext_key, ext_val in rec.external_ids.items():
            key = f"{ext_key}:{ext_val}"
            mapping.setdefault(key, []).append(rec.entity_id)
    return {k: v for k, v in mapping.items() if len(v) > 1}


def triple_count(record: LinkedRecord) -> int:
    """Count triples in a record."""
    return len(record.triples)


def filter_by_namespace(record: LinkedRecord, ns: Namespace) -> list[Triple]:
    """Return triples from a specific namespace."""
    return [t for t in record.triples if t.namespace == ns]


def merge_records(a: LinkedRecord, b: LinkedRecord) -> LinkedRecord:
    """Merge two records for the same entity (union of triples and external_ids)."""
    if a.entity_id != b.entity_id:
        raise ValueError("Cannot merge records with different entity_ids")
    seen = {(t.subject, t.predicate, t.obj) for t in a.triples}
    merged_triples = list(a.triples)
    for t in b.triples:
        key = (t.subject, t.predicate, t.obj)
        if key not in seen:
            merged_triples.append(t)
            seen.add(key)
    merged_ext = {**a.external_ids, **b.external_ids}
    return LinkedRecord(
        entity_id=a.entity_id,
        entity_type=a.entity_type,
        triples=merged_triples,
        external_ids=merged_ext,
    )
