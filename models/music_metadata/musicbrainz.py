"""MusicBrainz data models: artists, recordings, releases, works, relationships.

Implements the core MusicBrainz entity model with MBIDs, relationships,
and metadata resolution for music information retrieval and cataloging.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EntityType(Enum):
    ARTIST = "artist"
    RELEASE_GROUP = "release_group"
    RELEASE = "release"
    RECORDING = "recording"
    WORK = "work"
    LABEL = "label"
    AREA = "area"
    PLACE = "place"
    EVENT = "event"
    SERIES = "series"
    INSTRUMENT = "instrument"


class ArtistType(Enum):
    PERSON = "person"
    GROUP = "group"
    ORCHESTRA = "orchestra"
    CHOIR = "choir"
    CHARACTER = "character"
    OTHER = "other"


class ReleaseType(Enum):
    ALBUM = "album"
    SINGLE = "single"
    EP = "ep"
    COMPILATION = "compilation"
    SOUNDTRACK = "soundtrack"
    LIVE = "live"
    REMIX = "remix"
    DJ_MIX = "dj_mix"
    MIXTAPE = "mixtape"
    DEMO = "demo"
    OTHER = "other"


class RelationshipType(Enum):
    PERFORMER = "performer"
    COMPOSER = "composer"
    LYRICIST = "lyricist"
    ARRANGER = "arranger"
    PRODUCER = "producer"
    ENGINEER = "engineer"
    CONDUCTOR = "conductor"
    MIX = "mix"
    MASTER = "master"
    REMIX = "remix"
    COVER = "cover"
    SAMPLE = "sample"
    RECORDING_OF = "recording_of"
    PART_OF = "part_of"


MBID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def is_valid_mbid(mbid: str) -> bool:
    return bool(MBID_PATTERN.match(mbid))


def generate_deterministic_mbid(seed: str) -> str:
    """Generate a deterministic UUID-like MBID from a seed string."""
    h = hashlib.md5(seed.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


@dataclass
class MBEntity:
    mbid: str
    entity_type: EntityType
    name: str
    disambiguation: str = ""
    aliases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mbid and not is_valid_mbid(self.mbid):
            raise ValueError(f"Invalid MBID format: {self.mbid}")


@dataclass
class Artist(MBEntity):
    artist_type: ArtistType = ArtistType.PERSON
    sort_name: str = ""
    country: str = ""
    begin_date: str = ""
    end_date: str = ""
    ipi_codes: list[str] = field(default_factory=list)
    isni_codes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.entity_type = EntityType.ARTIST
        super().__post_init__()
        if not self.sort_name:
            self.sort_name = self.name


@dataclass
class Recording(MBEntity):
    duration_ms: int = 0
    isrc_codes: list[str] = field(default_factory=list)
    artist_credit: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.entity_type = EntityType.RECORDING
        super().__post_init__()

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class Work(MBEntity):
    work_type: str = ""
    language: str = ""
    iswc: str = ""
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entity_type = EntityType.WORK
        super().__post_init__()


@dataclass
class Release(MBEntity):
    release_type: ReleaseType = ReleaseType.ALBUM
    date: str = ""
    country: str = ""
    barcode: str = ""
    label: str = ""
    catalog_number: str = ""
    tracks: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.entity_type = EntityType.RELEASE
        super().__post_init__()


@dataclass(frozen=True)
class Relationship:
    source_mbid: str
    target_mbid: str
    relationship_type: RelationshipType
    source_type: EntityType
    target_type: EntityType
    attributes: tuple[str, ...] = ()
    begin_date: str = ""
    end_date: str = ""


class EntityGraph:
    """In-memory graph of MusicBrainz entities and relationships."""

    def __init__(self) -> None:
        self._entities: dict[str, MBEntity] = {}
        self._relationships: list[Relationship] = []
        self._adj: dict[str, list[int]] = {}

    def add_entity(self, entity: MBEntity) -> None:
        self._entities[entity.mbid] = entity
        if entity.mbid not in self._adj:
            self._adj[entity.mbid] = []

    def add_relationship(self, rel: Relationship) -> None:
        idx = len(self._relationships)
        self._relationships.append(rel)
        self._adj.setdefault(rel.source_mbid, []).append(idx)
        self._adj.setdefault(rel.target_mbid, []).append(idx)

    def get_entity(self, mbid: str) -> MBEntity | None:
        return self._entities.get(mbid)

    def get_relationships(self, mbid: str) -> list[Relationship]:
        indices = self._adj.get(mbid, [])
        return [self._relationships[i] for i in indices]

    def get_related_entities(
        self,
        mbid: str,
        rel_type: RelationshipType | None = None,
    ) -> list[MBEntity]:
        rels = self.get_relationships(mbid)
        results: list[MBEntity] = []
        for rel in rels:
            if rel_type and rel.relationship_type != rel_type:
                continue
            other_mbid = rel.target_mbid if rel.source_mbid == mbid else rel.source_mbid
            ent = self._entities.get(other_mbid)
            if ent:
                results.append(ent)
        return results

    @property
    def n_entities(self) -> int:
        return len(self._entities)

    @property
    def n_relationships(self) -> int:
        return len(self._relationships)

    def find_by_name(self, name: str, entity_type: EntityType | None = None) -> list[MBEntity]:
        name_lower = name.lower()
        results: list[MBEntity] = []
        for ent in self._entities.values():
            if entity_type and ent.entity_type != entity_type:
                continue
            if name_lower in ent.name.lower() or any(name_lower in a.lower() for a in ent.aliases):
                results.append(ent)
        return results
