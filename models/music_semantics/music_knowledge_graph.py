"""Music knowledge graph: entities, relations, and inference over structured music facts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class EntityType(Enum):
    ARTIST = "artist"
    TRACK = "track"
    ALBUM = "album"
    GENRE = "genre"
    INSTRUMENT = "instrument"
    KEY = "key"
    TIME_SIGNATURE = "time_signature"
    MOOD = "mood"
    ERA = "era"
    TECHNIQUE = "technique"
    FORM = "form"
    SCALE = "scale"
    CHORD_PROGRESSION = "chord_progression"


class RelationType(Enum):
    COMPOSED_BY = "composed_by"
    PERFORMED_BY = "performed_by"
    BELONGS_TO_GENRE = "belongs_to_genre"
    HAS_KEY = "has_key"
    USES_INSTRUMENT = "uses_instrument"
    INFLUENCES = "influences"
    SIMILAR_TO = "similar_to"
    CONTAINS_TRACK = "contains_track"
    HAS_MOOD = "has_mood"
    FROM_ERA = "from_era"
    USES_TECHNIQUE = "uses_technique"
    HAS_FORM = "has_form"
    USES_SCALE = "uses_scale"
    DERIVED_FROM = "derived_from"
    SUBGENRE_OF = "subgenre_of"


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: EntityType
    name: str
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Relation:
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    evidence: str = ""


@dataclass
class MusicKnowledgeGraph:
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    _adjacency: dict[str, list[tuple[str, RelationType, float]]] = field(
        default_factory=dict, repr=False
    )

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.entity_id] = entity

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)
        if relation.source_id not in self._adjacency:
            self._adjacency[relation.source_id] = []
        self._adjacency[relation.source_id].append(
            (relation.target_id, relation.relation_type, relation.weight)
        )

    def neighbors(self, entity_id: str,
                  relation_type: RelationType | None = None) -> list[tuple[str, float]]:
        if entity_id not in self._adjacency:
            return []
        results: list[tuple[str, float]] = []
        for target, rt, w in self._adjacency[entity_id]:
            if relation_type is None or rt == relation_type:
                results.append((target, w))
        return results

    def path_exists(self, source: str, target: str, max_depth: int = 5) -> bool:
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(source, 0)]
        while queue:
            current, depth = queue.pop(0)
            if current == target:
                return True
            if depth >= max_depth or current in visited:
                continue
            visited.add(current)
            for neighbor, _ in self.neighbors(current):
                queue.append((neighbor, depth + 1))
        return False

    def entities_of_type(self, entity_type: EntityType) -> list[Entity]:
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def genre_hierarchy(self) -> dict[str, list[str]]:
        hierarchy: dict[str, list[str]] = {}
        for rel in self.relations:
            if rel.relation_type == RelationType.SUBGENRE_OF:
                parent = rel.target_id
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(rel.source_id)
        return hierarchy

    def influence_chain(self, artist_id: str, max_depth: int = 5) -> list[str]:
        chain: list[str] = []
        visited: set[str] = set()
        current = artist_id
        for _ in range(max_depth):
            if current in visited:
                break
            visited.add(current)
            chain.append(current)
            influences = self.neighbors(current, RelationType.INFLUENCES)
            if not influences:
                break
            current = max(influences, key=lambda x: x[1])[0]
        return chain

    def count_by_type(self) -> dict[EntityType, int]:
        counts: dict[EntityType, int] = {}
        for e in self.entities.values():
            counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
        return counts
