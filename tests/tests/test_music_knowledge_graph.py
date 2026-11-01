"""Tests for music knowledge graph: entities, relations, traversal, inference."""
import pytest

from models.music_semantics.music_knowledge_graph import (
    EntityType, RelationType, Entity, Relation, MusicKnowledgeGraph,
)


class TestEntityType:
    def test_count(self):
        assert len(EntityType) == 13

    def test_includes_core_types(self):
        for t in ("artist", "track", "album", "genre", "instrument"):
            assert t in [e.value for e in EntityType]


class TestRelationType:
    def test_count(self):
        assert len(RelationType) == 15

    def test_includes_core_relations(self):
        for r in ("composed_by", "performed_by", "belongs_to_genre", "influences"):
            assert r in [rt.value for rt in RelationType]


class TestEntity:
    def test_frozen(self):
        e = Entity("a1", EntityType.ARTIST, "Bach")
        with pytest.raises(AttributeError):
            e.name = "Mozart"  # type: ignore[misc]

    def test_attributes_tuple(self):
        e = Entity("a1", EntityType.ARTIST, "Bach",
                    attributes=(("born", "1685"), ("died", "1750")))
        assert len(e.attributes) == 2


class TestRelation:
    def test_frozen(self):
        r = Relation("a1", "g1", RelationType.BELONGS_TO_GENRE)
        with pytest.raises(AttributeError):
            r.weight = 0.5  # type: ignore[misc]

    def test_default_weight(self):
        r = Relation("a1", "g1", RelationType.BELONGS_TO_GENRE)
        assert r.weight == 1.0


class TestMusicKnowledgeGraph:
    @pytest.fixture
    def graph(self):
        g = MusicKnowledgeGraph()
        g.add_entity(Entity("bach", EntityType.ARTIST, "J.S. Bach"))
        g.add_entity(Entity("mozart", EntityType.ARTIST, "W.A. Mozart"))
        g.add_entity(Entity("baroque", EntityType.GENRE, "Baroque"))
        g.add_entity(Entity("classical", EntityType.GENRE, "Classical"))
        g.add_entity(Entity("bwv846", EntityType.TRACK, "Prelude in C Major"))
        g.add_relation(Relation("bach", "baroque", RelationType.BELONGS_TO_GENRE))
        g.add_relation(Relation("mozart", "classical", RelationType.BELONGS_TO_GENRE))
        g.add_relation(Relation("bwv846", "bach", RelationType.COMPOSED_BY))
        g.add_relation(Relation("bach", "mozart", RelationType.INFLUENCES, weight=0.7))
        g.add_relation(Relation("baroque", "classical", RelationType.SUBGENRE_OF, weight=0.0))
        return g

    def test_entity_count(self, graph):
        assert len(graph.entities) == 5

    def test_relation_count(self, graph):
        assert len(graph.relations) == 5

    def test_neighbors_all(self, graph):
        nbrs = graph.neighbors("bach")
        assert len(nbrs) == 2

    def test_neighbors_filtered(self, graph):
        nbrs = graph.neighbors("bach", RelationType.BELONGS_TO_GENRE)
        assert len(nbrs) == 1
        assert nbrs[0][0] == "baroque"

    def test_neighbors_empty(self, graph):
        assert graph.neighbors("nonexistent") == []

    def test_path_exists_direct(self, graph):
        assert graph.path_exists("bwv846", "bach")

    def test_path_exists_indirect(self, graph):
        assert graph.path_exists("bwv846", "baroque")

    def test_path_not_exists(self, graph):
        assert not graph.path_exists("classical", "bwv846")

    def test_entities_of_type(self, graph):
        artists = graph.entities_of_type(EntityType.ARTIST)
        assert len(artists) == 2

    def test_genre_hierarchy(self, graph):
        h = graph.genre_hierarchy()
        assert "classical" in h
        assert "baroque" in h["classical"]

    def test_influence_chain(self, graph):
        chain = graph.influence_chain("bach")
        assert chain[0] == "bach"
        assert "mozart" in chain

    def test_count_by_type(self, graph):
        counts = graph.count_by_type()
        assert counts[EntityType.ARTIST] == 2
        assert counts[EntityType.GENRE] == 2
        assert counts[EntityType.TRACK] == 1

    def test_empty_graph(self):
        g = MusicKnowledgeGraph()
        assert g.entities_of_type(EntityType.ARTIST) == []
        assert g.count_by_type() == {}
