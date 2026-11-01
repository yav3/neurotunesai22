"""Tests for music ontology knowledge graph."""
import numpy as np
import pytest

from models.music_semantics.music_ontology import (
    MusicOntology, MusicKnowledgeBase,
    OntologyNode, OntologyEdge,
    NodeType, RelationType,
)


def _small_graph():
    g = MusicOntology()
    g.add_node(OntologyNode("a", "A", NodeType.CONCEPT))
    g.add_node(OntologyNode("b", "B", NodeType.GENRE))
    g.add_node(OntologyNode("c", "C", NodeType.GENRE))
    g.add_node(OntologyNode("d", "D", NodeType.INSTRUMENT))
    g.add_edge(OntologyEdge("a", "b", RelationType.HAS_MEMBER))
    g.add_edge(OntologyEdge("a", "c", RelationType.HAS_MEMBER))
    g.add_edge(OntologyEdge("b", "d", RelationType.USED_IN))
    return g


class TestOntologyBasics:
    def test_add_node(self):
        g = MusicOntology()
        g.add_node(OntologyNode("x", "X", NodeType.CONCEPT))
        assert g.n_nodes == 1

    def test_add_edge(self):
        g = _small_graph()
        assert g.n_edges == 3

    def test_get_node(self):
        g = _small_graph()
        n = g.get_node("b")
        assert n is not None
        assert n.name == "B"

    def test_get_node_missing(self):
        g = _small_graph()
        assert g.get_node("zzz") is None

    def test_get_children(self):
        g = _small_graph()
        ch = g.get_children("a")
        ids = {n.id for n in ch}
        assert ids == {"b", "c"}

    def test_get_parents(self):
        g = _small_graph()
        p = g.get_parents("b")
        assert len(p) == 1 and p[0].id == "a"

    def test_get_related(self):
        g = _small_graph()
        rel = g.get_related("b", RelationType.USED_IN)
        assert len(rel) == 1 and rel[0].id == "d"

    def test_get_related_empty(self):
        g = _small_graph()
        rel = g.get_related("b", RelationType.CONTRASTS_WITH)
        assert rel == []


class TestGraphAlgorithms:
    def test_shortest_path_direct(self):
        g = _small_graph()
        path = g.shortest_path("a", "b")
        assert path == ["a", "b"]

    def test_shortest_path_two_hops(self):
        g = _small_graph()
        path = g.shortest_path("a", "d")
        assert path == ["a", "b", "d"]

    def test_shortest_path_self(self):
        g = _small_graph()
        assert g.shortest_path("a", "a") == ["a"]

    def test_shortest_path_unreachable(self):
        g = MusicOntology()
        g.add_node(OntologyNode("x", "X", NodeType.CONCEPT))
        g.add_node(OntologyNode("y", "Y", NodeType.CONCEPT))
        assert g.shortest_path("x", "y") is None

    def test_semantic_distance(self):
        g = _small_graph()
        d = g.semantic_distance("a", "b")
        assert d == pytest.approx(1.0)

    def test_semantic_distance_two_hops(self):
        g = _small_graph()
        d = g.semantic_distance("a", "d")
        assert d == pytest.approx(0.5)

    def test_semantic_distance_unreachable(self):
        g = MusicOntology()
        g.add_node(OntologyNode("x", "X", NodeType.CONCEPT))
        g.add_node(OntologyNode("y", "Y", NodeType.CONCEPT))
        assert g.semantic_distance("x", "y") == float("inf")

    def test_subgraph(self):
        g = _small_graph()
        sub = g.subgraph({"a", "b"})
        assert sub.n_nodes == 2
        assert sub.n_edges == 1

    def test_adjacency_matrix(self):
        g = _small_graph()
        mat, ids = g.to_adjacency_matrix()
        assert mat.shape == (4, 4)
        assert len(ids) == 4
        assert mat.sum() == 3.0

    def test_topological_sort(self):
        g = _small_graph()
        order = g.topological_sort()
        assert order[0] == "a"
        assert order.index("b") < order.index("d")

    def test_find_by_type(self):
        g = _small_graph()
        genres = g.find_by_type(NodeType.GENRE)
        assert len(genres) == 2

    def test_find_by_name(self):
        g = _small_graph()
        results = g.find_by_name("b")
        assert any(n.id == "b" for n in results)


class TestMusicKnowledgeBase:
    def test_build(self):
        g = MusicKnowledgeBase.build()
        assert g.n_nodes > 100
        assert g.n_edges > 80

    def test_has_instruments(self):
        g = MusicKnowledgeBase.build()
        instruments = g.find_by_type(NodeType.INSTRUMENT)
        assert len(instruments) >= 29

    def test_has_genres(self):
        g = MusicKnowledgeBase.build()
        genres = g.find_by_type(NodeType.GENRE)
        assert len(genres) >= 30

    def test_has_scales(self):
        g = MusicKnowledgeBase.build()
        scales = g.find_by_type(NodeType.SCALE)
        assert len(scales) >= 15

    def test_has_chords(self):
        g = MusicKnowledgeBase.build()
        chords = g.find_by_type(NodeType.CHORD)
        assert len(chords) >= 14

    def test_has_techniques(self):
        g = MusicKnowledgeBase.build()
        techs = g.find_by_type(NodeType.TECHNIQUE)
        assert len(techs) >= 15

    def test_cross_links_exist(self):
        g = MusicKnowledgeBase.build()
        rel = g.get_related("genre_blues", RelationType.INFLUENCES)
        ids = {n.id for n in rel}
        assert "genre_jazz" in ids or "genre_rock" in ids

    def test_path_across_domains(self):
        g = MusicKnowledgeBase.build()
        path = g.shortest_path("genre_blues", "scale_blues")
        assert path is not None
        assert len(path) <= 3
