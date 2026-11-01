"""Tests for visualization modules: tonnetz, circular, form diagram."""
import numpy as np
import pytest
from models.music_visualization.tonnetz_viz import (
    TonnetzPoint, TonnetzEdge, ChordRegion,
    tonnetz_coordinates, chord_tonnetz_centroid,
    chord_trajectory, tonnetz_distance, tonnetz_grid,
    neo_riemannian_edges, harmonic_function_color, region_data,
)
from models.music_visualization.circular_viz import (
    CirclePoint, ArcPath, FIFTHS_ORDER,
    circle_of_fifths_layout, chromatic_circle_layout,
    pitch_class_distribution, key_region_highlight,
    chord_progression_arcs, interval_vector_radial,
    fifths_distance, angular_distance,
)
from models.music_visualization.form_diagram import (
    FormSection, FormTree, COMMON_FORMS,
    section_boundaries, build_form_tree,
    repetition_matrix, novelty_curve,
    section_similarity_matrix, classify_form,
)


class TestTonnetzViz:
    def test_tonnetz_coordinates_c(self):
        p = tonnetz_coordinates(0)
        assert isinstance(p, TonnetzPoint)
        assert p.x == pytest.approx(0.0)
        assert p.y == pytest.approx(0.0)

    def test_chord_tonnetz_centroid(self):
        c = chord_tonnetz_centroid([0, 4, 7])
        assert isinstance(c, TonnetzPoint)

    def test_chord_trajectory(self):
        chords = [[0, 4, 7], [5, 9, 0], [7, 11, 2]]
        traj = chord_trajectory(chords)
        assert len(traj) == 3

    def test_tonnetz_distance(self):
        d = tonnetz_distance(0, 7)
        assert d > 0.0

    def test_tonnetz_grid(self):
        grid = tonnetz_grid(3, 4)
        assert len(grid) == 3 * 4

    def test_neo_riemannian_edges(self):
        edges = neo_riemannian_edges()
        assert len(edges) > 0
        assert all(isinstance(e, TonnetzEdge) for e in edges)

    def test_harmonic_function_color(self):
        c = harmonic_function_color("tonic")
        assert isinstance(c, str)

    def test_region_data(self):
        rd = region_data([0, 4, 7], "C major")
        assert isinstance(rd, ChordRegion)


class TestCircularViz:
    def test_fifths_order(self):
        assert len(FIFTHS_ORDER) == 12
        assert FIFTHS_ORDER[0] == 0

    def test_circle_of_fifths_layout(self):
        points = circle_of_fifths_layout()
        assert len(points) == 12
        assert all(isinstance(p, CirclePoint) for p in points)

    def test_chromatic_circle_layout(self):
        points = chromatic_circle_layout()
        assert len(points) == 12

    def test_pitch_class_distribution(self):
        pcs = [0, 0, 4, 7, 0, 4]
        dist = pitch_class_distribution(pcs)
        assert len(dist) == 12
        assert dist[0] == pytest.approx(3 / 6)

    def test_key_region_highlight(self):
        region = key_region_highlight(0, "major")
        assert isinstance(region, list)

    def test_chord_progression_arcs(self):
        chords = [[0, 4, 7], [5, 9, 0], [7, 11, 2]]
        arcs = chord_progression_arcs(chords)
        assert len(arcs) == 2
        assert all(isinstance(a, ArcPath) for a in arcs)

    def test_fifths_distance(self):
        assert fifths_distance(0, 7) == 1
        assert fifths_distance(0, 5) == 1

    def test_angular_distance(self):
        d = angular_distance(0.0, 180.0)
        assert d == pytest.approx(180.0)
        d2 = angular_distance(350.0, 10.0)
        assert d2 == pytest.approx(20.0)


class TestFormDiagram:
    def test_common_forms(self):
        assert "binary" in COMMON_FORMS
        assert "ternary" in COMMON_FORMS
        assert "sonata" in COMMON_FORMS
        assert COMMON_FORMS["ternary"] == ["A", "B", "A"]

    def test_section_boundaries(self):
        labels = ["A", "B", "A"]
        sections = section_boundaries(labels)
        assert len(sections) == 3
        assert sections[0].label == "A"

    def test_build_form_tree(self):
        sections = section_boundaries(["A", "B", "A"])
        tree = build_form_tree(sections)
        assert isinstance(tree, FormTree)
        assert len(tree.children) == 3
        assert tree.depth == 1

    def test_repetition_matrix(self):
        features = np.random.randn(4, 8)
        rm = repetition_matrix(features)
        assert rm.shape == (4, 4)
        for i in range(4):
            assert rm[i, i] == pytest.approx(1.0, abs=1e-6)

    def test_novelty_curve(self):
        sim = np.eye(16)
        nc = novelty_curve(sim, kernel_size=4)
        assert len(nc) == 16

    def test_section_similarity_matrix(self):
        sections = section_boundaries(["A", "B", "A"])
        sim = section_similarity_matrix(sections)
        assert sim.shape == (3, 3)
        assert sim[0, 2] == 1.0
        assert sim[0, 1] == 0.0

    def test_classify_form_binary(self):
        assert classify_form(["A", "B"]) == "binary"

    def test_classify_form_ternary(self):
        assert classify_form(["A", "B", "A"]) == "ternary"

    def test_classify_form_rondo(self):
        assert classify_form(["A", "B", "A", "C", "A"]) == "rondo"

    def test_classify_form_strophic(self):
        assert classify_form(["A", "A", "A"]) == "strophic"

    def test_classify_form_through_composed(self):
        assert classify_form(["A", "B", "C", "D"]) == "through_composed"
