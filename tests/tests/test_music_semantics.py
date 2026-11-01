"""Tests for music_semantics: genre ontology, instrument taxonomy, form analysis."""
from __future__ import annotations
import pytest

from models.music_semantics.genre_ontology import (
    GENRE_TREE, GenreNode,
    genre_ancestors, genre_descendants, common_ancestor,
    genre_distance, features_to_genre,
)
from models.music_semantics.instrument_taxonomy import (
    INSTRUMENT_CATALOG, Instrument,
    instrument_compatibility, ensemble_suggestion,
)
from models.music_semantics.form_analysis import (
    MUSICAL_FORMS, FormSection,
    classify_form, form_complexity,
)


# ── Genre Ontology ───────────────────────────────────────────────────────

class TestGenreOntology:
    def test_tree_has_root(self):
        assert "root" in GENRE_TREE

    def test_tree_has_top_level_genres(self):
        for g in ("rock", "pop", "jazz", "classical", "electronic", "hip_hop"):
            assert g in GENRE_TREE

    def test_tree_has_subgenres(self):
        for g in ("indie_rock", "bebop", "house", "trap"):
            assert g in GENRE_TREE

    def test_genre_ancestors_leaf(self):
        path = genre_ancestors("bebop")
        assert path[0] == "bebop"
        assert "jazz" in path
        assert "root" in path

    def test_genre_ancestors_root(self):
        path = genre_ancestors("root")
        assert path == ["root"]

    def test_genre_descendants_rock(self):
        desc = genre_descendants("rock")
        assert "rock" in desc
        assert "indie_rock" in desc
        assert "punk_rock" in desc

    def test_genre_descendants_unknown(self):
        assert genre_descendants("nonexistent") == []

    def test_common_ancestor_siblings(self):
        ca = common_ancestor("indie_rock", "punk_rock")
        assert ca == "rock"

    def test_common_ancestor_cousins(self):
        ca = common_ancestor("bebop", "indie_rock")
        assert ca == "root"

    def test_common_ancestor_same(self):
        assert common_ancestor("jazz", "jazz") == "jazz"

    def test_genre_distance_self(self):
        assert genre_distance("rock", "rock") == 0.0

    def test_genre_distance_siblings(self):
        d = genre_distance("indie_rock", "punk_rock")
        assert 0.0 < d < 1.0

    def test_genre_distance_far(self):
        d_near = genre_distance("indie_rock", "punk_rock")
        d_far = genre_distance("bebop", "death_metal")
        assert d_far >= d_near

    def test_genre_distance_unknown(self):
        assert genre_distance("bebop", "nonexistent") == 1.0

    def test_features_to_genre_ambient(self):
        assert features_to_genre(bpm=80, energy=0.2, valence=0.3, acousticness=0.9) == "ambient"

    def test_features_to_genre_metal(self):
        assert features_to_genre(bpm=180, energy=0.9, valence=0.2, acousticness=0.1) == "metal"

    def test_features_to_genre_pop(self):
        g = features_to_genre(bpm=125, energy=0.6, valence=0.7, acousticness=0.3)
        assert g == "pop"


# ── Instrument Taxonomy ──────────────────────────────────────────────────

class TestInstrumentTaxonomy:
    def test_catalog_size(self):
        assert len(INSTRUMENT_CATALOG) >= 40

    def test_instrument_fields(self):
        piano = INSTRUMENT_CATALOG["piano"]
        assert piano.family == "keyboard"
        assert piano.pitch_range[0] < piano.pitch_range[1]
        assert piano.is_pitched is True

    def test_compatibility_self(self):
        piano = INSTRUMENT_CATALOG["piano"]
        assert instrument_compatibility(piano, piano) == 1.0

    def test_compatibility_same_family(self):
        violin = INSTRUMENT_CATALOG["violin"]
        cello = INSTRUMENT_CATALOG["cello"]
        assert instrument_compatibility(violin, cello) > 0.7

    def test_compatibility_different_family(self):
        piano = INSTRUMENT_CATALOG["piano"]
        drum_kit = INSTRUMENT_CATALOG["drum_kit"]
        compat = instrument_compatibility(piano, drum_kit)
        assert 0.0 < compat < 1.0

    def test_compatibility_symmetry(self):
        a = INSTRUMENT_CATALOG["violin"]
        b = INSTRUMENT_CATALOG["trumpet"]
        assert abs(instrument_compatibility(a, b) - instrument_compatibility(b, a)) < 1e-10

    def test_ensemble_suggestion_size(self):
        ens = ensemble_suggestion("jazz", 4)
        assert len(ens) == 4

    def test_ensemble_suggestion_genre_match(self):
        ens = ensemble_suggestion("classical", 3)
        for name in ens:
            inst = INSTRUMENT_CATALOG[name]
            assert "classical" in inst.common_genres

    def test_ensemble_suggestion_unknown_genre(self):
        ens = ensemble_suggestion("nonexistent_genre", 3)
        assert len(ens) == 3

    def test_ensemble_family_diversity(self):
        ens = ensemble_suggestion("jazz", 5)
        families = {INSTRUMENT_CATALOG[n].family for n in ens}
        assert len(families) >= 2


# ── Form Analysis ────────────────────────────────────────────────────────

class TestFormAnalysis:
    def test_musical_forms_count(self):
        assert len(MUSICAL_FORMS) >= 10

    def test_classify_binary(self):
        sections = [
            FormSection("A", 1, 8),
            FormSection("B", 9, 16),
        ]
        assert classify_form(sections) == "binary"

    def test_classify_ternary(self):
        sections = [
            FormSection("A", 1, 8),
            FormSection("B", 9, 16),
            FormSection("A", 17, 24),
        ]
        assert classify_form(sections) == "ternary"

    def test_classify_strophic(self):
        sections = [FormSection("A", i*8+1, (i+1)*8) for i in range(4)]
        assert classify_form(sections) == "strophic"

    def test_classify_through_composed(self):
        labels = ["A", "B", "C", "D", "E"]
        sections = [FormSection(l, i*8+1, (i+1)*8) for i, l in enumerate(labels)]
        assert classify_form(sections) == "through_composed"

    def test_classify_verse_chorus(self):
        sections = [
            FormSection("V", 1, 8),
            FormSection("C", 9, 16),
            FormSection("V", 17, 24),
            FormSection("C", 25, 32),
        ]
        assert classify_form(sections) == "verse_chorus"

    def test_classify_rondo(self):
        sections = [
            FormSection("A", 1, 8),
            FormSection("B", 9, 16),
            FormSection("A", 17, 24),
            FormSection("C", 25, 32),
            FormSection("A", 33, 40),
        ]
        assert classify_form(sections) == "rondo"

    def test_classify_sonata(self):
        sections = [
            FormSection("Exp", 1, 40, is_development=False),
            FormSection("Dev", 41, 80, is_development=True),
            FormSection("Rec", 81, 120, is_development=False),
        ]
        assert classify_form(sections) == "sonata"

    def test_classify_empty(self):
        assert classify_form([]) == "unknown"

    def test_form_complexity_range(self):
        for form_name in MUSICAL_FORMS:
            c = form_complexity(form_name)
            assert 0.0 <= c <= 1.0

    def test_form_complexity_ordering(self):
        assert form_complexity("strophic") < form_complexity("sonata")

    def test_form_complexity_unknown(self):
        assert form_complexity("nonexistent") == 0.5

    def test_form_section_length(self):
        s = FormSection("A", 1, 17)
        assert s.length == 16
