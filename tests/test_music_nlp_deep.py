"""Deep tests for music_nlp: discourse structure, sentiment lexicon,
review analysis, sight-reading difficulty."""
import numpy as np
import pytest

from models.music_nlp.discourse_structure import (
    SectionRole, RhetoricalRelation, DiscourseUnit, DiscourseTree,
    classify_sentence, segment_document, build_discourse_tree,
    argument_chain, section_proportions, discourse_feature_vector,
)
from models.music_nlp.sentiment_lexicon import (
    LexiconEntry, MUSIC_LEXICON, tokenize, score_token, score_text,
    sentiment_trajectory, lexicon_coverage,
)
from models.music_nlp.review_analysis import (
    analyze_review, rating_estimate, aspect_radar,
    review_feature_vector, lexicon_density,
)


# ---- Discourse Structure ----

class TestDiscourseClassification:
    def test_intro_detected(self):
        role, conf = classify_sentence("The band returns with their latest release.")
        assert role == SectionRole.INTRODUCTION

    def test_verdict_detected(self):
        role, conf = classify_sentence("Overall this is a must-listen album.")
        assert role == SectionRole.VERDICT

    def test_analysis_detected(self):
        role, conf = classify_sentence("The harmonic progression builds tension.")
        assert role == SectionRole.ANALYSIS

    def test_comparison_detected(self):
        role, conf = classify_sentence("This sounds similar to early Radiohead.")
        assert role == SectionRole.COMPARISON

    def test_default_description(self):
        role, conf = classify_sentence("The sky was blue that morning.")
        assert role == SectionRole.DESCRIPTION

    def test_confidence_bounded(self):
        _, conf = classify_sentence("Overall the mix is excellent.")
        assert 0.0 <= conf <= 1.0


class TestDiscourseSegmentation:
    REVIEW = (
        "The band returns with their debut album. "
        "The opening track features lush harmonies and a driving rhythm. "
        "The melodic structure builds through modulation. "
        "Compared to their earlier EP, the production is far more polished. "
        "Overall this is essential listening."
    )

    def test_segment_returns_units(self):
        units = segment_document(self.REVIEW)
        assert len(units) >= 3
        assert all(isinstance(u, DiscourseUnit) for u in units)

    def test_first_sentence_intro(self):
        units = segment_document(self.REVIEW)
        assert units[0].role == SectionRole.INTRODUCTION

    def test_last_sentence_verdict(self):
        units = segment_document(self.REVIEW)
        assert units[-1].role == SectionRole.VERDICT

    def test_tree_has_edges(self):
        tree = build_discourse_tree(self.REVIEW)
        assert isinstance(tree, DiscourseTree)
        assert len(tree.edges) == len(tree.units) - 1

    def test_argument_chain_extraction(self):
        tree = build_discourse_tree(self.REVIEW)
        chains = argument_chain(tree)
        assert isinstance(chains, list)

    def test_section_proportions_sum_to_one(self):
        tree = build_discourse_tree(self.REVIEW)
        props = section_proportions(tree)
        assert abs(sum(props.values()) - 1.0) < 1e-6

    def test_feature_vector_shape(self):
        fv = discourse_feature_vector(self.REVIEW)
        assert fv.shape == (9,)
        assert np.all(np.isfinite(fv))

    def test_empty_text(self):
        units = segment_document("")
        assert len(units) == 0


# ---- Sentiment Lexicon ----

class TestSentimentLexicon:
    def test_lexicon_has_entries(self):
        assert len(MUSIC_LEXICON) > 20

    def test_lexicon_entry_bounds(self):
        for entry in MUSIC_LEXICON.values():
            assert -1.0 <= entry.valence <= 1.0
            assert -1.0 <= entry.arousal <= 1.0
            assert -1.0 <= entry.dominance <= 1.0

    def test_score_known_token(self):
        entry = score_token("crescendo")
        assert entry is not None
        assert entry.valence > 0

    def test_score_unknown_token(self):
        assert score_token("xyzabc") is None

    def test_score_text_vad(self):
        v, a, d = score_text("The crescendo was electrifying and soaring.")
        assert v > 0  # positive valence
        assert a > 0  # high arousal

    def test_negation_flips_valence(self):
        v_pos, _, _ = score_text("The crescendo was virtuoso.")
        v_neg, _, _ = score_text("The crescendo was not virtuoso.")
        assert v_neg < v_pos

    def test_sentiment_trajectory_shape(self):
        texts = ["A soaring crescendo.", "A droning dirge.", "A delicate lullaby."]
        traj = sentiment_trajectory(texts)
        assert traj.shape == (3, 3)

    def test_lexicon_coverage_nonzero(self):
        cov = lexicon_coverage("The crescendo builds into a cacophony of dissonance.")
        assert cov > 0.0

    def test_lexicon_coverage_zero_for_nonmusic(self):
        cov = lexicon_coverage("The cat sat on the mat.")
        assert cov == 0.0

    def test_empty_text_returns_zeros(self):
        v, a, d = score_text("")
        assert v == 0.0 and a == 0.0 and d == 0.0

    def test_tokenize_lowercases(self):
        tokens = tokenize("CRESCENDO Fortissimo")
        assert "crescendo" in tokens
        assert "fortissimo" in tokens


# ---- Review Analysis ----

class TestReviewAnalysis:
    POSITIVE_REVIEW = (
        "This is a stunning masterpiece of an album. The production is brilliant "
        "and the vocal performance is outstanding. The melodies are captivating "
        "and the lyrics are profound. An original and innovative work that pushes "
        "boundaries. A must-listen for any fan of the genre."
    )
    NEGATIVE_REVIEW = (
        "A terrible and disappointing release. The mixing is poor and muddy. "
        "The performance is sloppy and uninspired. Generic and derivative "
        "songwriting with boring melodies. Skip this."
    )

    def test_positive_sentiment(self):
        result = analyze_review(self.POSITIVE_REVIEW)
        assert result.overall_sentiment > 0.5

    def test_negative_sentiment(self):
        result = analyze_review(self.NEGATIVE_REVIEW)
        assert result.overall_sentiment < -0.3

    def test_aspects_present(self):
        result = analyze_review(self.POSITIVE_REVIEW)
        assert "production" in result.aspects
        assert "performance" in result.aspects
        assert "composition" in result.aspects

    def test_superlatives_detected(self):
        result = analyze_review(self.POSITIVE_REVIEW)
        assert len(result.superlatives) > 0

    def test_recommendation_positive(self):
        result = analyze_review(self.POSITIVE_REVIEW)
        assert result.recommendation_signal > 0

    def test_rating_estimate_high(self):
        rating = rating_estimate(self.POSITIVE_REVIEW)
        assert rating > 7.0

    def test_rating_estimate_low(self):
        rating = rating_estimate(self.NEGATIVE_REVIEW)
        assert rating < 4.0

    def test_rating_scale(self):
        rating = rating_estimate(self.POSITIVE_REVIEW, scale=(0.0, 5.0))
        assert 0.0 <= rating <= 5.0

    def test_aspect_radar_bounded(self):
        radar = aspect_radar(self.POSITIVE_REVIEW)
        for name, score in radar.items():
            assert 0.0 <= score <= 1.0

    def test_feature_vector_shape(self):
        fv = review_feature_vector(self.POSITIVE_REVIEW)
        assert fv.shape == (10,)

    def test_lexicon_density_bounded(self):
        d = lexicon_density(self.POSITIVE_REVIEW)
        assert 0.0 <= d <= 1.0

    def test_empty_review(self):
        result = analyze_review("")
        assert result.overall_sentiment == 0.0
