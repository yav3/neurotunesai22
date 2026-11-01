"""Music similarity, recommendation, and clustering."""
from models.music_similarity.embedding import TrackEmbedding, EmbeddingSpace
from models.music_similarity.similarity import SimilarityEngine
from models.music_similarity.recommender import ContentBasedRecommender, PlaylistRecommender
from models.music_similarity.clustering import MusicClusterer, ClusterProfile
from models.music_similarity.cover_detection import (
    CoverDetector, CoverMatch,
    chroma_from_signal, optimal_transposition_index, transpose_chroma,
    cross_recurrence_matrix, qmax_score, dtw_chroma, hpcp,
)

from models.music_similarity.ssm_analysis import (
    StructuralSegment, self_similarity_matrix, recurrence_matrix,
    diagonal_score, find_repetitions, novelty_curve,
    segment_from_novelty, label_segments, form_string,
    form_similarity, symmetry_score, repetitiveness,
)
from models.music_similarity.structural_similarity import (
    SectionType, FormSection, StructuralMatch,
    chroma_cosine_similarity, form_edit_distance, normalized_form_similarity,
    dtw_distance, qmax_similarity,
    melodic_contour_similarity, rhythmic_pattern_similarity,
    cover_song_score,
)

from models.music_similarity.sequence_alignment import (
    dtw_distance as dtw_align, edit_distance_melodic,
    longest_common_subsequence, smith_waterman_alignment, needleman_wunsch,
    SequenceAlignment, ChromaAlignment, MelodyAlignment, RhythmAlignment,
)
from models.music_similarity.embedding_distance import (
    EmbeddingDistanceMetrics, StructuralSimilarity,
)
from models.music_similarity.embedding_similarity import (
    FeatureAggregator, EmbeddingSimilarity, TripletEvaluator, CrossModalSimilarity,
)

__all__ = [
    "SectionType", "FormSection", "StructuralMatch",
    "chroma_cosine_similarity", "form_edit_distance", "normalized_form_similarity",
    "dtw_distance", "qmax_similarity",
    "melodic_contour_similarity", "rhythmic_pattern_similarity",
    "cover_song_score",
    "StructuralSegment", "self_similarity_matrix", "recurrence_matrix",
    "diagonal_score", "find_repetitions", "novelty_curve",
    "segment_from_novelty", "label_segments", "form_string",
    "form_similarity", "symmetry_score", "repetitiveness",
    "TrackEmbedding", "EmbeddingSpace",
    "SimilarityEngine",
    "ContentBasedRecommender", "PlaylistRecommender",
    "MusicClusterer", "ClusterProfile",
    "CoverDetector", "CoverMatch",
    "chroma_from_signal", "optimal_transposition_index", "transpose_chroma",
    "cross_recurrence_matrix", "qmax_score", "dtw_chroma", "hpcp",
    "dtw_align", "edit_distance_melodic",
    "longest_common_subsequence", "smith_waterman_alignment", "needleman_wunsch",
    "EmbeddingDistanceMetrics", "StructuralSimilarity",
    "SequenceAlignment", "ChromaAlignment", "MelodyAlignment", "RhythmAlignment",
    "FeatureAggregator", "EmbeddingSimilarity", "TripletEvaluator", "CrossModalSimilarity",
]
