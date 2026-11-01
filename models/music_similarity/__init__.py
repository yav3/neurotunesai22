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
    "HashMethod", "SpectralPeak", "Landmark", "FingerprintMatch",
    "stft_peaks", "peaks_to_landmarks", "fingerprint_signal",
    "build_hash_table", "match_fingerprints",
    "chroma_fingerprint", "hamming_distance", "chroma_match_score",
    "bit_error_rate", "robustness_score", "discriminability",
    "SimilarityMetric", "MelodicSimilarityResult",
    "edit_distance_normalised", "pitch_similarity", "interval_similarity",
    "contour_similarity", "rhythm_similarity", "mongeau_sankoff",
    "mongeau_sankoff_similarity", "melodic_similarity",
    "contour_reduction", "contour_reduction_similarity",
    "transposition_invariant_similarity", "inversion_similarity",
    "retrograde_similarity", "pairwise_melodic_similarity", "most_similar_pair",
    "DistanceMetric", "ml_euclidean", "ml_cosine_distance",
    "ml_manhattan", "ml_chebyshev", "ml_mahalanobis", "ml_pairwise_distances",
    "Triplet", "triplet_loss", "batch_triplet_loss",
    "contrastive_loss", "hard_negative_mining", "semi_hard_negative_mining",
    "recall_at_k", "mean_average_precision", "normalized_discounted_cumulative_gain",
    "embedding_calibration", "embedding_whitening", "proxy_nca_loss",
    "TonalSimilarityResult", "MAJOR_PROFILE", "MINOR_PROFILE",
    "chroma_from_pitches", "tonal_chroma_cosine",
    "tonal_centroid_distance", "detect_key", "key_distance_fifths",
    "key_similarity", "harmonic_rhythm_similarity",
    "tonal_similarity", "chroma_cross_correlation", "optimal_transposition",
    "pairwise_tonal_similarity",
]

from models.music_similarity.audio_fingerprint_matching import (
    HashMethod, SpectralPeak, Landmark, FingerprintMatch,
    stft_peaks, peaks_to_landmarks, fingerprint_signal,
    build_hash_table, match_fingerprints,
    chroma_fingerprint, hamming_distance, chroma_match_score,
    bit_error_rate, robustness_score, discriminability,
)
from models.music_similarity.metric_learning import (
    DistanceMetric,
    euclidean as ml_euclidean,
    cosine_distance as ml_cosine_distance,
    manhattan as ml_manhattan,
    chebyshev as ml_chebyshev,
    mahalanobis as ml_mahalanobis,
    pairwise_distances as ml_pairwise_distances,
    Triplet, triplet_loss, batch_triplet_loss, contrastive_loss,
    hard_negative_mining, semi_hard_negative_mining,
    recall_at_k, mean_average_precision, normalized_discounted_cumulative_gain,
    embedding_calibration, embedding_whitening, proxy_nca_loss,
)
from models.music_similarity.melodic_similarity import (
    SimilarityMetric, MelodicSimilarityResult,
    edit_distance_normalised, pitch_similarity, interval_similarity,
    contour_similarity, rhythm_similarity, mongeau_sankoff,
    mongeau_sankoff_similarity, melodic_similarity,
    contour_reduction, contour_reduction_similarity,
    transposition_invariant_similarity, inversion_similarity,
    retrograde_similarity, pairwise_melodic_similarity, most_similar_pair,
)
from models.music_similarity.tonal_similarity import (
    TonalSimilarityResult,
    MAJOR_PROFILE, MINOR_PROFILE,
    chroma_from_pitches, chroma_cosine_similarity as tonal_chroma_cosine,
    tonal_centroid_distance, detect_key, key_distance_fifths,
    key_similarity, harmonic_rhythm_similarity,
    tonal_similarity, chroma_cross_correlation, optimal_transposition,
    pairwise_tonal_similarity,
)
from models.music_similarity.audio_fingerprint_matching import (
    HashMethod, SpectralPeak, Landmark, FingerprintMatch,
    stft_peaks, peaks_to_landmarks, fingerprint_signal,
    build_hash_table, match_fingerprints,
    chroma_fingerprint, hamming_distance, chroma_match_score,
    bit_error_rate, robustness_score, discriminability,
)
