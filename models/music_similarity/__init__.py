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

__all__ = [
    "TrackEmbedding", "EmbeddingSpace",
    "SimilarityEngine",
    "ContentBasedRecommender", "PlaylistRecommender",
    "MusicClusterer", "ClusterProfile",
    "CoverDetector", "CoverMatch",
    "chroma_from_signal", "optimal_transposition_index", "transpose_chroma",
    "cross_recurrence_matrix", "qmax_score", "dtw_chroma", "hpcp",
]
