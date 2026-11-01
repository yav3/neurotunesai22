"""Music recommendation: similarity, embeddings, clustering, recommender, taste modeling, discovery."""
from models.music_recommendation.similarity import (
    cosine_similarity,
    euclidean_distance,
    weighted_feature_similarity,
    mood_similarity,
    harmonic_compatibility,
    rhythm_similarity,
    FeatureSimilarity,
    HarmonicSimilarity,
    RhythmicSimilarity,
    EmotionalSimilarity,
    CompositeSimilarity,
)
from models.music_recommendation.embeddings import (
    TrackEmbedding,
    EmbeddingSpace,
    build_embedding,
    pca_reduce,
)
from models.music_recommendation.clustering import (
    MusicClusterer,
    MoodClusterer,
    GenreProfiler,
    ClusterLabel,
)
from models.music_recommendation.recommender import (
    ContentBasedRecommender,
    MoodBasedRecommender,
    PlaylistContinuationRecommender,
    ContextualRecommender,
    DiversityMixer,
)
from models.music_recommendation.taste_profile import (
    ListeningHistory,
    TasteProfile,
    TasteEvolution,
    TasteDistance,
)
from models.music_recommendation.discovery import (
    NoveltyScorer,
    ExplorationExploitation,
    GenreBridging,
    MoodJourney,
)
from models.music_recommendation.filters import (
    BPMFilter,
    GenreFilter,
    KeyFilter,
    DurationFilter,
    EnergyFilter,
    ExclusionFilter,
    CompositeFilter,
)

__all__ = [
    "cosine_similarity", "euclidean_distance", "weighted_feature_similarity",
    "mood_similarity", "harmonic_compatibility", "rhythm_similarity",
    "FeatureSimilarity", "HarmonicSimilarity", "RhythmicSimilarity",
    "EmotionalSimilarity", "CompositeSimilarity",
    "TrackEmbedding", "EmbeddingSpace", "build_embedding", "pca_reduce",
    "MusicClusterer", "MoodClusterer", "GenreProfiler", "ClusterLabel",
    "ContentBasedRecommender", "MoodBasedRecommender",
    "PlaylistContinuationRecommender", "ContextualRecommender", "DiversityMixer",
    "ListeningHistory", "TasteProfile", "TasteEvolution", "TasteDistance",
    "NoveltyScorer", "ExplorationExploitation", "GenreBridging", "MoodJourney",
    "BPMFilter", "GenreFilter", "KeyFilter", "DurationFilter",
    "EnergyFilter", "ExclusionFilter", "CompositeFilter",
]
