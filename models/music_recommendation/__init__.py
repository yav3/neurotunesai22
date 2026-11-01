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
from models.music_recommendation.collaborative import (
    InteractionMatrix,
    MFModel,
    PopularityRecommender,
    als_train,
    bm25_weight,
    implicit_confidence,
    item_knn_recommend,
    user_knn_recommend,
)
from models.music_recommendation.session_rec import (
    SessionMomentumTracker,
    SessionState,
    SequentialPattern,
    exponential_decay_centroid,
    find_sequential_patterns,
    markov_predict,
    markov_transition_matrix,
    session_centroid,
    session_coherence_score,
    session_diversity_score,
    session_knn_recommend,
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
    "InteractionMatrix", "MFModel", "PopularityRecommender",
    "als_train", "bm25_weight", "implicit_confidence",
    "item_knn_recommend", "user_knn_recommend",
    "SessionMomentumTracker", "SessionState", "SequentialPattern",
    "exponential_decay_centroid", "find_sequential_patterns",
    "markov_predict", "markov_transition_matrix", "session_centroid",
    "session_coherence_score", "session_diversity_score", "session_knn_recommend",
    "BanditStrategy", "ArmStats", "LinUCBArm", "LinUCBRecommender",
    "ThompsonSamplingRecommender", "EpsilonGreedyRecommender",
    "ucb1_scores", "exp3_weights", "RewardSignal", "evaluate_bandit",
]

from models.music_recommendation.contextual_bandit import (
    BanditStrategy, ArmStats, LinUCBArm, LinUCBRecommender,
    ThompsonSamplingRecommender, EpsilonGreedyRecommender,
    ucb1_scores, exp3_weights, RewardSignal, evaluate_bandit,
)
from models.music_recommendation.collaborative_filtering import (
    UserItemMatrix, cosine_similarity as cf_cosine_similarity,
    pearson_similarity, adjusted_cosine_similarity,
    user_knn as cf_user_knn, predict_rating_user_knn, item_knn as cf_item_knn,
    MFModel as CFMFModel, matrix_factorization,
    implicit_weight, cold_start_popularity, cold_start_content_bridge,
    intra_list_diversity, serendipity_score, diversity_rerank,
    coverage as rec_coverage, novelty as rec_novelty,
)
from models.music_recommendation.context_aware import (
    TimeOfDay, Activity, ListeningContext, ContextualPreference,
    BanditArm, epsilon_greedy, ucb1, thompson_sampling,
    SessionState as CtxSessionState, session_decay_scores,
    ListeningHistory as CtxListeningHistory, freshness_score, recency_boost,
    context_similarity, ContextualHistory, ACTIVITY_AUDIO_PRIORS,
    activity_match_score,
)
