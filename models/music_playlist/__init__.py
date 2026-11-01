from __future__ import annotations

from models.music_playlist.transition import (
    TransitionType,
    TransitionScore,
    TransitionScorer,
)
from models.music_playlist.energy_curve import (
    EnergyCurve,
    EnergyCurveShape,
    EnergyCurveDesigner,
)
from models.music_playlist.playlist_optimizer import (
    TrackSlot,
    PlaylistConstraints,
    PlaylistOptimizer,
)

from models.music_playlist.diversity import (
    genre_diversity, tempo_diversity, artist_diversity,
    decade_spread, mood_coverage, PlaylistDiversityReport,
    compute_diversity,
)
from models.music_playlist.sequencing import (
    MixStrategy, CAMELOT_WHEEL, SequenceTrack, PlaylistSequencer,
    camelot_compatible, camelot_distance, tempo_compatibility,
    transition_score, greedy_sequence, tempo_ramp_sequence,
    energy_arc_sequence,
)

__all__ = [
    "TransitionType",
    "TransitionScore",
    "TransitionScorer",
    "EnergyCurve",
    "EnergyCurveShape",
    "EnergyCurveDesigner",
    "TrackSlot",
    "PlaylistConstraints",
    "PlaylistOptimizer",
    "genre_diversity", "tempo_diversity", "artist_diversity",
    "decade_spread", "mood_coverage", "PlaylistDiversityReport",
    "compute_diversity",
    "MixStrategy", "CAMELOT_WHEEL", "SequenceTrack", "PlaylistSequencer",
    "camelot_compatible", "camelot_distance", "tempo_compatibility",
    "transition_score", "greedy_sequence", "tempo_ramp_sequence",
    "energy_arc_sequence",
    "MoodArc", "MoodPoint", "ARC_TEMPLATES",
    "plan_mood_journey", "match_track_to_target", "sequence_for_arc",
    "TransitionFeature", "TransitionMethod", "TransitionDetail", "FullTransitionScore",
    "key_compatibility", "energy_flow", "timbre_similarity",
    "compute_transition_score", "optimal_crossfade_point",
    "full_transition_score", "optimal_crossfade_duration",
    "classify_transition", "transition_matrix",
    "playlist_flow_curve", "detect_flow_breaks",
    "reorder_for_flow", "traveling_salesman_playlist",
    "dj_set_segmentation", "energy_arc_score",
    "laplacian_smoothness", "spectral_flow", "spectral_flow_score",
    "detect_segments", "segment_coherence", "full_coherence_analysis",
    "CoherenceReport",
    "SetTrack", "SetConstraints",
    "audience_attention_curve", "fatigue_penalty", "engagement_score",
    "greedy_set_optimize", "set_duration", "optimize_and_report", "SetReport",
    "PlaylistStats", "playlist_stats",
    "coherence_score", "surprise_curve", "listening_fatigue",
    "tempo_smoothness", "energy_arc_fit",
    "FlowScore", "compute_flow",
    "TrackCandidate", "GenerationRules",
    "generate_playlist", "fill_gap", "split_playlist",
    "ListenerState", "TrackEvent", "SessionState", "SessionAnalysis",
    "attention_decay", "attention_boost", "fatigue_accumulation",
    "ls_novelty_score", "classify_listener_state",
    "ls_energy_trajectory", "smooth_energy_curve",
    "skip_rate", "mean_skip_position", "replay_affinity",
    "detect_flow_segments", "ls_fatigue_onset", "optimal_session_length",
    "pacing_recommendation", "simulate_session",
    "UserProfile", "ItemProfile", "RatingMatrix", "CFConfig",
    "build_rating_matrix", "user_similarity", "item_similarity",
    "user_based_predict", "item_based_predict",
    "matrix_factorization_als", "implicit_feedback_scores",
    "cold_start_fallback", "popularity_baseline",
    "ndcg_at_k", "precision_at_k", "recall_at_k",
    "mean_reciprocal_rank", "catalog_coverage",
]

from models.music_playlist.mood_journey import (
    MoodArc, MoodPoint, ARC_TEMPLATES,
    plan_mood_journey, match_track_to_target, sequence_for_arc,
)
from models.music_playlist.transition_analysis import (
    TransitionFeature, TransitionMethod, TransitionDetail, FullTransitionScore,
    key_compatibility, energy_flow, timbre_similarity,
    compute_transition_score, optimal_crossfade_point,
    full_transition_score, optimal_crossfade_duration,
    classify_transition, transition_matrix,
    playlist_flow_curve, detect_flow_breaks,
    reorder_for_flow, traveling_salesman_playlist,
    dj_set_segmentation, energy_arc_score,
)
from models.music_playlist.coherence import (
    laplacian_smoothness, spectral_flow, spectral_flow_score,
    detect_segments, segment_coherence, full_coherence_analysis,
    CoherenceReport,
)
from models.music_playlist.set_optimizer import (
    SetTrack, SetConstraints,
    audience_attention_curve, fatigue_penalty, engagement_score,
    greedy_set_optimize, set_duration, optimize_and_report, SetReport,
)
from models.music_playlist.playlist_analytics import (
    PlaylistStats, playlist_stats,
    coherence_score, surprise_curve, listening_fatigue,
    tempo_smoothness, energy_arc_fit,
    FlowScore, compute_flow,
)
from models.music_playlist.playlist_generation import (
    TrackCandidate, GenerationRules,
    generate_playlist, fill_gap, split_playlist,
)
from models.music_playlist.transition_model import (
    TransitionFeatures as TransitionFeatures2,
    transition_score as transition_score2,
    key_compatibility_score as key_compat_score2,
    tempo_compatibility as tempo_compat2,
    energy_curve_fitness, crossfade_duration,
    PlaylistOptimizer as PlaylistOptimizer2,
)
from models.music_playlist.collaborative_filtering import (
    UserProfile, ItemProfile, RatingMatrix, CFConfig,
    build_rating_matrix, user_similarity, item_similarity,
    user_based_predict, item_based_predict,
    matrix_factorization_als, implicit_feedback_scores,
    cold_start_fallback, popularity_baseline,
    ndcg_at_k, precision_at_k, recall_at_k,
    mean_reciprocal_rank, catalog_coverage,
)
from models.music_playlist.listening_session import (
    ListenerState, TrackEvent, SessionState, SessionAnalysis,
    attention_decay, attention_boost, fatigue_accumulation,
    novelty_score as ls_novelty_score,
    classify_listener_state,
    energy_trajectory as ls_energy_trajectory,
    smooth_energy_curve, skip_rate, mean_skip_position, replay_affinity,
    detect_flow_segments,
    fatigue_onset as ls_fatigue_onset,
    optimal_session_length, pacing_recommendation, simulate_session,
)
