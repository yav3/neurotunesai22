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
    "TransitionFeature",
    "key_compatibility", "energy_flow",
    "compute_transition_score", "optimal_crossfade_point",
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
]

from models.music_playlist.mood_journey import (
    MoodArc, MoodPoint, ARC_TEMPLATES,
    plan_mood_journey, match_track_to_target, sequence_for_arc,
)
from models.music_playlist.transition_analysis import (
    TransitionFeature,
    key_compatibility, energy_flow,
    compute_transition_score, optimal_crossfade_point,
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
