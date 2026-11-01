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
