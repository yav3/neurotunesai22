"""Music performance analysis: expression, tempo, intonation, ensemble, practice."""
from .expression import (
    DynamicLevel,
    ExpressionCurve,
    DynamicsAnalyzer,
    VibratoAnalyzer,
    ArticulationDetector,
)
from .tempo_expression import (
    RubatoDetector,
    AccelerandoDetector,
    AgogicAccent,
    MetricalStrength,
)
from .intonation import (
    PitchTracker,
    IntonationAnalyzer,
    TemperamentComparator,
    MicrotoneDetector,
)
from .ensemble import (
    SynchronyAnalyzer,
    BlendAnalyzer,
    BalanceAnalyzer,
    TuningAnalyzer,
)
from .practice import (
    PracticeSession,
    ProgressTracker,
    DifficultyEstimator,
    PracticeScheduler,
)

__all__ = [
    "DynamicLevel", "ExpressionCurve", "DynamicsAnalyzer",
    "VibratoAnalyzer", "ArticulationDetector",
    "RubatoDetector", "AccelerandoDetector", "AgogicAccent", "MetricalStrength",
    "PitchTracker", "IntonationAnalyzer", "TemperamentComparator", "MicrotoneDetector",
    "SynchronyAnalyzer", "BlendAnalyzer", "BalanceAnalyzer", "TuningAnalyzer",
    "PracticeSession", "ProgressTracker", "DifficultyEstimator", "PracticeScheduler",
    "OrnamentType", "Ornament", "MicroEvent",
    "realize_trill", "realize_mordent", "realize_turn",
    "realize_ornament", "apply_ornaments", "OrnamentDensity", "compute_density",
    "ArticulationType", "NoteArticulation", "classify_articulation",
    "analyze_articulations", "articulation_consistency", "legato_score",
    "staccato_ratio", "dynamic_accent_positions", "phrase_articulation_profile",
    "expressive_range",
    "SightReadingProfile", "PracticeExercise", "INTERVAL_DIFFICULTY", "KEY_ACCIDENTALS",
    "sr_rhythmic_complexity_score", "sr_pitch_complexity_score", "interval_difficulty_score",
    "key_complexity_score", "accidental_density", "temporal_irregularity",
    "sr_estimate_difficulty", "generate_exercise", "progressive_sequence",
    "difficulty_feature_vector",
    "tr_RubatoType", "TempoEvent", "TempoCurve",
    "ioi_to_tempo", "tempo_curve_from_onsets", "phrase_rubato",
    "cadential_ritardando", "timing_profile", "micro_timing_grid",
    "swing_ratio", "expressive_timing_score",
    "TempoFeature", "TempoPoint", "TempoArc", "td_AgogicEvent",
    "ioi_from_onsets", "onset_to_local_tempo", "smooth_tempo_curve",
    "onset_deviations_ms", "find_rubato_regions", "find_agogic_accents",
    "find_ritardando", "find_accelerando", "detect_phrase_arcs",
    "tempo_cv", "tempo_stability_score", "expressiveness_index",
    "performance_consistency", "final_rit_strength",
]

from .ornamentation import (
    OrnamentType, Ornament, MicroEvent,
    realize_trill, realize_mordent, realize_turn,
    realize_ornament, apply_ornaments, OrnamentDensity, compute_density,
)
from .articulation import (
    ArticulationType, NoteArticulation, classify_articulation,
    analyze_articulations, articulation_consistency, legato_score,
    staccato_ratio, dynamic_accent_positions, phrase_articulation_profile,
    expressive_range,
)
from .stage_acoustics import (
    VenueProfile, estimate_rt60, clarity_c80, definition_d50,
    center_time, bass_ratio, stage_support, analyze_venue,
)
from .performance_metrics import (
    PerformanceScore, timing_accuracy, pitch_accuracy,
    dynamic_fidelity, articulation_score as perf_articulation_score,
    tempo_stability as perf_tempo_stability, rhythmic_evenness,
    ProgressTracker as MetricsProgressTracker,
)
from .expression_analysis import (
    ArticulationType as ExprArticulationType, NoteEvent,
    RubatoSegment, detect_rubato, rubato_amount,
    DynamicShape, detect_dynamic_shapes, dynamic_range,
    classify_articulation as classify_expr_articulation,
    articulation_profile, VibratoAnalysis, analyze_vibrato,
    expressive_timing_deviation,
)
from .ensemble_sync import (
    PerformerTrack, SyncMetrics, inter_onset_alignment,
    pairwise_sync_matrix, granger_precedence, detect_leader,
    sync_error_distribution, phase_coupling, tempo_negotiation,
    sectional_balance, balance_deviation, ensemble_cohesion,
    synchronization_over_time,
)
from models.music_performance.tempo_rubato import (
    RubatoType as tr_RubatoType, TempoEvent, TempoCurve,
    ioi_to_tempo, tempo_curve_from_onsets, phrase_rubato,
    cadential_ritardando, timing_profile, micro_timing_grid,
    swing_ratio, expressive_timing_score,
)
from .sight_reading import (
    SightReadingProfile, PracticeExercise, INTERVAL_DIFFICULTY, KEY_ACCIDENTALS,
    rhythmic_complexity_score as sr_rhythmic_complexity_score,
    pitch_complexity_score as sr_pitch_complexity_score,
    interval_difficulty_score, key_complexity_score,
    accidental_density, temporal_irregularity,
    estimate_difficulty as sr_estimate_difficulty,
    generate_exercise, progressive_sequence, difficulty_feature_vector,
)
from .tempo_dynamics import (
    TempoFeature, TempoPoint, TempoArc,
    AgogicEvent as td_AgogicEvent,
    ioi_from_onsets, onset_to_local_tempo, smooth_tempo_curve,
    onset_deviations_ms, find_rubato_regions, find_agogic_accents,
    find_ritardando, find_accelerando, detect_phrase_arcs,
    tempo_cv, tempo_stability_score, expressiveness_index,
    performance_consistency, final_rit_strength,
)
