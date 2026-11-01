"""Music cognition: consonance, expectation, groove, tension, memory."""
from models.music_cognition.consonance import (
    RoughnessModel, IntervalConsonance,
    roughness, interval_consonance, chord_consonance,
    INTERVAL_CONSONANCE,
)
from models.music_cognition.expectation import (
    MarkovExpectation, MelodicExpectation,
    information_content, melodic_surprise,
    NarmourExpectancy, narmour_curve,
    StatisticalLearner, surprise_curve,
)
from models.music_cognition.groove import (
    GrooveAnalyzer, MicrotimingProfile,
    syncopation_index, swing_ratio,
)
from models.music_cognition.tension import (
    TensionCurve, SurpriseTracker,
    harmonic_tension, voice_leading_cost, minimal_voice_leading,
)
from models.music_cognition.memory import (
    RepetitionDetector, FamiliarityModel,
    find_repetitions, familiarity_preference,
)

from models.music_cognition.rhythm_perception import (
    AccentType, PerceptualAccent, MetricalGrid,
    compute_syncopation, compute_groove_complexity,
    beat_induction, phase_coherence, PerceptualGrouping,
)
from models.music_cognition.auditory_attention import (
    SalienceMap, spectral_salience, onset_salience, compute_salience_map,
    AuditoryStream, stream_segregation, attention_decay, surprise_score,
)
from models.music_cognition.emotional_response import (
    BRECVEMAMechanism, EmotionalResponse,
    predict_emotional_response, chills_predictor, nostalgia_index,
    mechanism_activation_profile, valence_arousal_from_features, emotion_trajectory,
)
from models.music_cognition.learning import (
    SkillLevel, SkillProfile,
    skill_progression, optimal_practice_schedule, sight_reading_difficulty,
    IntervalType, INTERVAL_DIFFICULTY, ear_training_accuracy,
    spaced_repetition_interval, practice_efficiency,
)

__all__ = [
    "RoughnessModel", "IntervalConsonance",
    "roughness", "interval_consonance", "chord_consonance", "INTERVAL_CONSONANCE",
    "MarkovExpectation", "MelodicExpectation",
    "information_content", "melodic_surprise",
    "NarmourExpectancy", "narmour_curve",
    "StatisticalLearner", "surprise_curve",
    "GrooveAnalyzer", "MicrotimingProfile",
    "syncopation_index", "swing_ratio",
    "TensionCurve", "SurpriseTracker",
    "harmonic_tension", "voice_leading_cost", "minimal_voice_leading",
    "RepetitionDetector", "FamiliarityModel",
    "find_repetitions", "familiarity_preference",
    "AccentType", "PerceptualAccent", "MetricalGrid",
    "compute_syncopation", "compute_groove_complexity",
    "beat_induction", "phase_coherence", "PerceptualGrouping",
    "SalienceMap", "spectral_salience", "onset_salience", "compute_salience_map",
    "AuditoryStream", "stream_segregation", "attention_decay", "surprise_score",
    "BRECVEMAMechanism", "EmotionalResponse",
    "predict_emotional_response", "chills_predictor", "nostalgia_index",
    "mechanism_activation_profile", "valence_arousal_from_features", "emotion_trajectory",
    "SkillLevel", "SkillProfile",
    "skill_progression", "optimal_practice_schedule", "sight_reading_difficulty",
    "IntervalType", "INTERVAL_DIFFICULTY", "ear_training_accuracy",
    "spaced_repetition_interval", "practice_efficiency",
    "KRUMHANSL_MAJOR", "KRUMHANSL_MINOR",
    "TonalMode", "KeyEstimate", "TonalTensionProfile",
    "pitch_class_distribution", "estimate_key", "key_profile_correlation",
    "tonal_tension", "modulation_distance", "surprisal", "tonal_entropy", "key_clarity",
    "AuditoryStream", "StreamingEvent", "SceneAnalysis",
    "segregate_streams", "gestalt_proximity_score", "gestalt_similarity_score",
    "gestalt_common_fate", "gestalt_good_continuation",
    "detect_stream_boundaries", "perceptual_voice_separation",
    "scene_attention_salience", "stream_coherence",
    "CognitiveLoadEstimate", "InformationDensity", "WorkingMemoryModel",
    "harmonic_information_rate", "pitch_entropy", "rhythmic_entropy",
    "textural_density", "estimate_cognitive_load", "attention_resource_allocation",
    "processing_bottleneck", "musical_syntax_complexity",
    "PerformanceCognitiveLoad", "estimate_performance_load",
    "ProlongationType", "SyntacticRelation", "SyntacticEvent", "SyntacticViolation",
    "ProlongationalNode", "detect_syntactic_violations", "harmonic_syntax_score",
    "prolongational_reduction", "syntactic_complexity", "cadence_detection",
    "hierarchical_tension", "expectancy_violation_eran",
    "BodyPart", "BODY_PARTS", "MovementPhase",
    "movement_periodicity", "entrainment_strength", "phase_coupling",
    "kinetic_energy", "movement_fluency", "gesture_segmentation",
    "body_resonance", "optimal_tempo_for_movement",
    "movement_audio_synchrony", "groove_body_response",
    "DanceMetrics", "analyze_dance",
]

from models.music_cognition.schema_theory import (
    SchemaTemplate, SCHEMA_CATALOG,
    ROMANESCA, PRINNER, FONTE, MONTE, DO_RE_MI, SOL_FA_MI,
    QUIESCENZA, CADENZA_DOPPIA, INDUGIO, FENAROLI,
    pitch_to_degree, pitches_to_degrees,
    match_schema, find_schemata,
    generate_schema_melody, generate_schema_bass,
    schema_complexity, schema_catalog_summary,
)
from models.music_cognition.tonal_expectation import (
    KRUMHANSL_MAJOR, KRUMHANSL_MINOR,
    Mode as TonalMode, KeyEstimate, TonalTensionProfile,
    pitch_class_distribution, estimate_key, key_profile_correlation,
    tonal_tension, modulation_distance, surprisal,
    entropy as tonal_entropy, key_clarity,
)
from models.music_cognition.auditory_scene import (
    AuditoryStream, StreamingEvent, SceneAnalysis,
    segregate_streams, gestalt_proximity_score, gestalt_similarity_score,
    gestalt_common_fate, gestalt_good_continuation,
    detect_stream_boundaries, perceptual_voice_separation,
    attention_salience as scene_attention_salience, stream_coherence,
)
from models.music_cognition.cognitive_load import (
    CognitiveLoadEstimate, InformationDensity, WorkingMemoryModel,
    harmonic_information_rate, pitch_entropy, rhythmic_entropy,
    textural_density, estimate_cognitive_load, attention_resource_allocation,
    processing_bottleneck, musical_syntax_complexity,
    PerformanceCognitiveLoad, estimate_performance_load,
)
from models.music_cognition.musical_syntax import (
    ProlongationType, SyntacticRelation, SyntacticEvent, SyntacticViolation,
    ProlongationalNode, detect_syntactic_violations, harmonic_syntax_score,
    prolongational_reduction, syntactic_complexity, cadence_detection,
    hierarchical_tension, expectancy_violation_eran,
)
from models.music_cognition.embodied_cognition import (
    BodyPart, BODY_PARTS, MovementPhase,
    movement_periodicity, entrainment_strength, phase_coupling,
    kinetic_energy, movement_fluency, gesture_segmentation,
    body_resonance, optimal_tempo_for_movement,
    movement_audio_synchrony, groove_body_response,
    DanceMetrics, analyze_dance,
)
