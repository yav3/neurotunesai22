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
]
