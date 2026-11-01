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
]
