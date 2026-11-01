"""Music cognition: consonance, expectation, groove, tension, memory."""
from models.music_cognition.consonance import (
    RoughnessModel, IntervalConsonance,
    roughness, interval_consonance, chord_consonance,
    INTERVAL_CONSONANCE,
)
from models.music_cognition.expectation import (
    MarkovExpectation, MelodicExpectation,
    information_content, melodic_surprise,
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

__all__ = [
    "RoughnessModel", "IntervalConsonance",
    "roughness", "interval_consonance", "chord_consonance", "INTERVAL_CONSONANCE",
    "MarkovExpectation", "MelodicExpectation",
    "information_content", "melodic_surprise",
    "GrooveAnalyzer", "MicrotimingProfile",
    "syncopation_index", "swing_ratio",
    "TensionCurve", "SurpriseTracker",
    "harmonic_tension", "voice_leading_cost", "minimal_voice_leading",
    "RepetitionDetector", "FamiliarityModel",
    "find_repetitions", "familiarity_preference",
    "AccentType", "PerceptualAccent", "MetricalGrid",
    "compute_syncopation", "compute_groove_complexity",
    "beat_induction", "phase_coherence", "PerceptualGrouping",
]
