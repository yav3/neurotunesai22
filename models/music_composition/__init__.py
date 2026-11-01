"""Music composition: counterpoint, form, orchestration, voice leading, Markov."""
from models.music_composition.counterpoint import (
    FirstSpecies, SecondSpecies,
    check_parallel_motion, is_consonant, contrary_motion_score,
    CONSONANT_INTERVALS, PERFECT_CONSONANCES,
)
from models.music_composition.form import (
    SongForm, FormGenerator, VariationEngine,
    transpose, invert, retrograde,
)
from models.music_composition.orchestration import (
    InstrumentRange, INSTRUMENTS, Orchestrator,
    fits_range, assign_voices,
)
from models.music_composition.voice_leading import (
    VoiceLeading, VoiceLeader, NeoRiemannian,
    minimal_voice_leading, voice_leading_smoothness,
)
from models.music_composition.markov_composer import (
    PitchMarkov, RhythmMarkov, MarkovComposer,
)

__all__ = [
    "FirstSpecies", "SecondSpecies",
    "check_parallel_motion", "is_consonant", "contrary_motion_score",
    "CONSONANT_INTERVALS", "PERFECT_CONSONANCES",
    "SongForm", "FormGenerator", "VariationEngine",
    "transpose", "invert", "retrograde",
    "InstrumentRange", "INSTRUMENTS", "Orchestrator",
    "fits_range", "assign_voices",
    "VoiceLeading", "VoiceLeader", "NeoRiemannian",
    "minimal_voice_leading", "voice_leading_smoothness",
    "PitchMarkov", "RhythmMarkov", "MarkovComposer",
]
