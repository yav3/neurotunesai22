"""Music generation and synthesis: waveforms, melody, harmony, arrangement, structure."""
from models.music_generation.waveform import (
    Oscillator,
    WaveShape,
    ADSREnvelope,
    WaveformSynthesizer,
    NoiseGenerator,
)
from models.music_generation.melody import (
    MelodyConstraints,
    MarkovMelodyGenerator,
    ContourMelodyGenerator,
    MelodyMutator,
)
from models.music_generation.harmony import (
    VoiceLeadingRules,
    ChordProgressionGenerator,
    HarmonySuggester,
)
from models.music_generation.arrangement import (
    InstrumentRange,
    ArrangementTemplate,
    ArrangementEngine,
    DynamicsEngine,
)
from models.music_generation.structure import (
    SongSection,
    SongStructureTemplate,
    SongStructureGenerator,
    SectionTransition,
)

from models.music_generation.counterpoint import (
    MotionType, CounterpointRules, CounterpointReport, CounterpointAnalyzer,
    classify_motion, is_consonant, is_perfect,
    check_parallel_fifths, check_parallel_octaves,
    check_voice_crossing, check_hidden_fifths,
    first_species,
)

__all__ = [
    "Oscillator", "WaveShape", "ADSREnvelope", "WaveformSynthesizer", "NoiseGenerator",
    "MelodyConstraints", "MarkovMelodyGenerator", "ContourMelodyGenerator", "MelodyMutator",
    "VoiceLeadingRules", "ChordProgressionGenerator", "HarmonySuggester",
    "InstrumentRange", "ArrangementTemplate", "ArrangementEngine", "DynamicsEngine",
    "SongSection", "SongStructureTemplate", "SongStructureGenerator", "SectionTransition",
    "MotionType", "CounterpointRules", "CounterpointReport", "CounterpointAnalyzer",
    "classify_motion", "is_consonant", "is_perfect",
    "check_parallel_fifths", "check_parallel_octaves",
    "check_voice_crossing", "check_hidden_fifths", "first_species",
]
