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

__all__ = [
    "Oscillator", "WaveShape", "ADSREnvelope", "WaveformSynthesizer", "NoiseGenerator",
    "MelodyConstraints", "MarkovMelodyGenerator", "ContourMelodyGenerator", "MelodyMutator",
    "VoiceLeadingRules", "ChordProgressionGenerator", "HarmonySuggester",
    "InstrumentRange", "ArrangementTemplate", "ArrangementEngine", "DynamicsEngine",
    "SongSection", "SongStructureTemplate", "SongStructureGenerator", "SectionTransition",
]
