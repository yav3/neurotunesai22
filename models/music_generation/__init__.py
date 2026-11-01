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
    "RhythmGrid", "TimeSignature", "GenRhythmPattern",
    "euclidean_rhythm_gen", "rhythm_density", "syncopation_index",
    "generate_kick_pattern", "generate_hihat_pattern",
    "combine_patterns", "apply_groove", "pattern_similarity",
    "mutate_rhythm",
    "LSystem as CompLSystem", "LSystemRule as CompLSystemRule",
    "CellularAutomaton", "ca_to_rhythm as comp_ca_to_rhythm",
    "MarkovChainComposer", "ConstraintSolver",
    "stochastic_process", "stochastic_to_pitches",
    "euclidean_rhythm as comp_euclidean_rhythm",
    "serial_row", "serial_transformations", "all_transpositions",
    "fractal_melody",
]

from models.music_generation.rhythm_generation import (
    RhythmGrid, TimeSignature, RhythmPattern as GenRhythmPattern,
    euclidean_rhythm as euclidean_rhythm_gen, density as rhythm_density,
    syncopation_index,
    generate_kick_pattern, generate_hihat_pattern,
    combine_patterns, apply_groove, pattern_similarity,
    mutate_rhythm,
)
from models.music_generation.algorithmic import (
    LSystemRule, LSystem, lsystem_to_pitches,
    CARule, elementary_ca, ca_to_rhythm, ca_to_pitches,
    brownian_melody, ornstein_uhlenbeck_melody, markov_rhythm,
    cantor_rhythm, sierpinski_rhythm, fibonacci_durations, golden_ratio_pitches,
)
from models.music_generation.texture_generation import (
    TextureType, GrainParams, DroneParams, TextureLayer, TextureScore,
    generate_drone_partials, grain_envelope, generate_granular_texture,
    noise_sculpt, spectral_freeze, additive_texture,
    layer_mix, texture_density, evolve_parameter,
)
from models.music_generation.algorithmic_composition import (
    LSystem as CompLSystem, LSystemRule as CompLSystemRule,
    CellularAutomaton, MarkovChainComposer, ConstraintSolver,
    stochastic_process, stochastic_to_pitches,
    euclidean_rhythm as comp_euclidean_rhythm,
    serial_row, serial_transformations, all_transpositions,
    fractal_melody,
)
