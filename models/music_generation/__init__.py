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
    "MarkovModel", "build_transition_matrix", "generate_sequence",
    "pitch_markov", "rhythm_markov", "chord_markov",
    "transition_entropy", "stationarity_check", "sequence_likelihood",
    "ConstraintType", "CSPSolution", "VoiceLeadingConstraints",
    "no_parallel_fifths", "no_parallel_octaves", "leap_constraint",
    "range_constraint", "scale_membership", "contrary_motion_score",
    "arc_consistency_3", "backtrack_search",
    "harmonize_melody", "generate_with_constraints",
    "OUParams", "LogisticParams", "IFSRule",
    "gp_ornstein_uhlenbeck", "logistic_map", "henon_map", "brownian_bridge",
    "ifs_attractor", "ifs_to_pitches",
    "gp_self_similar_rhythm", "gp_cantor_rhythm",
    "pink_noise_melody", "poisson_rhythm", "gp_markov_rhythm", "tendency_mask",
    "WaveNetConfig", "SampleRNNConfig", "VocoderConfig", "SynthesisResult",
    "causal_pad", "dilated_conv1d", "causal_dilated_conv1d",
    "gated_activation", "wavenet_residual_block", "wavenet_forward",
    "init_wavenet_weights", "receptive_field",
    "frame_level_rnn", "sample_level_mlp", "upsample_conditioning",
    "samplernn_forward", "init_samplernn_weights",
    "temperature_sample", "nucleus_sample", "top_k_sample",
    "apply_repetition_penalty", "mu_law_encode", "mu_law_decode",
    "additive_synthesis", "harmonic_amplitudes", "subtractive_synthesis",
    "noise_band_synthesis", "pulse_train",
    "get_window", "ns_stft", "ns_istft",
    "griffin_lim", "overlap_add", "mel_spectrogram_to_audio",
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
from models.music_generation.markov_chain import (
    MarkovModel, build_transition_matrix, generate_sequence,
    pitch_markov, rhythm_markov, chord_markov,
    transition_entropy, stationarity_check, sequence_likelihood,
)
from models.music_generation.algorithmic_composition import (
    LSystem as CompLSystem, LSystemRule as CompLSystemRule,
    CellularAutomaton, MarkovChainComposer, ConstraintSolver,
    stochastic_process, stochastic_to_pitches,
    euclidean_rhythm as comp_euclidean_rhythm,
    serial_row, serial_transformations, all_transpositions,
    fractal_melody,
)
from models.music_generation.constraint_satisfaction import (
    ConstraintType, CSPSolution, VoiceLeadingConstraints,
    no_parallel_fifths, no_parallel_octaves, leap_constraint,
    range_constraint, scale_membership, contrary_motion_score,
    arc_consistency_3, backtrack_search,
    harmonize_melody, generate_with_constraints,
)
from models.music_generation.generative_processes import (
    OUParams, LogisticParams, IFSRule,
    ornstein_uhlenbeck as gp_ornstein_uhlenbeck,
    logistic_map, henon_map, brownian_bridge,
    ifs_attractor, ifs_to_pitches,
    self_similar_rhythm as gp_self_similar_rhythm,
    cantor_rhythm as gp_cantor_rhythm,
    pink_noise_melody, poisson_rhythm,
    markov_rhythm as gp_markov_rhythm,
    tendency_mask,
)
from models.music_generation.neural_synthesis import (
    WaveNetConfig, SampleRNNConfig, VocoderConfig, SynthesisResult,
    causal_pad, dilated_conv1d, causal_dilated_conv1d,
    gated_activation, wavenet_residual_block, wavenet_forward,
    init_wavenet_weights, receptive_field,
    frame_level_rnn, sample_level_mlp, upsample_conditioning,
    samplernn_forward, init_samplernn_weights,
    temperature_sample, nucleus_sample, top_k_sample,
    apply_repetition_penalty, mu_law_encode, mu_law_decode,
    additive_synthesis, harmonic_amplitudes, subtractive_synthesis,
    noise_band_synthesis, pulse_train,
    get_window, stft as ns_stft, istft as ns_istft,
    griffin_lim, overlap_add, mel_spectrogram_to_audio,
)
from models.music_generation.deep_generation import (
    TransformerGenConfig, TransformerWeights, init_transformer_weights,
    transformer_forward, temperature_sample as deep_temperature_sample,
    nucleus_sample as deep_nucleus_sample, generate_sequence as deep_generate_sequence,
    VAEConfig, vae_encode, reparameterize, vae_decode, kl_divergence, latent_interpolate,
    CADENCE_TYPES, CHORD_TENSION, tension_arc as gen_tension_arc,
    generate_tonal_plan, insert_cadences, modulate,
    GrooveProfile, swing_groove, humanize, push_pull,
    euclidean_rhythm as deep_euclidean_rhythm, rotate_pattern, polymetric_combine,
    optimal_rotation, necklace_class,
    augmentation, diminution, retrograde, inversion, retrograde_inversion,
    transposition as motif_transposition, fragmentation, sequencing,
    rhythmic_displacement, develop_motif,
    GenerationResult, GenerationPipeline,
)
