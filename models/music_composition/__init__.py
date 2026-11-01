"""Music composition: counterpoint, form, orchestration, voice leading, Markov."""
from models.music_composition.counterpoint import (
    FirstSpecies, SecondSpecies, ThirdSpecies, FourthSpecies,
    check_parallel_motion, is_consonant, contrary_motion_score,
    classify_motion, motion_profile, generate_first_species,
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
from models.music_composition.texture_analysis import (
    TextureType, TextureProfile, TextureSegment,
    estimate_voice_count, pitch_spread, onset_synchrony,
    rhythmic_independence_score, textural_density,
    classify_texture, analyze_texture, segment_texture,
)
from models.music_composition.rhythm_pattern import (
    RhythmPattern, PolymetricPattern,
    euclidean_rhythm, polyrhythm, swing_quantize,
    syncopation_score, onset_to_ioi, ioi_regularity,
    create_polymetric, groove_template, humanize,
)

from models.music_composition.motivic_transformation import (
    Motif, retrograde as motif_retrograde, inversion as motif_inversion,
    retrograde_inversion, augmentation, diminution, transposition,
    fragmentation, sequence as motif_sequence,
    motivic_similarity, interval_sequence, contour, contour_similarity,
    develop_motif, permute_rhythm, TRANSFORMS as MOTIF_TRANSFORMS,
)
from models.music_composition.cadence import (
    CadenceType, Cadence, detect_cadence, generate_cadence,
    cadential_strength, find_cadences, suggest_cadence,
    is_cadential_64, cadence_rate,
)

__all__ = [
    "FirstSpecies", "SecondSpecies",
    "check_parallel_motion", "is_consonant", "contrary_motion_score",
    "CONSONANT_INTERVALS", "PERFECT_CONSONANCES",
    "ThirdSpecies", "FourthSpecies",
    "classify_motion", "motion_profile", "generate_first_species",
    "SongForm", "FormGenerator", "VariationEngine",
    "transpose", "invert", "retrograde",
    "InstrumentRange", "INSTRUMENTS", "Orchestrator",
    "fits_range", "assign_voices",
    "VoiceLeading", "VoiceLeader", "NeoRiemannian",
    "minimal_voice_leading", "voice_leading_smoothness",
    "PitchMarkov", "RhythmMarkov", "MarkovComposer",
    "RhythmPattern", "PolymetricPattern",
    "euclidean_rhythm", "polyrhythm", "swing_quantize",
    "syncopation_score", "onset_to_ioi", "ioi_regularity",
    "create_polymetric", "groove_template", "humanize",
    "TextureType", "TextureProfile", "TextureSegment",
    "estimate_voice_count", "pitch_spread", "onset_synchrony",
    "rhythmic_independence_score", "textural_density",
    "classify_texture", "analyze_texture", "segment_texture",
    "Motif", "motif_retrograde", "motif_inversion",
    "retrograde_inversion", "augmentation", "diminution", "transposition",
    "fragmentation", "motif_sequence",
    "motivic_similarity", "interval_sequence", "contour", "contour_similarity",
    "develop_motif", "permute_rhythm", "MOTIF_TRANSFORMS",
    "CadenceType", "Cadence", "detect_cadence", "generate_cadence",
    "cadential_strength", "find_cadences", "suggest_cadence",
    "is_cadential_64", "cadence_rate",
    "ToneRow", "RowForm", "RowTransformation",
    "validate_row", "prime", "serial_retrograde", "serial_inversion",
    "serial_retrograde_inversion", "transpose_row", "row_matrix",
    "interval_vector", "all_interval_row", "combinatoriality",
    "derive_row",
    "HarmonicFunction", "Chord", "Progression", "ModulationEvent",
    "ROMAN_TO_SCALE_DEGREE", "FUNCTION_MAP", "TENSION_VALUES",
    "parse_roman", "chord_tension", "functional_analysis",
    "tension_arc", "root_motion_profile", "fifth_motion_ratio",
    "harmonic_rhythm_density", "harmonic_variety",
    "progression_complexity", "detect_cadence_type",
    "detect_modulations", "progression_from_symbols",
    "DevMotive", "TransformType",
    "dev_transpose", "dev_invert", "dev_retrograde", "dev_retrograde_inv",
    "augment", "diminish", "fragment", "dev_sequence",
    "tonal_transpose", "interval_edit_distance", "dev_contour_sim",
    "develop", "find_motive_occurrences",
    "dev_motive_similarity", "detect_motive_recurrence",
    "development_density", "motive_network",
    "contour_reduction", "interval_class_vector", "motive_complexity",
    "find_latent_motives", "motive_catalogue",
    "tp_TextureType", "tp_TextureSegment", "TexturePlan",
    "onset_density", "registral_spread", "registral_centroid",
    "tp_classify_texture", "voice_onset_correlation",
    "plan_texture_arc", "density_interpolation", "voice_count_curve",
    "VariationType", "EmbellishmentType", "Theme", "Variation", "VariationSet",
    "melodic_embellishment", "rhythmic_augmentation", "rhythmic_diminution",
    "rhythmic_displacement", "dotted_rhythm", "lombardic_rhythm",
    "reharmonize", "invertible_counterpoint", "ground_bass_variations",
    "theme_similarity", "developing_variation", "sequence_variation",
    "ornamental_variation", "textural_thickening",
    "variation_distance", "cumulative_variation_arc",
    "ConstraintKind", "PitchRange", "IntervalConstraint", "ConsonanceTarget",
    "DensityConstraint", "ContourConstraint", "VoiceLeadingConstraint",
    "CompositionConstraints", "GeneratedMelody",
    "generate_constrained_melody", "generate_bass_line",
    "two_voice_counterpoint", "melody_constraint_score", "stochastic_walk",
]

from models.music_composition.constraint_composition import (
    ConstraintKind, PitchRange, IntervalConstraint, ConsonanceTarget,
    DensityConstraint, ContourConstraint, VoiceLeadingConstraint,
    CompositionConstraints, GeneratedMelody,
    generate_constrained_melody, generate_bass_line,
    two_voice_counterpoint, melody_constraint_score, stochastic_walk,
)
from models.music_composition.motivic_development import (
    Motive as DevMotive, TransformType,
    transpose as dev_transpose, invert as dev_invert,
    retrograde as dev_retrograde, retrograde_inversion as dev_retrograde_inv,
    augment, diminish, fragment, sequence as dev_sequence,
    tonal_transpose, interval_edit_distance, contour_similarity as dev_contour_sim,
    develop, find_motive_occurrences,
    motive_similarity as dev_motive_similarity,
    detect_motive_recurrence, development_density, motive_network,
    contour_reduction, interval_class_vector, motive_complexity,
    find_latent_motives, motive_catalogue,
)
from models.music_composition.harmonic_planning import (
    ChordSymbol, TensionScore, HarmonicPlan,
    surface_dissonance, harmonic_distance, tonal_distance,
    compute_tension, arch_tension, ramp_tension, wave_tension,
    plan_progression, valid_next_chords, degree_to_chord,
    FUNCTIONAL_GRAMMAR, DIATONIC_SETS, CHORD_TYPES,
)
from models.music_composition.harmonic_analysis import (
    HarmonicFunction, Chord, Progression, ModulationEvent,
    ROMAN_TO_SCALE_DEGREE, FUNCTION_MAP, TENSION_VALUES,
    parse_roman, chord_tension, functional_analysis,
    tension_arc, root_motion_profile, fifth_motion_ratio,
    harmonic_rhythm_density, harmonic_variety,
    progression_complexity, detect_cadence_type,
    detect_modulations, progression_from_symbols,
)
from models.music_composition.serial_technique import (
    ToneRow, RowForm, RowTransformation,
    validate_row, prime,
    retrograde as serial_retrograde,
    inversion as serial_inversion,
    retrograde_inversion as serial_retrograde_inversion,
    transpose_row, row_matrix,
    interval_vector, all_interval_row, combinatoriality,
    derive_row,
)
from models.music_composition.texture_planning import (
    TextureType as tp_TextureType, TextureSegment as tp_TextureSegment,
    TexturePlan, onset_density, registral_spread, registral_centroid,
    classify_texture as tp_classify_texture, voice_onset_correlation,
    plan_texture_arc, density_interpolation, voice_count_curve,
)
from models.music_composition.variation_technique import (
    VariationType, EmbellishmentType, Theme, Variation, VariationSet,
    melodic_embellishment, rhythmic_augmentation, rhythmic_diminution,
    rhythmic_displacement, dotted_rhythm, lombardic_rhythm,
    reharmonize, invertible_counterpoint, ground_bass_variations,
    theme_similarity, developing_variation, sequence_variation,
    ornamental_variation, textural_thickening,
    variation_distance, cumulative_variation_arc,
)
from models.music_composition.deep_composition import (
    MarkovChain,
    LSystem, dragon_curve_melody, sierpinski_melody,
    pink_noise, brown_noise, fractal_melody,
    elementary_ca, ca_to_melody, ca_to_rhythm,
    GeneticMelodyConfig, genetic_melody,
    interference_pattern, resultant_rhythm, complementary_rhythm,
    poisson_events, gaussian_cloud, logistic_map_sequence,
    ConstraintSpec, check_stepwise_motion, check_range, check_scale_adherence,
    constraint_generate,
    fitness_tonal_coherence, fitness_rhythmic_interest, fitness_contour_shape,
    DeepCompositionAnalysis, analyze_composition_deep,
)
