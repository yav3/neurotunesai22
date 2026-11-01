"""Musical instrument modeling: physical models, synthesis parameters, taxonomy, tuning."""

from .taxonomy import (
    InstrumentFamily, InstrumentClass, Instrument, INSTRUMENT_DATABASE,
    instruments_by_family, pitch_range, transposition_interval,
    is_transposing, concert_pitch_offset,
)
from .physical_models import (
    StringModel, string_frequency, string_harmonics,
    TubeModel, tube_resonance, open_pipe_harmonics, closed_pipe_harmonics,
    MembraneModel, membrane_modes,
    BarModel, bar_frequency,
)
from .synthesis_params import (
    ADSREnvelope, OscillatorType, SynthPatch,
    default_patch, fm_ratio, detune_cents_to_ratio,
    filter_cutoff_from_velocity, lfo_depth_curve,
)
from .tuning_systems import (
    TuningSystem, equal_temperament, pythagorean,
    just_intonation, meantone, well_temperament,
    cents_deviation, frequency_ratio, compare_tunings,
)

from .extended_techniques import (
    InstrumentFamily as ExtInstrumentFamily, TechniqueCategory, ExtendedTechnique,
    HarmonicNode, MultiphonicSpectrum, PreparedPianoModification,
    natural_harmonic_series, harmonic_amplitude_model, string_harmonic_nodes,
    multiphonic_roughness, technique_difficulty_score, COMMON_TECHNIQUES,
)

from .instrument_ranges import (
    InstrumentRange, INSTRUMENT_RANGES, note_in_range, comfortable_range,
    transpose_for_instrument, find_instruments_for_range, range_overlap,
)
from .articulation import (
    ArticulationType, ArticulationProfile, ARTICULATION_PROFILES,
    apply_articulation, detect_articulation, articulation_compatibility,
)

from .string_resonance import (
    StringState, harmonic_series, harmonic_amplitude,
    sympathetic_coupling, resonance_spectrum, total_sympathetic_energy,
    decay_time, inharmonicity_coefficient, stretched_partial,
    beating_frequency, ResonanceAnalysis, analyze_resonance, wolf_tone_risk,
)
from .playing_techniques import (
    TechniqueFamily, BowingTechnique, PluckingTechnique, WindTechnique,
    PercussionTechnique, SpectralModification, BOWING_SPECTRA,
    PLUCKING_SPECTRA, WIND_SPECTRA, spectral_envelope,
    apply_technique_to_spectrum,
)
from .instrument_range_db import (
    InstrumentRange as InstrumentRangeDB, INSTRUMENT_RANGES as RANGES_DB,
    range_overlap as range_overlap_db, tessitura, ensemble_range,
    instruments_in_range,
)
from .formant_models import FormantSynthesizer, ResonanceModel
from .resonance_models import (
    StringResonance, MembraneResonance, BarResonance, ResonatorBank, CoupledStringModel,
)
from .instrument_classification import (
    InstrumentFeatureExtractor, InstrumentSimilarityMatrix, InstrumentFamilyClassifier,
)

__all__ = [
    "StringState", "harmonic_series", "harmonic_amplitude",
    "sympathetic_coupling", "resonance_spectrum", "total_sympathetic_energy",
    "decay_time", "inharmonicity_coefficient", "stretched_partial",
    "beating_frequency", "ResonanceAnalysis", "analyze_resonance", "wolf_tone_risk",
    "InstrumentFamily", "InstrumentClass", "Instrument", "INSTRUMENT_DATABASE",
    "instruments_by_family", "pitch_range", "transposition_interval",
    "is_transposing", "concert_pitch_offset",
    "StringModel", "string_frequency", "string_harmonics",
    "TubeModel", "tube_resonance", "open_pipe_harmonics", "closed_pipe_harmonics",
    "MembraneModel", "membrane_modes",
    "BarModel", "bar_frequency",
    "ADSREnvelope", "OscillatorType", "SynthPatch",
    "default_patch", "fm_ratio", "detune_cents_to_ratio",
    "filter_cutoff_from_velocity", "lfo_depth_curve",
    "TuningSystem", "equal_temperament", "pythagorean",
    "just_intonation", "meantone", "well_temperament",
    "cents_deviation", "frequency_ratio", "compare_tunings",
    "InstrumentRange", "INSTRUMENT_RANGES", "note_in_range", "comfortable_range",
    "transpose_for_instrument", "find_instruments_for_range", "range_overlap",
    "ArticulationType", "ArticulationProfile", "ARTICULATION_PROFILES",
    "apply_articulation", "detect_articulation", "articulation_compatibility",
    "TechniqueCategory", "ExtendedTechnique",
    "HarmonicNode", "MultiphonicSpectrum", "PreparedPianoModification",
    "natural_harmonic_series", "harmonic_amplitude_model", "string_harmonic_nodes",
    "multiphonic_roughness", "technique_difficulty_score", "COMMON_TECHNIQUES",
    "TechniqueFamily", "BowingTechnique", "PluckingTechnique", "WindTechnique",
    "PercussionTechnique", "SpectralModification", "BOWING_SPECTRA",
    "PLUCKING_SPECTRA", "WIND_SPECTRA", "spectral_envelope",
    "apply_technique_to_spectrum",
    "InstrumentRangeDB", "RANGES_DB", "range_overlap_db", "tessitura",
    "ensemble_range", "instruments_in_range",
    "FormantSynthesizer", "ResonanceModel",
    "StringResonance", "MembraneResonance", "BarResonance", "ResonatorBank", "CoupledStringModel",
    "InstrumentFeatureExtractor", "InstrumentSimilarityMatrix", "InstrumentFamilyClassifier",
    "KarplusStrongParams", "karplus_strong", "karplus_strong_extended",
    "ReedModelParams", "reed_model", "ModalParams", "modal_synthesis",
    "pm_bar_frequencies", "pm_membrane_frequencies", "pm_bar_model", "pm_membrane_model",
    "BowedStringParams", "bowed_string", "EstimatedParams", "pm_estimate_parameters",
    "excitation_resonance_decompose",
    # bowed_string_physics (bsp_ prefix for collision-prone names)
    "BowedStringState",
    "bsp_StringParams", "bsp_string_impedance",
    "helmholtz_motion", "bridge_force_waveform", "bow_string_friction",
    "bsp_modal_frequencies", "mode_amplitudes_at_position",
    "bsp_simulate_bowed_string", "bsp_wolf_note_frequency",
    "sul_ponticello_spectrum", "col_legno_spectrum", "vibrato_modulation",
    "BowStroke", "BowParams", "bm_StringParams",
    "bow_force_envelope", "spectral_brightness", "harmonic_weighting",
    "friction_model", "pizzicato_excitation", "string_decay_time",
    "natural_harmonic_nodes", "synthesize_bowed_tone",
    "bm_vibrato_modulation", "bow_noise",
    "MaterialProperties", "ResonanceMode", "InstrumentBody",
    "SPRUCE", "MAPLE", "EBONY", "STEEL", "NYLON", "BRASS",
    "string_tension", "string_wave_speed", "body_mode_frequencies",
    "helmholtz_resonance", "soundboard_radiation", "impedance_matching",
    "coupling_efficiency", "instrument_formants", "spectral_envelope_model",
    "pp_decay_rates", "pp_sympathetic_resonance", "pp_wolf_tone_frequency",
    "radiation_directivity", "bridge_admittance",
]

from .physical_modeling import (
    KarplusStrongParams, karplus_strong, karplus_strong_extended,
    ReedModelParams, reed_model, ModalParams, modal_synthesis,
    bar_frequencies as pm_bar_frequencies,
    membrane_frequencies as pm_membrane_frequencies,
    bar_model as pm_bar_model,
    membrane_model as pm_membrane_model,
    BowedStringParams, bowed_string,
    EstimatedParams, estimate_parameters as pm_estimate_parameters,
    excitation_resonance_decompose,
)

from .bowed_string_physics import (
    BowedStringState,
    StringParams as bsp_StringParams,
    string_impedance as bsp_string_impedance,
    helmholtz_motion,
    bridge_force_waveform,
    bow_string_friction,
    modal_frequencies as bsp_modal_frequencies,
    mode_amplitudes_at_position,
    simulate_bowed_string as bsp_simulate_bowed_string,
    wolf_note_frequency as bsp_wolf_note_frequency,
    sul_ponticello_spectrum,
    col_legno_spectrum,
    vibrato_modulation,
)
from .bowing_models import (
    BowStroke, BowParams, StringParams as bm_StringParams,
    bow_force_envelope, spectral_brightness, harmonic_weighting,
    friction_model, pizzicato_excitation, string_decay_time,
    natural_harmonic_nodes, synthesize_bowed_tone,
    vibrato_modulation as bm_vibrato_modulation, bow_noise,
)
from .physical_properties import (
    MaterialProperties, ResonanceMode, InstrumentBody,
    SPRUCE, MAPLE, EBONY, STEEL, NYLON, BRASS,
    string_tension, string_wave_speed, body_mode_frequencies,
    helmholtz_resonance, soundboard_radiation, impedance_matching,
    coupling_efficiency, instrument_formants, spectral_envelope_model,
    decay_rates as pp_decay_rates, sympathetic_resonance as pp_sympathetic_resonance,
    wolf_tone_frequency as pp_wolf_tone_frequency,
    radiation_directivity, bridge_admittance,
)
