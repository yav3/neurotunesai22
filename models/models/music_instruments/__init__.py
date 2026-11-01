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
]
