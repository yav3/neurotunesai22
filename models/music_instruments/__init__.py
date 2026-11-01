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

__all__ = [
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
]
