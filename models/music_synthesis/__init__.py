"""Music synthesis: oscillators, FM, additive, granular, wavetable, physical modeling."""
from .oscillators import (
    Oscillator,
    SineOscillator,
    SawtoothOscillator,
    SquareOscillator,
    TriangleOscillator,
    NoiseGenerator,
    NoiseType,
    PulseWidthModulator,
)
from .fm_synthesis import (
    ADSREnvelope,
    FMOperator,
    FMAlgorithm,
    FMPatch,
    FMSynthesizer,
)
from .additive import (
    AdditivePartial,
    AdditiveSpectrum,
    AdditiveSynthesizer,
    SpectralMorphing,
)
from .granular import (
    Grain,
    GrainCloud,
    GranularSynthesizer,
    GranularFreeze,
)
from .wavetable import (
    Wavetable,
    WavetableBank,
    WavetableOscillator,
    WavetableSynthesizer,
)
from .physical_model import (
    KarplusStrong,
    WaveguideString,
    ModalSynthesizer,
    TubeSynthesizer,
)

__all__ = [
    "Oscillator", "SineOscillator", "SawtoothOscillator", "SquareOscillator",
    "TriangleOscillator", "NoiseGenerator", "NoiseType", "PulseWidthModulator",
    "ADSREnvelope", "FMOperator", "FMAlgorithm", "FMPatch", "FMSynthesizer",
    "AdditivePartial", "AdditiveSpectrum", "AdditiveSynthesizer", "SpectralMorphing",
    "Grain", "GrainCloud", "GranularSynthesizer", "GranularFreeze",
    "Wavetable", "WavetableBank", "WavetableOscillator", "WavetableSynthesizer",
    "KarplusStrong", "WaveguideString", "ModalSynthesizer", "TubeSynthesizer",
]
