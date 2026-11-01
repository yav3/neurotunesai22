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
    GrainEnvelope,
    EnvelopeShape,
    GranularSynthesizer,
)

# Aliases expected by tests
GrainCloud = GranularSynthesizer


class GranularFreeze:
    """Convenience wrapper around GranularSynthesizer.freeze."""
    def __init__(self, source: "np.ndarray", position: int = 0,
                 grain_size: int = 2048, spread: int = 0):
        self.source = source
        self.position = position
        self.grain_size = grain_size
        self.spread = spread

    def render(self, n_grains: int = 50, seed: int = 42) -> "np.ndarray":
        return GranularSynthesizer.freeze(
            self.source, self.position, self.grain_size, n_grains, self.spread, seed
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
from .subtractive import (
    FilterType,
    biquad_coefficients,
    apply_biquad,
    apply_biquad_tv,
    FilterEnvelope,
    LFO,
    SubtractiveSynth,
)
from .spectral_synthesis import (
    stft_frames,
    detect_peaks,
    SinusoidalPeak,
    SinusoidalTrack,
    track_sinusoids,
    resynthesize_sinusoids,
    extract_residual,
    spectral_morph,
    sms_analyze,
)
from .granular_synthesis import (
    GrainParams, generate_grain, grain_cloud,
    time_stretch_granular, pitch_shift_granular, granular_freeze,
)
from .physical_modeling import (
    karplus_strong, damped_string, bowed_string,
    flute_waveguide, drum_membrane,
)
from .modular_routing import (
    SignalType, Module, Port, ModulePort, PatchMatrix, PatchCable,
    cv_to_frequency, frequency_to_cv, quantize_cv, cv_offset, cv_scale,
    gate_from_cv, trigger_from_gate, clock_signal, clock_divider, clock_multiplier,
    gate_logic_and, gate_logic_or, gate_logic_xor,
    slew_limiter, sample_and_hold, voltage_controlled_mix,
    ring_modulator, rectifier, wavefolder, envelope_follower,
)

__all__ = [
    "Oscillator", "SineOscillator", "SawtoothOscillator", "SquareOscillator",
    "TriangleOscillator", "NoiseGenerator", "NoiseType", "PulseWidthModulator",
    "ADSREnvelope", "FMOperator", "FMAlgorithm", "FMPatch", "FMSynthesizer",
    "AdditivePartial", "AdditiveSpectrum", "AdditiveSynthesizer", "SpectralMorphing",
    "Grain", "GrainEnvelope", "EnvelopeShape", "GranularSynthesizer",
    "GrainCloud", "GranularFreeze",
    "Wavetable", "WavetableBank", "WavetableOscillator", "WavetableSynthesizer",
    "KarplusStrong", "WaveguideString", "ModalSynthesizer", "TubeSynthesizer",
    "FilterType", "biquad_coefficients", "apply_biquad", "apply_biquad_tv",
    "FilterEnvelope", "LFO", "SubtractiveSynth",
    "stft_frames", "detect_peaks", "SinusoidalPeak", "SinusoidalTrack",
    "track_sinusoids", "resynthesize_sinusoids", "extract_residual",
    "spectral_morph", "sms_analyze",
    "GrainParams", "generate_grain", "grain_cloud",
    "time_stretch_granular", "pitch_shift_granular", "granular_freeze",
    "karplus_strong", "damped_string", "bowed_string",
    "flute_waveguide", "drum_membrane",
]
