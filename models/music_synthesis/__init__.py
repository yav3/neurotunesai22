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

from .filter_design import (
    FilterType as fd_FilterType, BiquadCoeffs, FilterState, SVFState,
    biquad_lowpass as fd_biquad_lowpass, biquad_highpass as fd_biquad_highpass,
    biquad_bandpass as fd_biquad_bandpass, biquad_notch as fd_biquad_notch,
    biquad_peak as fd_biquad_peak,
    biquad_low_shelf as fd_biquad_low_shelf, biquad_high_shelf as fd_biquad_high_shelf,
    apply_biquad as fd_apply_biquad, cascade_biquads, svf_tick, svf_filter,
    moog_ladder, parametric_eq as fd_parametric_eq,
    frequency_response, filter_envelope,
)
from .envelope_generator import (
    EnvelopeStage, LFOWaveform, ADSRParams, DAHDSRParams, LFOParams,
    ModulationRoute, ModulationMatrix,
    adsr_envelope, dahdsr_envelope, exponential_adsr, multistage_envelope,
    lfo_signal, portamento, legato_portamento,
    velocity_curve, key_tracking, aftertouch_to_mod,
)
from .advanced_granular import (
    GrainEnvelopeType, GrainSpec, StereoCloud, CloudStatistics,
    envelope_function, extract_grain, pitch_shift_grain,
    synthesize_grain, render_cloud,
    synchronous_granular, asynchronous_granular,
    granular_time_stretch as adv_granular_time_stretch,
    granular_freeze as adv_granular_freeze,
    spectral_grain, grain_density_curve, cloud_statistics,
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
    "GrainEnvelopeType", "GrainSpec", "StereoCloud", "CloudStatistics",
    "envelope_function", "extract_grain", "pitch_shift_grain",
    "synthesize_grain", "render_cloud",
    "synchronous_granular", "asynchronous_granular",
    "adv_granular_time_stretch", "adv_granular_freeze",
    "spectral_grain", "grain_density_curve", "cloud_statistics",
    "SignalType", "Module", "Port", "ModulePort", "PatchMatrix", "PatchCable",
    "cv_to_frequency", "frequency_to_cv", "quantize_cv", "cv_offset", "cv_scale",
    "gate_from_cv", "trigger_from_gate", "clock_signal", "clock_divider", "clock_multiplier",
    "gate_logic_and", "gate_logic_or", "gate_logic_xor",
    "slew_limiter", "sample_and_hold", "voltage_controlled_mix",
    "ring_modulator", "rectifier", "wavefolder", "envelope_follower",
    "fd_FilterType", "BiquadCoeffs", "FilterState", "SVFState",
    "fd_biquad_lowpass", "fd_biquad_highpass", "fd_biquad_bandpass",
    "fd_biquad_notch", "fd_biquad_peak",
    "fd_biquad_low_shelf", "fd_biquad_high_shelf",
    "fd_apply_biquad", "cascade_biquads", "svf_tick", "svf_filter",
    "moog_ladder", "fd_parametric_eq", "frequency_response", "filter_envelope",
    "EnvelopeStage", "LFOWaveform", "ADSRParams", "DAHDSRParams", "LFOParams",
    "ModulationRoute", "ModulationMatrix",
    "adsr_envelope", "dahdsr_envelope", "exponential_adsr", "multistage_envelope",
    "lfo_signal", "portamento", "legato_portamento",
    "velocity_curve", "key_tracking", "aftertouch_to_mod",
    "FormantFreqs", "VOWEL_FORMANTS", "glottal_pulse", "aspiration_noise",
    "formant_filter", "interpolate_formants", "vowel_trajectory",
    "vibrato", "singing_synthesis", "formant_analysis", "vocal_tract_length_estimate",
]

from .formant_synthesis import (
    FormantFreqs, VOWEL_FORMANTS, glottal_pulse, aspiration_noise,
    formant_filter, interpolate_formants, vowel_trajectory,
    vibrato, singing_synthesis, formant_analysis, vocal_tract_length_estimate,
)
