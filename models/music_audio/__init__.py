"""Music audio processing: spatial, stems, MIDI, effects."""
from models.music_audio.spatial import (
    SpatialPosition,
    SpatialPanner,
    AmbisonicsEncoder,
    RoomSimulator,
)
from models.music_audio.stems import (
    StemType,
    StemSeparator,
    SpectralStemEstimator,
    StemRemixer,
)
from models.music_audio.midi import (
    MIDINote,
    MIDITrack,
    MIDISequence,
    PianoRollConverter,
)
from models.music_audio.effects import (
    AudioEffect,
    Reverb,
    Delay,
    Compressor,
    EQBand,
    EQ,
    EffectsChain,
)

from models.music_audio.onset_detection import (
    OnsetDetector,
    OnsetType, OnsetResult,
    spectral_flux as od_spectral_flux,
    high_frequency_content as od_high_frequency_content,
    complex_domain as od_complex_domain,
    superflux, onset_strength_envelope as od_onset_strength_envelope,
    adaptive_threshold as od_adaptive_threshold,
    pick_onsets, onset_backtrack,
    classify_onset_type, transient_steady_separation,
    attack_time, compute_tempo_from_onsets,
    detect_onsets as od_detect_onsets,
)

from models.music_audio.dynamics import (
    compute_rms, compute_crest_factor, compute_dynamic_range,
    compute_loudness_envelope, detect_clipping,
    DynamicsCompressor, DynamicsLimiter,
)
from models.music_audio.onset import (
    OnsetMethod, Onset, OnsetFeatures,
    spectral_flux, high_frequency_content, complex_domain_odf,
    phase_deviation_odf, energy_odf,
    adaptive_threshold, pick_peaks, detect_onsets,
    compute_onset_features, multi_method_onset_fusion,
    onset_to_beat_alignment,
)

__all__ = [
    "SpatialPosition", "SpatialPanner", "AmbisonicsEncoder", "RoomSimulator",
    "StemType", "StemSeparator", "SpectralStemEstimator", "StemRemixer",
    "MIDINote", "MIDITrack", "MIDISequence", "PianoRollConverter",
    "AudioEffect", "Reverb", "Delay", "Compressor", "EQBand", "EQ", "EffectsChain",
    "compute_rms", "compute_crest_factor", "compute_dynamic_range",
    "compute_loudness_envelope", "detect_clipping",
    "DynamicsCompressor", "DynamicsLimiter",
    "OnsetMethod", "Onset", "OnsetFeatures",
    "spectral_flux", "high_frequency_content", "complex_domain_odf",
    "phase_deviation_odf", "energy_odf",
    "adaptive_threshold", "pick_peaks", "detect_onsets",
    "compute_onset_features", "multi_method_onset_fusion", "onset_to_beat_alignment",
    "OnsetDetector",
    "OnsetType", "OnsetResult",
    "od_spectral_flux", "od_high_frequency_content",
    "od_complex_domain", "superflux", "od_onset_strength_envelope",
    "od_adaptive_threshold", "pick_onsets", "onset_backtrack",
    "classify_onset_type", "transient_steady_separation",
    "attack_time", "compute_tempo_from_onsets", "od_detect_onsets",
    "ideal_binary_mask", "ideal_ratio_mask", "wiener_filter",
    "harmonic_percussive_separation", "griffin_lim",
    "SDR", "SIR", "SAR",
    "AudioFormat", "mu_law_encode", "mu_law_decode",
    "a_law_encode", "a_law_decode", "float_to_pcm", "pcm_to_float",
    "triangular_dither", "noise_shape_first_order",
    "resample_sinc", "resample_linear",
    "split_channels", "interleave_channels",
    "compute_rms_db", "compute_peak_db",
    "PitchResult", "pd_hz_to_midi", "pd_midi_to_hz",
    "pd_autocorrelation_pitch", "yin_pitch",
    "pd_harmonic_product_spectrum", "pitch_track",
    "median_pitch_filter", "voiced_unvoiced_decision",
    "pitch_stability", "pd_vibrato_detection", "pitch_histogram",
    "FilterType", "sp_EQBand", "biquad_coefficients", "apply_biquad",
    "parametric_eq", "spectral_whitening", "spectral_gate",
    "sp_spectral_centroid", "sp_spectral_spread", "sp_spectral_flux",
    "sp_spectral_rolloff", "cross_synthesis", "spectral_freeze",
    "harmonic_percussive_ratio",
]

from models.music_audio.source_separation import (
    ideal_binary_mask, ideal_ratio_mask, wiener_filter,
    harmonic_percussive_separation, griffin_lim,
    SDR, SIR, SAR,
)
from models.music_audio.audio_codec import (
    AudioFormat, mu_law_encode, mu_law_decode,
    a_law_encode, a_law_decode, float_to_pcm, pcm_to_float,
    triangular_dither, noise_shape_first_order,
    resample_sinc, resample_linear,
    split_channels, interleave_channels,
    compute_rms_db, compute_peak_db,
)
from models.music_audio.pitch_detection import (
    PitchResult, hz_to_midi as pd_hz_to_midi, midi_to_hz as pd_midi_to_hz,
    autocorrelation_pitch as pd_autocorrelation_pitch,
    yin_pitch, harmonic_product_spectrum as pd_harmonic_product_spectrum,
    pitch_track, median_pitch_filter, voiced_unvoiced_decision,
    pitch_stability, vibrato_detection as pd_vibrato_detection,
    pitch_histogram,
)
from models.music_audio.audio_restoration import (
    NoiseProfile, estimate_noise_profile, spectral_subtraction,
    wiener_filter as wiener_filter_restore,
    detect_clipping as detect_clipping_restore,
    declip_cubic, bandwidth_extension,
    estimate_reverb_tail, dereverberate, adaptive_noise_gate,
)
from models.music_audio.spectral_processing import (
    FilterType, EQBand as sp_EQBand, biquad_coefficients, apply_biquad,
    parametric_eq, spectral_whitening, spectral_gate,
    spectral_centroid as sp_spectral_centroid,
    spectral_spread as sp_spectral_spread,
    spectral_flux as sp_spectral_flux,
    spectral_rolloff as sp_spectral_rolloff,
    cross_synthesis, spectral_freeze, harmonic_percussive_ratio,
)
