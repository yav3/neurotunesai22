"""Digital signal processing fundamentals for music: windowing, filters, resampling, dynamics, spectral analysis."""

from .windowing import (
    WindowType, window_function, overlap_add_window,
    equivalent_noise_bandwidth, scalloping_loss, coherent_gain,
    cola_check, optimal_hop, apply_window,
)
from .filters import (
    FilterType, BiquadCoeffs, biquad_design, apply_biquad,
    cascade_response, parametric_eq, a_weighting, c_weighting,
)
from .resampling import (
    sample_rate_ratio, resample_sinc, decimate,
    interpolate_linear, interpolate_cubic, fractional_delay,
)
from .dynamics import (
    CompressorParams, compute_gain_reduction, envelope_follower,
    compressor, limiter, gate, rms_level, peak_level,
    lufs_momentary, crest_factor, dynamic_range,
)
from .spectral_tools import (
    stft, istft, magnitude_to_db, db_to_magnitude,
    hz_to_mel, mel_to_hz, mel_filterbank, cqt_frequencies,
    spectral_centroid, spectral_bandwidth, spectral_contrast,
    spectral_rolloff, chromagram,
)

__all__ = [
    "WindowType", "window_function", "overlap_add_window",
    "equivalent_noise_bandwidth", "scalloping_loss", "coherent_gain",
    "cola_check", "optimal_hop", "apply_window",
    "FilterType", "BiquadCoeffs", "biquad_design", "apply_biquad",
    "cascade_response", "parametric_eq", "a_weighting", "c_weighting",
    "sample_rate_ratio", "resample_sinc", "decimate",
    "interpolate_linear", "interpolate_cubic", "fractional_delay",
    "CompressorParams", "compute_gain_reduction", "envelope_follower",
    "compressor", "limiter", "gate", "rms_level", "peak_level",
    "lufs_momentary", "crest_factor", "dynamic_range",
    "stft", "istft", "magnitude_to_db", "db_to_magnitude",
    "hz_to_mel", "mel_to_hz", "mel_filterbank", "cqt_frequencies",
    "spectral_centroid", "spectral_bandwidth", "spectral_contrast",
    "spectral_rolloff", "chromagram",
    "spectral_flux", "high_frequency_content", "complex_domain",
    "energy_onset", "pick_peaks", "frames_to_time", "onset_detect", "OnsetResult",
    "hz_to_midi", "midi_to_hz", "pitch_class", "cents_difference", "note_name",
    "autocorrelation_pitch", "yin_frame", "track_pitch", "harmonic_product_spectrum",
    "median_filter_pitch", "PitchResult",
    "compute_mfcc", "mfcc_delta", "mfcc_delta_delta", "lifter", "MFCCResult",
    "analytic_signal", "instantaneous_amplitude", "instantaneous_phase",
    "instantaneous_frequency", "envelope_follower_hilbert",
    "rms_envelope", "AMFMDecomposition", "am_fm_decompose",
    "tremolo_rate", "tremolo_depth", "envelope_centroid",
    "attack_time", "decay_time",
    "AdaptiveFilterType", "AdaptiveFilterState",
    "lms_update", "nlms_update", "rls_update", "kalman_filter_step",
    "noise_cancellation", "echo_cancellation",
    "system_identification", "compute_misalignment",
    "affine_projection",
]

from .harmonic_product_spectrum import (
    PitchEstimate as HPSPitchEstimate,
    harmonic_product_spectrum as hps_pitch,
    autocorrelation_pitch as ac_pitch,
    cepstral_pitch,
    multi_pitch_estimate,
    consensus_pitch,
)
from .adaptive_filtering import (
    FilterType as AdaptiveFilterType,
    AdaptiveFilterState,
    lms_update, nlms_update, rls_update, kalman_filter_step,
    noise_cancellation, echo_cancellation,
    system_identification, compute_misalignment,
    affine_projection,
)

from .onset_detection import (
    spectral_flux, high_frequency_content, complex_domain,
    energy_onset, pick_peaks, frames_to_time, onset_detect, OnsetResult,
)
from .pitch_tracking import (
    hz_to_midi, midi_to_hz, pitch_class, cents_difference, note_name,
    autocorrelation_pitch, yin_frame, track_pitch, harmonic_product_spectrum,
    median_filter_pitch, PitchResult,
)
from .mfcc import (
    compute_mfcc, mfcc_delta, mfcc_delta_delta, lifter, MFCCResult,
)
from .envelope import (
    analytic_signal, instantaneous_amplitude, instantaneous_phase,
    instantaneous_frequency,
    envelope_follower as envelope_follower_hilbert,
    rms_envelope, AMFMDecomposition, am_fm_decompose,
    tremolo_rate, tremolo_depth, envelope_centroid,
    attack_time, decay_time,
)
from .signal_quality import (
    QualityReport, snr, snr_from_noisy, segmental_snr,
    thd, thd_plus_n, sinad, enob,
    noise_floor_estimate, dynamic_range_from_signal,
    crest_factor as crest_factor_quality,
    dc_offset, inter_channel_phase, phase_correlation,
    spectral_distortion, itakura_saito_distance,
    clipping_detection, quality_report,
)
from .source_separation import (
    SeparationResult, nmf, nmf_separate,
    hpss, wiener_filter, spectral_subtraction,
    ideal_binary_mask, ideal_ratio_mask,
    sdr, sir, sar, harmonic_mask,
)
