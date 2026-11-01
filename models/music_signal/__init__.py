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
]

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
