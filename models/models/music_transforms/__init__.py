"""Music transforms: spectral, temporal, harmonic, wavelet, filterbank."""
from models.music_transforms.spectral import (
    stft, istft, magnitude_spectrogram, power_spectrogram, phase_spectrogram,
    hz_to_mel, mel_to_hz, mel_filterbank, mel_spectrogram,
    amplitude_to_db, cqt_frequencies, pseudo_cqt, chromagram,
)
from models.music_transforms.temporal import (
    onset_strength, spectral_flux, zero_crossing_rate, rms_energy,
    autocorrelation, temporal_centroid,
)
from models.music_transforms.harmonic import (
    hpss, harmonic_product_spectrum, harmonic_to_noise_ratio,
    subharmonic_summation, autocorrelation_pitch, zero_crossing_rate as zcr_frames,
)
from models.music_transforms.wavelet import (
    cwt_morlet, scalogram, dwt_haar, idwt_haar, morlet_wavelet,
    multi_resolution_analysis, wavelet_packet_decomposition,
    wavelet_energy_bands, wavelet_entropy, scales_to_frequencies,
)
from models.music_transforms.filterbank import (
    mel_filterbank as mel_fb, bark_filterbank, erb_filterbank,
    gammatone_filterbank, gammatone_impulse_response,
    octave_band_edges, third_octave_band_edges,
    windowed_sinc_lowpass, apply_filter,
)

from models.music_transforms.cepstral import (
    compute_mfcc, compute_mfcc_full, MFCCResult,
    lifter, delta_features, delta_delta_features,
    lpc_coefficients, lpc_to_cepstrum,
    mel_filterbank_matrix, dct_ii,
)
from models.music_transforms.phase_vocoder import (
    phase_vocoder_stretch, pitch_shift, identity_phase_locking,
    detect_transients, time_stretch_with_transient_preservation,
    formant_preserving_pitch_shift, TimeStretchResult,
)

__all__ = [
    "stft", "istft", "magnitude_spectrogram", "power_spectrogram", "phase_spectrogram",
    "hz_to_mel", "mel_to_hz", "mel_filterbank", "mel_spectrogram",
    "amplitude_to_db", "cqt_frequencies", "pseudo_cqt", "chromagram",
    "onset_strength", "spectral_flux", "zero_crossing_rate", "rms_energy",
    "autocorrelation", "temporal_centroid",
    "hpss", "harmonic_product_spectrum", "harmonic_to_noise_ratio",
    "subharmonic_summation", "autocorrelation_pitch", "zcr_frames",
    "cwt_morlet", "scalogram", "dwt_haar", "idwt_haar", "morlet_wavelet",
    "multi_resolution_analysis", "wavelet_packet_decomposition",
    "wavelet_energy_bands", "wavelet_entropy", "scales_to_frequencies",
    "mel_fb", "bark_filterbank", "erb_filterbank",
    "gammatone_filterbank", "gammatone_impulse_response",
    "octave_band_edges", "third_octave_band_edges",
    "windowed_sinc_lowpass", "apply_filter",
    "phase_vocoder_stretch", "pitch_shift", "identity_phase_locking",
    "detect_transients", "time_stretch_with_transient_preservation",
    "formant_preserving_pitch_shift", "TimeStretchResult",
    "compute_mfcc", "compute_mfcc_full", "MFCCResult",
    "lifter", "delta_features", "delta_delta_features",
    "lpc_coefficients", "lpc_to_cepstrum",
    "mel_filterbank_matrix", "dct_ii",
    "CQTParams", "cqt_frequencies_log", "cqt_kernel", "compute_cqt",
    "cqt_to_chroma", "cqt_spectrogram", "harmonic_cqt", "pitch_salience",
    "ChromagramExtractor", "ChromagramResult",
    "KeyDetector", "KeyProfile", "KeyEstimate",
    "PITCH_CLASSES", "KRUMHANSL_MAJOR", "KRUMHANSL_MINOR",
    "cwt_morlet_wavelet", "cwt_transform", "cwt_scalogram", "cwt_scale_to_frequency",
    "wavelet_ridge_extraction", "instantaneous_frequency_from_cwt",
    "cross_wavelet_coherence",
    "compute_stft_derivatives", "reassign_time_frequency",
    "reassigned_spectrogram", "synchrosqueezed_transform",
    "extract_ridges", "ridge_to_frequency",
]

from models.music_transforms.constant_q import (
    CQTParams, cqt_frequencies as cqt_frequencies_log, cqt_kernel, compute_cqt,
    cqt_to_chroma, cqt_spectrogram, harmonic_cqt, pitch_salience,
)
from models.music_transforms.wavelet_analysis import (
    morlet_wavelet as cwt_morlet_wavelet, cwt_transform, scalogram as cwt_scalogram,
    scale_to_frequency as cwt_scale_to_frequency,
    wavelet_ridge_extraction, instantaneous_frequency_from_cwt,
    cross_wavelet_coherence,
)
from models.music_transforms.reassigned_spectrogram import (
    compute_stft_derivatives, reassign_time_frequency,
    reassigned_spectrogram, synchrosqueezed_transform,
    extract_ridges, ridge_to_frequency,
)
from models.music_transforms.chromagram import (
    ChromagramExtractor, ChromagramResult,
    KeyDetector, KeyProfile, KeyEstimate,
    PITCH_CLASSES, KRUMHANSL_MAJOR, KRUMHANSL_MINOR,
)
