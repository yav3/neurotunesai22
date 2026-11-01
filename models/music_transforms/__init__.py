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
    "WaveletType", "ScaleogramResult", "WaveletFeatures",
    "cwt_morlet_wavelet", "mexican_hat_wavelet", "paul_wavelet",
    "cwt_transform", "cwt_scalogram",
    "wavelet_spectral_centroid", "energy_per_scale",
    "wavelet_ridge_extraction", "extract_wavelet_features",
    "dwt_decompose", "dwt_energy_distribution",
    "compute_stft_derivatives", "reassign_time_frequency",
    "reassigned_spectrogram", "synchrosqueezed_transform",
    "extract_ridges", "ridge_to_frequency",
    "GammatoneParams", "erb_space", "erb_bandwidth", "gammatone_ir",
    "apply_gammatone_bank", "bark_critical_band", "bark_space",
    "constant_q_bank", "wavelet_packet_bank", "adaptive_resolution",
    "subband_envelope", "cochleagram", "modulation_spectrum",
    "mel_gammatone_bank", "filter_energy_profile",
    "SpectralEnvelope", "cepstral_envelope", "lpc_envelope",
    "spectral_tilt", "spectral_flatness_per_band",
    "envelope_interpolation", "formant_shift",
    "ReassignedSpectrogram", "sr_reassigned_spectrogram",
    "instantaneous_frequency", "group_delay",
    "SynchrosqueezedResult", "synchrosqueeze",
    "PartialTrack", "track_spectral_peaks", "identity_phase_lock_vocoder",
    "ScatteringCoefficients", "FilterBankParams",
    "scat_morlet_wavelet", "build_wavelet_filterbank",
    "scattering_order_0", "scattering_order_1", "scattering_order_2",
    "compute_scattering", "scattering_distance",
    "time_shift_invariance_test", "scattering_modulation_spectrum",
    "joint_time_frequency_scattering",
    "SpectralFrame", "MorphParams", "MorphTrajectory",
    "morph_extract_envelope", "spectral_interpolation", "phase_vocoder_morph",
    "cross_synthesis", "timbre_distance", "morph_trajectory",
    "harmonic_matching", "frequency_warping", "spectral_blend", "morph_sequence",
]

from models.music_transforms.constant_q import (
    CQTParams, cqt_frequencies as cqt_frequencies_log, cqt_kernel, compute_cqt,
    cqt_to_chroma, cqt_spectrogram, harmonic_cqt, pitch_salience,
)
from models.music_transforms.wavelet_analysis import (
    WaveletType, ScaleogramResult, WaveletFeatures,
    morlet_wavelet as cwt_morlet_wavelet,
    mexican_hat_wavelet, paul_wavelet,
    cwt as cwt_transform, scaleogram as cwt_scalogram,
    wavelet_spectral_centroid, energy_per_scale,
    detect_ridges as wavelet_ridge_extraction,
    extract_wavelet_features,
    dwt_decompose, dwt_energy_distribution,
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
from models.music_transforms.adaptive_filterbank import (
    GammatoneParams, erb_space, erb_bandwidth, gammatone_ir,
    apply_gammatone_bank, bark_critical_band, bark_space,
    constant_q_bank, wavelet_packet_bank, adaptive_resolution,
    subband_envelope, cochleagram, modulation_spectrum,
    mel_gammatone_bank, filter_energy_profile,
)
from models.music_transforms.spectral_envelope import (
    SpectralEnvelope, cepstral_envelope, lpc_envelope,
    spectral_tilt, spectral_flatness_per_band,
    envelope_interpolation, formant_shift,
)
from models.music_transforms.spectral_reassignment import (
    ReassignedSpectrogram, reassigned_spectrogram as sr_reassigned_spectrogram,
    instantaneous_frequency, group_delay,
    SynchrosqueezedResult, synchrosqueeze,
    PartialTrack, track_spectral_peaks, identity_phase_lock_vocoder,
)
from models.music_transforms.scattering_transform import (
    ScatteringCoefficients, FilterBankParams,
    morlet_wavelet as scat_morlet_wavelet,
    build_wavelet_filterbank,
    scattering_order_0, scattering_order_1, scattering_order_2,
    compute_scattering, scattering_distance,
    time_shift_invariance_test,
    modulation_spectrum as scattering_modulation_spectrum,
    joint_time_frequency_scattering,
)
from models.music_transforms.spectral_morphing import (
    SpectralFrame, MorphParams, MorphTrajectory,
    extract_spectral_envelope as morph_extract_envelope,
    spectral_interpolation, phase_vocoder_morph,
    cross_synthesis, timbre_distance, morph_trajectory,
    harmonic_matching, frequency_warping, spectral_blend,
    morph_sequence,
)
