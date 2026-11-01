"""Music acoustics: psychoacoustic models, room acoustics, spatial audio, spectral analysis."""
from models.music_acoustics.psychoacoustics import (
    hz_to_mel, mel_to_hz, hz_to_bark, bark_to_hz, hz_to_erb, erb_to_hz,
    critical_bandwidth, EqualLoudnessPoint, equal_loudness_contour,
    a_weighting, c_weighting,
    MaskingResult, simultaneous_masking, temporal_masking,
    spl_to_sone, sone_to_phon, phon_to_sone,
)
from models.music_acoustics.room_acoustics import (
    RoomParameters, RT60Result, rt60_sabine, rt60_eyring, mean_free_path,
    compute_rt60, ClarityResult, clarity, early_decay_time, schroeder_curve,
    mixing_time_estimate,
    AbsorptionCoefficients, CONCRETE, CARPET, ACOUSTIC_TILE, CURTAIN_HEAVY, GLASS,
)
from models.music_acoustics.spatial_audio import (
    SPEED_OF_SOUND, HEAD_RADIUS, SpatialSource, itd, ild,
    AmbisonicsChannels, foa_encode, foa_decode_stereo, binaural_pan,
    distance_attenuation, air_absorption_db,
    StereoPosition, constant_power_pan, hrtf_delay_model,
)
from models.music_acoustics.noise_analysis import (
    NoiseProfile, NoiseGate,
    estimate_noise_floor, compute_snr, compute_snr_from_floor,
    noise_profile, spectral_subtraction, a_weighting_snr,
)
from models.music_acoustics.spectral_analysis import (
    stft, istft, mel_filterbank, mel_spectrogram,
    cqt_frequencies, constant_q_transform, chromagram,
    spectral_centroid, spectral_spread, spectral_flatness,
    spectral_rolloff, spectral_flux, compute_spectral_descriptors,
    SpectralDescriptors,
)

__all__ = [
    "hz_to_mel", "mel_to_hz", "hz_to_bark", "bark_to_hz", "hz_to_erb", "erb_to_hz",
    "critical_bandwidth", "EqualLoudnessPoint", "equal_loudness_contour",
    "a_weighting", "c_weighting",
    "MaskingResult", "simultaneous_masking", "temporal_masking",
    "spl_to_sone", "sone_to_phon", "phon_to_sone",
    "RoomParameters", "RT60Result", "rt60_sabine", "rt60_eyring", "mean_free_path",
    "compute_rt60", "ClarityResult", "clarity", "early_decay_time", "schroeder_curve",
    "mixing_time_estimate",
    "AbsorptionCoefficients", "CONCRETE", "CARPET", "ACOUSTIC_TILE", "CURTAIN_HEAVY", "GLASS",
    "SPEED_OF_SOUND", "HEAD_RADIUS", "SpatialSource", "itd", "ild",
    "AmbisonicsChannels", "foa_encode", "foa_decode_stereo", "binaural_pan",
    "distance_attenuation", "air_absorption_db",
    "StereoPosition", "constant_power_pan", "hrtf_delay_model",
    "stft", "istft", "mel_filterbank", "mel_spectrogram",
    "cqt_frequencies", "constant_q_transform", "chromagram",
    "spectral_centroid", "spectral_spread", "spectral_flatness",
    "spectral_rolloff", "spectral_flux", "compute_spectral_descriptors",
    "SpectralDescriptors",
    "NoiseProfile", "NoiseGate",
    "estimate_noise_floor", "compute_snr", "compute_snr_from_floor",
    "noise_profile", "spectral_subtraction", "a_weighting_snr",
]
