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
from models.music_acoustics.instrument_acoustics import (
    StringModel, PipeModel, MembraneModel, BarModel,
    BESSEL_ZEROS_01, BESSEL_ZEROS_11, FREE_BAR_NODES,
    string_overtone_series, string_decay_time,
    pipe_resonance_frequencies, membrane_modes, bar_frequencies,
    radiation_impedance, radiation_efficiency, coupling_coefficient,
    inharmonicity_factor,
)
from models.music_acoustics.propagation import (
    Medium, AIR_20C, WATER,
    propagation_delay,
    atmospheric_absorption_coefficient, atmospheric_absorption,
    ground_reflection_interference, diffraction_maekawa,
    outdoor_spl_at_distance,
    reverb_time_sabine, reverb_time_eyring, critical_distance,
)
from models.music_acoustics.reverb_models import (
    ReverbParameters as ReverbParams2,
    estimate_rt60 as estimate_rt60_schroeder,
    generate_impulse_response, apply_reverb,
    clarity_c80 as clarity_c80_direct, definition_d50 as definition_d50_direct,
    early_decay_time as edt_direct,
)
from models.music_acoustics.sound_propagation import (
    inverse_square_attenuation, air_absorption_coefficient,
    doppler_shift, interference_pattern, diffraction_coefficient,
    ground_reflection_coefficient,
)
from models.music_acoustics.reverberation import (
    ReverbParameters, EarlyReflection,
    rt60_sabine as reverb_rt60_sabine, rt60_eyring as reverb_rt60_eyring,
    mean_free_path as reverb_mean_free_path,
    mixing_time_ms, generate_early_reflections,
    late_field_density, energy_decay_curve,
    clarity_c80, definition_d50, center_time_ts,
    ReverbProfile, analyze_reverb, reverb_tail_level_db,
)
from models.music_acoustics.impulse_response import (
    ImpulseResponse, IRSynthesizer, Convolver,
)
from models.music_acoustics.acoustic_features import (
    RoomAcousticAnalyzer, SpeechIntelligibility, AcousticQualityMetrics, MaterialAbsorption,
)
from models.music_acoustics.material_absorption import (
    MaterialCoeffs, OCTAVE_BANDS, MATERIALS, Surface,
    total_absorption, weighted_nrc,
    sabine_rt60 as material_sabine_rt60, eyring_rt60 as material_eyring_rt60,
    optimal_rt60, treatment_area_needed,
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
    "StringModel", "PipeModel", "MembraneModel", "BarModel",
    "BESSEL_ZEROS_01", "BESSEL_ZEROS_11", "FREE_BAR_NODES",
    "string_overtone_series", "string_decay_time",
    "pipe_resonance_frequencies", "membrane_modes", "bar_frequencies",
    "radiation_impedance", "radiation_efficiency", "coupling_coefficient",
    "inharmonicity_factor",
    "Medium", "AIR_20C", "WATER",
    "propagation_delay",
    "atmospheric_absorption_coefficient", "atmospheric_absorption",
    "ground_reflection_interference", "diffraction_maekawa",
    "outdoor_spl_at_distance",
    "reverb_time_sabine", "reverb_time_eyring", "critical_distance",
    "estimate_rt60_schroeder", "generate_impulse_response", "apply_reverb",
    "clarity_c80_direct", "definition_d50_direct", "edt_direct",
    "inverse_square_attenuation", "air_absorption_coefficient",
    "doppler_shift", "interference_pattern", "diffraction_coefficient",
    "ground_reflection_coefficient",
    "ReverbParameters", "EarlyReflection",
    "reverb_rt60_sabine", "reverb_rt60_eyring", "reverb_mean_free_path",
    "mixing_time_ms", "generate_early_reflections",
    "late_field_density", "energy_decay_curve",
    "clarity_c80", "definition_d50", "center_time_ts",
    "ReverbProfile", "analyze_reverb", "reverb_tail_level_db",
    "MaterialCoeffs", "OCTAVE_BANDS", "MATERIALS", "Surface",
    "total_absorption", "weighted_nrc",
    "material_sabine_rt60", "material_eyring_rt60",
    "optimal_rt60", "treatment_area_needed",
    "ImpulseResponse", "IRSynthesizer", "Convolver",
    "RoomAcousticAnalyzer", "SpeechIntelligibility", "AcousticQualityMetrics",
    "MaterialAbsorption",
    "RoomDimensions", "rs_MaterialAbsorption", "RoomAcousticParams",
    "sabine_rt60", "eyring_rt60", "rs_mean_free_path",
    "image_source_rir", "rs_energy_decay_curve", "rs_rt60_from_edc",
    "rs_clarity", "rs_definition", "rs_centre_time", "rs_analyze_room",
]

from models.music_acoustics.room_simulation import (
    RoomDimensions, MaterialAbsorption as rs_MaterialAbsorption,
    RoomAcousticParams,
    sabine_rt60, eyring_rt60, mean_free_path as rs_mean_free_path,
    image_source_rir, energy_decay_curve as rs_energy_decay_curve,
    rt60_from_edc as rs_rt60_from_edc,
    clarity as rs_clarity, definition as rs_definition,
    centre_time as rs_centre_time, analyze_room as rs_analyze_room,
)
