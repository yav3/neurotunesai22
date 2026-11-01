"""Music production: mixing, mastering, analysis, tempo, gain staging, loudness."""
from models.music_production.mixing import (
    MixBus,
    MixerChannel,
    StereoMixer,
    GainStaging,
)
from models.music_production.mastering import (
    LoudnessMeter,
    StereoWidener,
    BrickwallLimiter,
    EQStage,
    CompressionStage,
    MasteringStage,
    MasteringChain,
)
from models.music_production.analysis import (
    SpectralBalance,
    DynamicProfile,
    StereoAnalysis,
    ReferenceComparison,
)
from models.music_production.tempo import (
    BeatGrid,
    TempoEstimator,
    GrooveTemplate,
    ClickTrack,
)

from models.music_production.eq_design import (
    EQBandType, EQBand, biquad_coefficients, apply_biquad,
    frequency_response, spectral_tilt, ParametricEQ, auto_eq_match,
)

__all__ = [
    "MixBus", "MixerChannel", "StereoMixer", "GainStaging",
    "LoudnessMeter", "StereoWidener", "BrickwallLimiter",
    "EQStage", "CompressionStage", "MasteringStage", "MasteringChain",
    "SpectralBalance", "DynamicProfile", "StereoAnalysis", "ReferenceComparison",
    "BeatGrid", "TempoEstimator", "GrooveTemplate", "ClickTrack",
    "EQBandType", "EQBand", "biquad_coefficients", "apply_biquad",
    "frequency_response", "spectral_tilt", "ParametricEQ", "auto_eq_match",
    "DynamicRangeStats", "CompressorSettings",
    "amplitude_to_db", "db_to_amplitude", "rms_level", "peak_level",
    "analyze_dynamic_range", "compression_curve", "gain_reduction",
    "apply_compression", "limiter_ceiling", "loudness_normalize",
    "multiband_split_points",
    "MasteringStageEnum", "BandConfig", "MasteringConfig", "MasteringResult",
    "apply_multiband_compression", "stereo_enhance",
    "true_peak", "mastering_limiter", "mastering_dither", "master",
    "SidechainParams", "db_to_linear", "linear_to_db",
    "envelope_follower", "compute_gain_reduction",
    "apply_sidechain", "ducking_curve", "pumping_rate",
    "stereo_correlation", "stereo_width", "stereo_balance",
    "mono_compatibility", "mid_side_ratio_db", "analyze_stereo",
    "to_mid_side", "from_mid_side", "widen_stereo",
]

from models.music_production.sidechain import (
    SidechainParams, db_to_linear, linear_to_db,
    envelope_follower, compute_gain_reduction,
    apply_sidechain, ducking_curve, pumping_rate,
)
from models.music_production.stereo_imaging import (
    stereo_correlation, stereo_width, stereo_balance,
    mono_compatibility, mid_side_ratio_db, analyze_stereo,
    to_mid_side, from_mid_side, widen_stereo,
)

from models.music_production.dynamic_range import (
    DynamicRangeStats, CompressorSettings,
    amplitude_to_db, db_to_amplitude, rms_level, peak_level,
    analyze_dynamic_range, compression_curve, gain_reduction,
    apply_compression, limiter_ceiling, loudness_normalize,
    multiband_split_points,
)
from models.music_production.mastering_chain import (
    MasteringStage as MasteringStageEnum,
    BandConfig, MasteringConfig, MasteringResult,
    apply_multiband_compression, stereo_enhance,
    true_peak, limiter as mastering_limiter,
    dither as mastering_dither, master,
)
