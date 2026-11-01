"""Spatial audio: HRTF, ambisonics, stereo field analysis, room modeling."""

from .hrtf import (
    HRTFParams, interaural_time_difference, interaural_level_difference,
    hrtf_filter_pair, distance_attenuation, HRTFInterpolator, SPEED_OF_SOUND,
)
from .ambisonics import (
    AmbisonicsOrder, n_channels, encode_source,
    AmbisonicsDecoder, rotate_soundfield, energy_vector,
)
from .stereo_field import (
    pan_law, stereo_width, mid_side_encode, mid_side_decode,
    stereo_balance, phantom_center_level, StereoField,
)
from .room_model import (
    RoomGeometry, ImageSource, rt60_sabine, rt60_eyring,
    image_source_model, early_decay_time, clarity_index,
    direct_to_reverberant_ratio,
)
from .binaural import (
    BinauralSource, BinauralScene, HEAD_RADIUS_M,
    itd_from_azimuth, ild_from_azimuth, crossfeed,
    render_binaural_mono, render_scene, head_shadow_filter,
)

__all__ = [
    "HRTFParams", "interaural_time_difference", "interaural_level_difference",
    "hrtf_filter_pair", "distance_attenuation", "HRTFInterpolator", "SPEED_OF_SOUND",
    "AmbisonicsOrder", "n_channels", "encode_source",
    "AmbisonicsDecoder", "rotate_soundfield", "energy_vector",
    "pan_law", "stereo_width", "mid_side_encode", "mid_side_decode",
    "stereo_balance", "phantom_center_level", "StereoField",
    "RoomGeometry", "ImageSource", "rt60_sabine", "rt60_eyring",
    "image_source_model", "early_decay_time", "clarity_index",
    "direct_to_reverberant_ratio",
    "BinauralSource", "BinauralScene", "HEAD_RADIUS_M",
    "itd_from_azimuth", "ild_from_azimuth", "crossfeed",
    "render_binaural_mono", "render_scene", "head_shadow_filter",
    "SourcePosition", "BinauralOutput",
    "itd_samples", "ild_factor", "render_binaural",
    "mix_binaural", "perceived_width",
    "RoomParameters", "AcousticParameters",
    "sabine_rt60", "eyring_rt60", "total_absorption",
    "schroeder_curve", "estimate_rt60_from_ir",
    "edt_from_ir", "ir_clarity", "definition_d50",
    "center_time", "bass_ratio",
    "image_source_reflections", "compute_acoustic_parameters",
    "BR_SourcePosition", "BinauralConfig",
    "itd_woodworth", "ild_simple", "distance_attenuation", "air_absorption",
    "hrtf_delay_samples", "apply_itd", "apply_ild",
    "spatialise_mono", "crossfeed_bauer",
    "stereo_width", "multi_source_render",
    "PanningMethod", "SpeakerLayout", "STEREO", "QUAD",
    "SURROUND_5_1", "SURROUND_7_1",
    "vbap_gains", "dbap_gains", "equal_power_pan",
    "speaker_distance_matrix", "pan_trajectory",
    "AtmosphericConditions", "air_absorption_coefficient", "apply_air_absorption",
    "doppler_factor", "doppler_shift_signal",
    "spherical_spreading_db", "cylindrical_spreading_db",
    "GroundReflection", "total_propagation_loss",
    "acn_index", "sn3d_norm", "n3d_norm", "real_spherical_harmonic",
    "encode_hoa", "encode_hoa_multi", "decode_hoa_speakers",
    "rotate_soundfield_euler",
    "sound_intensity_vector", "doa_from_intensity", "diffuseness",
    "plane_wave_decomposition", "max_re_weights", "in_phase_weights",
    "Vec3", "SpatialSource", "SpatialScene",
    "render_stereo", "render_binaural_simple",
    "Keyframe", "interpolate_trajectory", "source_velocity",
    "apparent_source_width", "listener_envelopment",
    "interaural_cross_correlation", "spatial_complexity",
    "Medium", "WavefrontSample",
    "spherical_spreading", "cylindrical_spreading",
    "apply_air_absorption", "delay_samples", "fractional_delay",
    "WaveguideMesh2D", "diffraction_edge",
    "reflection_coefficient", "absorption_to_impedance",
    "inverse_square_law", "sound_intensity_to_spl", "spl_to_intensity",
    "doppler_shift", "fresnel_zone_radius", "RayPath", "trace_ray_2d",
    "WFSArray", "WFSSource", "DrivingSignal",
    "compute_array_geometry", "planar_driving_function",
    "circular_driving_function", "focused_source_driving",
    "spatial_aliasing_frequency", "kirchhoff_helmholtz_weights",
    "render_wavefield_2d", "array_truncation_error",
    "AmbisonicSignal", "AmbisonicConfig",
    "spherical_harmonic_real", "ambisonic_encode", "ambisonic_decode",
    "sc_max_re_decode_weights", "binaural_render_ambisonic",
    "directional_loudness_map", "diffuseness_estimate", "spatial_coherence",
    "sc_energy_vector", "velocity_vector", "ambisonic_rotation",
]

from .binaural_render import (
    SourcePosition, BinauralOutput,
    itd_samples, ild_factor, render_binaural,
    mix_binaural, perceived_width,
)
from .binaural_rendering import (
    SourcePosition as BR_SourcePosition, BinauralConfig,
    itd_woodworth, ild_simple, distance_attenuation, air_absorption,
    hrtf_delay_samples, apply_itd, apply_ild,
    spatialise_mono, crossfeed as crossfeed_bauer,
    stereo_width, multi_source_render,
)
from .surround_panning import (
    PanningMethod, SpeakerLayout, STEREO, QUAD,
    SURROUND_5_1, SURROUND_7_1,
    vbap_gains, dbap_gains, equal_power_pan,
    speaker_distance_matrix, pan_trajectory,
)
from .wave_propagation import (
    Medium, WavefrontSample,
    spherical_spreading, cylindrical_spreading,
    air_absorption_coefficient, apply_air_absorption,
    delay_samples, fractional_delay,
    WaveguideMesh2D, diffraction_edge,
    reflection_coefficient, absorption_to_impedance,
    inverse_square_law, sound_intensity_to_spl, spl_to_intensity,
    doppler_shift, fresnel_zone_radius, RayPath, trace_ray_2d,
)
from .room_acoustics import (
    RoomParameters, AcousticParameters,
    sabine_rt60, eyring_rt60, total_absorption,
    schroeder_curve, estimate_rt60_from_ir,
    early_decay_time as edt_from_ir,
    clarity as ir_clarity, definition_d50, center_time, bass_ratio,
    image_source_reflections, compute_acoustic_parameters,
)
from .doppler_propagation import (
    AtmosphericConditions, air_absorption_coefficient, apply_air_absorption,
    doppler_factor, doppler_shift_signal,
    spherical_spreading_db, cylindrical_spreading_db,
    GroundReflection, total_propagation_loss,
)
from .hoa_transforms import (
    acn_index, sn3d_norm, n3d_norm, real_spherical_harmonic,
    encode_hoa, encode_hoa_multi, decode_hoa_speakers,
    rotate_soundfield_euler,
    sound_intensity_vector, doa_from_intensity, diffuseness,
    plane_wave_decomposition, max_re_weights, in_phase_weights,
)
from .wavefield_synthesis import (
    WFSArray, WFSSource, DrivingSignal,
    compute_array_geometry, planar_driving_function,
    circular_driving_function, focused_source_driving,
    spatial_aliasing_frequency, kirchhoff_helmholtz_weights,
    render_wavefield_2d, array_truncation_error,
)
from .spatial_coding import (
    AmbisonicSignal, AmbisonicConfig,
    spherical_harmonic_real, ambisonic_encode, ambisonic_decode,
    max_re_decode_weights as sc_max_re_decode_weights,
    binaural_render_ambisonic, directional_loudness_map,
    diffuseness_estimate, spatial_coherence,
    energy_vector as sc_energy_vector, velocity_vector,
    ambisonic_rotation,
)
from .spatial_scene import (
    Vec3, SpatialSource, SpatialScene,
    render_stereo, render_binaural_simple,
    Keyframe, interpolate_trajectory, source_velocity,
    apparent_source_width, listener_envelopment,
    interaural_cross_correlation, spatial_complexity,
)
