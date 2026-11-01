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
]

from .binaural_render import (
    SourcePosition, BinauralOutput,
    itd_samples, ild_factor, render_binaural,
    mix_binaural, perceived_width,
)
from .room_acoustics import (
    RoomParameters, AcousticParameters,
    sabine_rt60, eyring_rt60, total_absorption,
    schroeder_curve, estimate_rt60_from_ir,
    early_decay_time as edt_from_ir,
    clarity as ir_clarity, definition_d50, center_time, bass_ratio,
    image_source_reflections, compute_acoustic_parameters,
)
