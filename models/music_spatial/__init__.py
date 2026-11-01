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
]
