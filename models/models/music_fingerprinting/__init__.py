"""Music audio fingerprinting and identification algorithms."""
from models.music_fingerprinting.chromaprint import SpectralFingerprinter
from models.music_fingerprinting.landmarks import ConstellationMap, LandmarkIndex
from models.music_fingerprinting.matching import FingerprintMatcher, align_matches, score_match
from models.music_fingerprinting.hashing import MinHashFingerprint, LSHIndex

from models.music_fingerprinting.audio_codec import (
    CodecType, CodecArtifact, CodecRobustnessTest,
    simulate_codec, codec_distortion, analyze_codec_artifacts,
    quantize, pre_echo_simulation, CODEC_PARAMS,
)
from models.music_fingerprinting.robustness import (
    add_white_noise, time_stretch, pitch_shift_semitones,
    apply_lowpass, mp3_simulation, DegradationResult,
    evaluate_robustness,
)

__all__ = [
    "SpectralFingerprinter",
    "ConstellationMap",
    "LandmarkIndex",
    "FingerprintMatcher",
    "align_matches",
    "score_match",
    "MinHashFingerprint",
    "LSHIndex",
    "add_white_noise", "time_stretch", "pitch_shift_semitones",
    "apply_lowpass", "mp3_simulation", "DegradationResult",
    "evaluate_robustness",
    "CodecType", "CodecArtifact", "CodecRobustnessTest",
    "simulate_codec", "codec_distortion", "analyze_codec_artifacts",
    "quantize", "pre_echo_simulation", "CODEC_PARAMS",
    "AudioHash", "perceptual_hash", "hash_similarity",
    "content_id", "HashIndex", "bark_bands", "spectral_flatness",
]

from models.music_fingerprinting.audio_hash import (
    AudioHash, perceptual_hash, hash_similarity,
    content_id, HashIndex, bark_bands, spectral_flatness,
)
