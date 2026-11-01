"""Music audio fingerprinting and identification algorithms."""
from models.music_fingerprinting.chromaprint import SpectralFingerprinter
from models.music_fingerprinting.landmarks import ConstellationMap, LandmarkIndex
from models.music_fingerprinting.matching import FingerprintMatcher, align_matches, score_match
from models.music_fingerprinting.hashing import MinHashFingerprint, LSHIndex

__all__ = [
    "SpectralFingerprinter",
    "ConstellationMap",
    "LandmarkIndex",
    "FingerprintMatcher",
    "align_matches",
    "score_match",
    "MinHashFingerprint",
    "LSHIndex",
]
