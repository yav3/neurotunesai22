"""Music IP detection: plagiarism, originality scoring, melody comparison, sampling detection."""
from models.music_copyright.melody_comparator import (
    pitch_sequence_similarity, rhythm_sequence_similarity,
    interval_contour_similarity, transposition_invariant_similarity,
)
from models.music_copyright.originality import (
    OriginalityScorer, n_gram_novelty,
)
from models.music_copyright.plagiarism_detector import (
    PlagiarismDetector, PlagiarismReport,
)
from models.music_copyright.fingerprint import (
    SpectralPeak, FingerprintHash, FingerprintDatabase,
    find_spectral_peaks, generate_fingerprints, fingerprint_similarity,
)
from models.music_copyright.sampling_detection import (
    SampleMatch, LoopDetection,
    cross_correlation_search, spectral_envelope, envelope_similarity,
    detect_pitch_shift, detect_time_stretch, detect_loops,
    find_sample_matches,
)
from models.music_copyright.rights_model import (
    RightsHolder, LicenseType, RightsChain, SampleClearance,
    calculate_split, mechanical_royalty_rate, streaming_royalty_estimate,
)

__all__ = [
    "pitch_sequence_similarity", "rhythm_sequence_similarity",
    "interval_contour_similarity", "transposition_invariant_similarity",
    "OriginalityScorer", "n_gram_novelty",
    "PlagiarismDetector", "PlagiarismReport",
    "SpectralPeak", "FingerprintHash", "FingerprintDatabase",
    "find_spectral_peaks", "generate_fingerprints", "fingerprint_similarity",
    "SampleMatch", "LoopDetection",
    "cross_correlation_search", "spectral_envelope", "envelope_similarity",
    "detect_pitch_shift", "detect_time_stretch", "detect_loops",
    "find_sample_matches",
    "RightsHolder", "LicenseType", "RightsChain", "SampleClearance",
    "calculate_split", "mechanical_royalty_rate", "streaming_royalty_estimate",
]
