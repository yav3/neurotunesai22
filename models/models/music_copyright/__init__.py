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
    SpectralPeak, FingerprintHash,
    FingerprintDatabase as SpectralFingerprintDatabase,
    find_spectral_peaks, generate_fingerprints, fingerprint_similarity,
)
from models.music_copyright.sampling_detection import (
    SampleMatch, LoopDetection,
    cross_correlation_search, spectral_envelope, envelope_similarity,
    detect_pitch_shift, detect_time_stretch, detect_loops,
    find_sample_matches,
)
from models.music_copyright.rights_model import (
    RightsHolder, RightsChain, SampleClearance,
    LicenseType as RightsLicenseType,
    calculate_split, mechanical_royalty_rate, streaming_royalty_estimate,
)

from models.music_copyright.content_id import (
    AudioFingerprint, ContentIDResult, spectral_hash,
    extract_landmarks, match_landmarks, FingerprintDatabase,
)
from models.music_copyright.license_model import (
    LicenseType, LicenseInfo, LICENSE_PROPERTIES,
    check_compatibility, most_restrictive, attribution_text,
)
from models.music_copyright.harmonic_similarity import (
    chord_to_pc, chords_to_intervals, chords_to_roman,
    harmonic_sequence_similarity, transposition_invariant_harmonic_similarity,
    functional_reduction, functional_similarity,
    chord_ngram_overlap, ngram_significance,
    HarmonicMatchSegment, sliding_harmonic_compare,
)
from models.music_copyright.rhythm_fingerprint import (
    onsets_to_grid, grid_hamming_similarity,
    metric_weight_vector, weighted_onset_similarity,
    RhythmicMotif, extract_motifs, motif_jaccard,
    ioi_vector, ioi_ratio_vector, ioi_ratio_similarity,
    cross_rhythm_correlation, RhythmComparisonResult, compare_rhythms,
)
from models.music_copyright.public_domain import (
    Jurisdiction, WorkType, CopyrightQuery, CopyrightStatus,
    determine_copyright_status, is_public_domain,
)
from models.music_copyright.fair_use_analysis import (
    Purpose, FairUseFactors, composite_fair_use_score, fair_use_risk_level,
    analyze_fair_use,
)
from models.music_copyright.royalty_calculation import (
    mechanical_royalty, streaming_royalty, SyncLicenseTerms, compute_statement,
)

__all__ = [
    "pitch_sequence_similarity", "rhythm_sequence_similarity",
    "interval_contour_similarity", "transposition_invariant_similarity",
    "OriginalityScorer", "n_gram_novelty",
    "PlagiarismDetector", "PlagiarismReport",
    "SpectralPeak", "FingerprintHash", "SpectralFingerprintDatabase",
    "find_spectral_peaks", "generate_fingerprints", "fingerprint_similarity",
    "SampleMatch", "LoopDetection",
    "cross_correlation_search", "spectral_envelope", "envelope_similarity",
    "detect_pitch_shift", "detect_time_stretch", "detect_loops",
    "find_sample_matches",
    "RightsHolder", "RightsLicenseType", "RightsChain", "SampleClearance",
    "calculate_split", "mechanical_royalty_rate", "streaming_royalty_estimate",
    "AudioFingerprint", "ContentIDResult", "spectral_hash",
    "extract_landmarks", "match_landmarks", "FingerprintDatabase",
    "LicenseType", "LicenseInfo", "LICENSE_PROPERTIES",
    "check_compatibility", "most_restrictive", "attribution_text",
    "chord_to_pc", "chords_to_intervals", "chords_to_roman",
    "harmonic_sequence_similarity", "transposition_invariant_harmonic_similarity",
    "functional_reduction", "functional_similarity",
    "chord_ngram_overlap", "ngram_significance",
    "HarmonicMatchSegment", "sliding_harmonic_compare",
    "onsets_to_grid", "grid_hamming_similarity",
    "metric_weight_vector", "weighted_onset_similarity",
    "RhythmicMotif", "extract_motifs", "motif_jaccard",
    "ioi_vector", "ioi_ratio_vector", "ioi_ratio_similarity",
    "cross_rhythm_correlation", "RhythmComparisonResult", "compare_rhythms",
    "Jurisdiction", "WorkType", "CopyrightQuery", "CopyrightStatus",
    "determine_copyright_status", "is_public_domain",
    "Purpose", "FairUseFactors", "composite_fair_use_score", "fair_use_risk_level",
    "analyze_fair_use",
    "mechanical_royalty", "streaming_royalty", "SyncLicenseTerms", "compute_statement",
]
