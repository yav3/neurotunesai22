"""Music analysis: key detection, beat tracking, structure, loudness, tonality."""
from models.music_analysis.key_detection import (
    KeyProfile, KeyDetector, detect_key,
    MAJOR_PROFILE, MINOR_PROFILE, PITCH_CLASSES,
)
from models.music_analysis.beat_tracking import (
    TempoEstimator, BeatTracker, estimate_tempo, track_beats,
)
from models.music_analysis.structure import (
    StructureAnalyzer, SectionLabeler, compute_ssm, find_boundaries,
)
from models.music_analysis.loudness import (
    LoudnessAnalyzer, DynamicRangeAnalyzer, rms_loudness, lufs_approximate,
)
from models.music_analysis.tonality import (
    TonalityAnalyzer, HarmonicRhythm, chromatic_ratio, harmonic_change_rate,
)

from models.music_analysis.onset_detection import (
    OnsetMethod, Onset, OnsetDetector,
    spectral_flux as onset_spectral_flux,
    high_frequency_content, complex_domain_odf,
    energy_odf, phase_deviation_odf,
    adaptive_threshold, pick_onsets, compute_odf,
)
from models.music_analysis.structure_segmentation import (
    SegmentType, Segment, self_similarity_matrix, novelty_curve,
    detect_boundaries, label_segments, repetition_structure,
    recurrence_plot, lag_matrix,
)

__all__ = [
    "KeyProfile", "KeyDetector", "detect_key",
    "MAJOR_PROFILE", "MINOR_PROFILE", "PITCH_CLASSES",
    "TempoEstimator", "BeatTracker", "estimate_tempo", "track_beats",
    "StructureAnalyzer", "SectionLabeler", "compute_ssm", "find_boundaries",
    "LoudnessAnalyzer", "DynamicRangeAnalyzer", "rms_loudness", "lufs_approximate",
    "TonalityAnalyzer", "HarmonicRhythm", "chromatic_ratio", "harmonic_change_rate",
    "SegmentType", "Segment", "self_similarity_matrix", "novelty_curve",
    "detect_boundaries", "label_segments", "repetition_structure",
    "recurrence_plot", "lag_matrix",
    "OnsetMethod", "Onset", "OnsetDetector",
    "onset_spectral_flux", "high_frequency_content", "complex_domain_odf",
    "energy_odf", "phase_deviation_odf",
    "adaptive_threshold", "pick_onsets", "compute_odf",
]
