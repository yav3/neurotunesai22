"""Music IP detection: plagiarism, originality scoring, melody comparison."""
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

__all__ = [
    "pitch_sequence_similarity", "rhythm_sequence_similarity",
    "interval_contour_similarity", "transposition_invariant_similarity",
    "OriginalityScorer", "n_gram_novelty",
    "PlagiarismDetector", "PlagiarismReport",
]
