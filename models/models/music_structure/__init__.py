"""Music structure analysis: segmentation, form, hierarchy, transitions."""
from models.music_structure.segmentation import (
    SelfSimilarityMatrix,
    RecurrenceMatrix,
    novelty_curve,
    segment_boundaries,
)
from models.music_structure.form import FormAnalyzer, FormTemplate
from models.music_structure.hierarchy import StructureTree, LagMatrix
from models.music_structure.transitions import TransitionDetector, TransitionGraph
from models.music_structure.self_similarity import (
    cosine_similarity_matrix as ssm_cosine,
    euclidean_distance_matrix, gaussian_similarity_matrix,
    recurrence_matrix as ssm_recurrence_matrix,
    novelty_curve as ssm_novelty_curve,
    find_boundaries, lag_similarity_matrix,
    enhance_diagonals, StructureSegment, label_segments,
    StructureAnalysis, analyze_structure,
)
from models.music_structure.repetition_analysis import (
    Motif, MotifOccurrence, RepetitionStructure,
    cosine_similarity_matrix, lag_similarity, lag_spectrum,
    find_repeated_segments, structural_thumbnail,
    compression_ratio as structural_compression_ratio,
    build_repetition_structure,
)

__all__ = [
    "SelfSimilarityMatrix",
    "RecurrenceMatrix",
    "novelty_curve",
    "segment_boundaries",
    "FormAnalyzer",
    "FormTemplate",
    "StructureTree",
    "LagMatrix",
    "TransitionDetector",
    "TransitionGraph",
    "Motif", "MotifOccurrence", "RepetitionStructure",
    "cosine_similarity_matrix", "lag_similarity", "lag_spectrum",
    "find_repeated_segments", "structural_thumbnail",
    "structural_compression_ratio",
    "build_repetition_structure",
    "ssm_cosine", "euclidean_distance_matrix", "gaussian_similarity_matrix",
    "ssm_recurrence_matrix", "ssm_novelty_curve",
    "find_boundaries", "lag_similarity_matrix",
    "enhance_diagonals", "StructureSegment", "label_segments",
    "StructureAnalysis", "analyze_structure",
]
