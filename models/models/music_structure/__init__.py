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
    "SegmentNode", "checkerboard_kernel", "compute_novelty_hierarchical",
    "pick_peaks_hierarchical", "laplacian_segmentation", "build_hierarchy",
    "segment_homogeneity", "boundary_precision_recall", "form_string_hierarchical",
    "NoveltyPeak", "compute_ssm", "foote_checkerboard_kernel",
    "novelty_from_ssm", "detect_novelty_boundaries",
    "lag_ssm", "recurrence_plot", "segment_from_novelty",
    "transition_matrix",
    "Phrase", "PhraseTree", "detect_phrase_boundaries",
    "group_into_periods", "phrase_length_regularity",
    "melodic_arch_score", "build_phrase_tree",
    "sentence_structure", "hypermeter_level",
]

from models.music_structure.novelty_curve import (
    NoveltyPeak, compute_ssm, checkerboard_kernel as foote_checkerboard_kernel,
    novelty_from_ssm, detect_boundaries as detect_novelty_boundaries,
    lag_ssm, recurrence_plot, segment_from_novelty,
    transition_matrix,
)
from models.music_structure.phrase_structure import (
    Phrase, PhraseTree, detect_phrase_boundaries,
    group_into_periods, phrase_length_regularity,
    melodic_arch_score, build_phrase_tree,
    sentence_structure, hypermeter_level,
)
from models.music_structure.hierarchical_segmentation import (
    SegmentNode, checkerboard_kernel,
    compute_novelty as compute_novelty_hierarchical,
    pick_peaks as pick_peaks_hierarchical,
    laplacian_segmentation, build_hierarchy,
    segment_homogeneity, boundary_precision_recall,
    form_string as form_string_hierarchical,
)
