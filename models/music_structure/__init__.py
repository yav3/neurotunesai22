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
]
