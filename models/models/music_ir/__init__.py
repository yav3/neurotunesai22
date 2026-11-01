"""Music Information Retrieval: fingerprinting, chroma, onset, search, tagging, segmentation."""

from .chromagram import (
    CHORD_TEMPLATES,
    KRUMHANSL_KESSLER_MAJOR,
    KRUMHANSL_KESSLER_MINOR,
    PITCH_CLASSES,
    ChordRecognizer,
    ChromaDistance,
    ChromaExtractor,
    KeyDetector,
)
from .fingerprint import (
    AudioFingerprinter,
    ConstellationMap,
    FingerprintHash,
    SpectrogramPeaks,
)
from .onset import (
    BeatTracker,
    OnsetDetector,
    SegmentBoundaryDetector,
    TempoHistogram,
)
from .similarity_search import (
    CoverSongDetector,
    FeatureIndex,
    LocalitySensitiveHash,
    SequenceAlignment,
)
from .search import (
    InvertedIndex,
    TrackIndex,
    QueryExpander,
    SearchRanker,
)
from .tagging import (
    AutoTagger,
    GenreClassifier,
    MoodTagger,
    InstrumentDetector,
)
from .segmentation import (
    StructuralSegmenter,
    ChromaAnalyzer,
)

__all__ = [
    "AudioFingerprinter", "BeatTracker", "CHORD_TEMPLATES",
    "ChordRecognizer", "ChromaDistance", "ChromaExtractor",
    "ConstellationMap", "CoverSongDetector", "FeatureIndex",
    "FingerprintHash", "KeyDetector", "KRUMHANSL_KESSLER_MAJOR",
    "KRUMHANSL_KESSLER_MINOR", "LocalitySensitiveHash",
    "OnsetDetector", "PITCH_CLASSES", "SegmentBoundaryDetector",
    "SequenceAlignment", "SpectrogramPeaks", "TempoHistogram",
    "InvertedIndex", "TrackIndex", "QueryExpander", "SearchRanker",
    "AutoTagger", "GenreClassifier", "MoodTagger", "InstrumentDetector",
    "StructuralSegmenter", "ChromaAnalyzer",
]
