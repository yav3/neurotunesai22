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
    "ExpandedQuery", "rocchio_expand", "tag_cooccurrence_matrix",
    "expand_by_cooccurrence", "BM25Index", "reciprocal_rank_fusion",
    "ModalityType", "RetrievalResult", "TextQuery", "AudioShingle",
    "extract_tags_from_text", "text_to_tag_vector", "compute_shingles",
    "score_chroma_from_midi", "chroma_dtw_distance",
    "text_to_audio_retrieval", "score_to_audio_retrieval",
    "reciprocal_rank", "average_precision", "ndcg_at_k",
    "FeedbackSession", "FeedbackResult", "rocchio_feedback", "ide_feedback",
    "pseudo_relevance_feedback", "query_expansion_by_feedback",
    "ListeningEvent", "implicit_relevance_score", "implicit_feedback_session",
    "diversity_reranking",
]

from .query_expansion import (
    ExpandedQuery, rocchio_expand, tag_cooccurrence_matrix,
    expand_by_cooccurrence, BM25Index, reciprocal_rank_fusion,
)
from .audio_thumbnailing import (
    compute_ssm, enhance_ssm, novelty_curve, detect_boundaries,
    repetition_score, find_thumbnail, structure_thumbnail,
    information_density, representativeness, fade_envelope,
    analyze_and_thumbnail, Thumbnail, SSMAnalysis,
)
from .audio_hashing import (
    AudioHash, spectral_band_hash, bark_scale_hash, chroma_hash,
    MinHashSignature, minhash_audio, HashIndex,
    bit_error_rate, hash_robustness,
)
from .cross_modal_retrieval import (
    ModalityType, RetrievalResult, TextQuery, AudioShingle,
    extract_tags_from_text, text_to_tag_vector, compute_shingles,
    score_chroma_from_midi, chroma_dtw_distance,
    text_to_audio_retrieval, score_to_audio_retrieval,
    reciprocal_rank, average_precision, ndcg_at_k,
)
from .relevance_feedback import (
    FeedbackSession, FeedbackResult, rocchio_feedback, ide_feedback,
    pseudo_relevance_feedback, query_expansion_by_feedback,
    ListeningEvent, implicit_relevance_score, implicit_feedback_session,
    diversity_reranking,
)
from .audio_fingerprint import (
    SpectralPeak, Landmark, FingerprintMatch, FingerprintDB,
    find_spectral_peaks, form_landmarks, hash_landmark,
    fingerprint_audio, query_fingerprints, time_coherence_score,
)
