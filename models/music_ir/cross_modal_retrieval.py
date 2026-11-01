"""Cross-modal music retrieval: text-to-audio, score-to-audio, humming-to-MIDI alignment."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal as sig


class ModalityType(Enum):
    AUDIO = "audio"
    TEXT = "text"
    SCORE = "score"
    HUMMING = "humming"
    MIDI = "midi"


@dataclass
class RetrievalResult:
    query_modality: ModalityType
    target_modality: ModalityType
    scores: np.ndarray      # similarity scores for each candidate
    ranked_indices: np.ndarray  # candidate indices sorted by score desc
    top_score: float


@dataclass
class TextQuery:
    raw_text: str
    tokens: list[str]
    tag_weights: dict[str, float]  # extracted musical tags with confidence


@dataclass
class AudioShingle:
    """Short overlapping audio feature segment for cross-modal matching."""
    start_frame: int
    length: int
    chroma: np.ndarray       # 12-dim chroma vector (mean over shingle)
    mfcc: np.ndarray         # 13-dim MFCC vector (mean over shingle)
    energy: float


# --- Tag extraction from text queries ---

MUSICAL_TAG_LEXICON: dict[str, list[str]] = {
    "happy": ["happy", "joy", "cheerful", "upbeat", "bright", "fun", "playful"],
    "sad": ["sad", "melancholy", "sorrowful", "blue", "crying", "tears", "lonely"],
    "energetic": ["energetic", "fast", "driving", "powerful", "intense", "pumping"],
    "calm": ["calm", "relaxing", "peaceful", "gentle", "soft", "quiet", "ambient"],
    "dark": ["dark", "heavy", "aggressive", "angry", "gloomy", "menacing"],
    "romantic": ["romantic", "love", "tender", "intimate", "passionate", "sensual"],
    "electronic": ["electronic", "synth", "techno", "house", "edm", "digital"],
    "acoustic": ["acoustic", "unplugged", "folk", "natural", "organic", "wooden"],
    "orchestral": ["orchestral", "classical", "symphony", "strings", "brass"],
    "vocal": ["vocal", "singing", "choir", "acapella", "harmonies", "voice"],
    "instrumental": ["instrumental", "solo", "no vocals", "without words"],
    "rhythmic": ["rhythmic", "groovy", "danceable", "beat", "percussion", "drums"],
}


def extract_tags_from_text(query: str) -> TextQuery:
    """Extract musical tags and weights from a natural-language query."""
    tokens = re.findall(r"[a-z]+", query.lower())
    tag_weights: dict[str, float] = {}

    for tag, keywords in MUSICAL_TAG_LEXICON.items():
        weight = 0.0
        for kw in keywords:
            kw_tokens = kw.split()
            for i in range(len(tokens) - len(kw_tokens) + 1):
                if tokens[i:i + len(kw_tokens)] == kw_tokens:
                    weight += 1.0
        if weight > 0:
            tag_weights[tag] = min(weight, 3.0) / 3.0  # normalise to [0,1]

    # Normalise so weights sum to 1
    total = sum(tag_weights.values())
    if total > 0:
        tag_weights = {k: v / total for k, v in tag_weights.items()}

    return TextQuery(query, tokens, tag_weights)


def text_to_tag_vector(query: TextQuery, tag_order: list[str] | None = None) -> np.ndarray:
    """Convert a text query to a tag-space vector."""
    if tag_order is None:
        tag_order = sorted(MUSICAL_TAG_LEXICON.keys())
    vec = np.zeros(len(tag_order))
    for i, tag in enumerate(tag_order):
        vec[i] = query.tag_weights.get(tag, 0.0)
    return vec


# --- Audio feature shingles ---

def compute_shingles(chroma: np.ndarray, mfcc: np.ndarray,
                     shingle_len: int = 10, hop: int = 5) -> list[AudioShingle]:
    """Create overlapping feature shingles from frame-level features.

    Args:
        chroma: (12, T) chroma features
        mfcc: (n_mfcc, T) MFCC features
        shingle_len: frames per shingle
        hop: hop between shingle starts
    """
    T = min(chroma.shape[1] if chroma.ndim > 1 else len(chroma),
            mfcc.shape[1] if mfcc.ndim > 1 else len(mfcc))

    if chroma.ndim == 1:
        chroma = chroma.reshape(12, -1)
    if mfcc.ndim == 1:
        mfcc = mfcc.reshape(-1, T)

    shingles: list[AudioShingle] = []
    for start in range(0, T - shingle_len + 1, hop):
        end = start + shingle_len
        c_mean = chroma[:, start:end].mean(axis=1)
        m_mean = mfcc[:, start:end].mean(axis=1)
        energy = float(np.sum(chroma[:, start:end] ** 2))
        shingles.append(AudioShingle(start, shingle_len, c_mean, m_mean, energy))

    return shingles


def shingle_to_vector(shingle: AudioShingle) -> np.ndarray:
    """Concatenate shingle features into a single vector."""
    return np.concatenate([shingle.chroma, shingle.mfcc, [shingle.energy]])


# --- Score-to-audio alignment (chroma-based DTW) ---

def score_chroma_from_midi(note_events: list[tuple[float, float, int]],
                           sr: float = 22050.0,
                           hop_length: int = 512) -> np.ndarray:
    """Synthesise a chroma representation from MIDI-like note events.

    Each event is (onset_sec, duration_sec, midi_pitch).
    Returns (12, T) chroma matrix.
    """
    if not note_events:
        return np.zeros((12, 1))

    max_time = max(onset + dur for onset, dur, _ in note_events)
    n_frames = int(np.ceil(max_time * sr / hop_length)) + 1
    chroma = np.zeros((12, n_frames))

    for onset, dur, pitch in note_events:
        pc = pitch % 12
        start_frame = int(onset * sr / hop_length)
        end_frame = int((onset + dur) * sr / hop_length)
        end_frame = min(end_frame, n_frames)
        chroma[pc, start_frame:end_frame] += 1.0

    # Normalise columns
    col_sums = chroma.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    chroma /= col_sums

    return chroma


def chroma_dtw_distance(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """DTW distance between two chroma sequences using cosine distance.

    Args:
        chroma_a: (12, T1) chroma features
        chroma_b: (12, T2) chroma features

    Returns:
        Normalised DTW distance (lower = more similar).
    """
    T1 = chroma_a.shape[1] if chroma_a.ndim > 1 else 1
    T2 = chroma_b.shape[1] if chroma_b.ndim > 1 else 1

    if chroma_a.ndim == 1:
        chroma_a = chroma_a.reshape(12, 1)
    if chroma_b.ndim == 1:
        chroma_b = chroma_b.reshape(12, 1)

    # Cost matrix: cosine distance between each pair of frames
    # Normalise columns
    norm_a = np.linalg.norm(chroma_a, axis=0, keepdims=True)
    norm_b = np.linalg.norm(chroma_b, axis=0, keepdims=True)
    norm_a[norm_a == 0] = 1.0
    norm_b[norm_b == 0] = 1.0
    ca = chroma_a / norm_a
    cb = chroma_b / norm_b

    # DTW with restricted band (Sakoe-Chiba)
    band = max(T1, T2) // 4 + 1
    INF = 1e18
    D = np.full((T1 + 1, T2 + 1), INF)
    D[0, 0] = 0.0

    for i in range(1, T1 + 1):
        j_lo = max(1, i - band)
        j_hi = min(T2, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = 1.0 - float(np.dot(ca[:, i-1], cb[:, j-1]))
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    path_len = T1 + T2  # approximate
    return float(D[T1, T2]) / path_len if path_len > 0 else 0.0


# --- Cross-modal similarity ---

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def text_to_audio_retrieval(query: str,
                            candidate_tags: list[dict[str, float]]) -> RetrievalResult:
    """Retrieve audio tracks from a text query using tag-space similarity.

    candidate_tags: list of tag-weight dicts per candidate track.
    """
    tq = extract_tags_from_text(query)
    tag_order = sorted(MUSICAL_TAG_LEXICON.keys())
    q_vec = text_to_tag_vector(tq, tag_order)

    scores = np.zeros(len(candidate_tags))
    for i, tags in enumerate(candidate_tags):
        c_vec = np.zeros(len(tag_order))
        for j, tag in enumerate(tag_order):
            c_vec[j] = tags.get(tag, 0.0)
        scores[i] = cosine_similarity(q_vec, c_vec)

    ranked = np.argsort(-scores)
    top = float(scores[ranked[0]]) if len(ranked) > 0 else 0.0
    return RetrievalResult(ModalityType.TEXT, ModalityType.AUDIO,
                           scores, ranked, top)


def score_to_audio_retrieval(score_chroma: np.ndarray,
                             candidate_chromas: list[np.ndarray]) -> RetrievalResult:
    """Retrieve audio by matching a musical score's chroma to audio chromas via DTW."""
    scores = np.zeros(len(candidate_chromas))
    for i, cc in enumerate(candidate_chromas):
        dist = chroma_dtw_distance(score_chroma, cc)
        scores[i] = 1.0 / (1.0 + dist)  # convert distance to similarity

    ranked = np.argsort(-scores)
    top = float(scores[ranked[0]]) if len(ranked) > 0 else 0.0
    return RetrievalResult(ModalityType.SCORE, ModalityType.AUDIO,
                           scores, ranked, top)


def reciprocal_rank(ranked_indices: np.ndarray, relevant: set[int]) -> float:
    """Mean Reciprocal Rank for a ranked result list."""
    for rank, idx in enumerate(ranked_indices, 1):
        if int(idx) in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_indices: np.ndarray, relevant: set[int]) -> float:
    """Average Precision for a ranked result list."""
    hits = 0
    ap_sum = 0.0
    for rank, idx in enumerate(ranked_indices, 1):
        if int(idx) in relevant:
            hits += 1
            ap_sum += hits / rank
    return ap_sum / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked_indices: np.ndarray, relevance: dict[int, float],
              k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at position k."""
    dcg = 0.0
    for i, idx in enumerate(ranked_indices[:k]):
        rel = relevance.get(int(idx), 0.0)
        dcg += rel / math.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0
