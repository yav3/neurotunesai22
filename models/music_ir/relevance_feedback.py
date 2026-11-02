"""Relevance feedback for music IR: user-in-the-loop query refinement."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeedbackSession:
    """Tracks a user's relevance judgements across retrieval iterations."""
    query_vector: np.ndarray
    relevant_indices: list[int] = field(default_factory=list)
    non_relevant_indices: list[int] = field(default_factory=list)
    iterations: int = 0


@dataclass
class FeedbackResult:
    new_query: np.ndarray
    shift_magnitude: float  # how far the query moved
    relevant_centroid: np.ndarray | None
    non_relevant_centroid: np.ndarray | None


def rocchio_feedback(session: FeedbackSession,
                     doc_vectors: np.ndarray,
                     alpha: float = 1.0,
                     beta: float = 0.75,
                     gamma: float = 0.15) -> FeedbackResult:
    """Classic Rocchio relevance feedback.

    q' = α·q + β·mean(rel) − γ·mean(non_rel)

    Args:
        session: current feedback session with query and judgements
        doc_vectors: (N, d) matrix of document feature vectors
        alpha: weight for original query
        beta: weight for relevant document centroid
        gamma: weight for non-relevant document centroid
    """
    q = session.query_vector.copy()
    d = len(q)

    rel_centroid = None
    if session.relevant_indices:
        rel_vecs = doc_vectors[session.relevant_indices]
        rel_centroid = rel_vecs.mean(axis=0)
    else:
        rel_centroid = np.zeros(d)

    nrel_centroid = None
    if session.non_relevant_indices:
        nrel_vecs = doc_vectors[session.non_relevant_indices]
        nrel_centroid = nrel_vecs.mean(axis=0)
    else:
        nrel_centroid = np.zeros(d)

    new_q = alpha * q + beta * rel_centroid - gamma * nrel_centroid

    # Clip negative components (standard in Rocchio)
    new_q = np.maximum(new_q, 0.0)

    shift = float(np.linalg.norm(new_q - q))

    return FeedbackResult(
        new_q,
        shift,
        rel_centroid if session.relevant_indices else None,
        nrel_centroid if session.non_relevant_indices else None,
    )


def ide_feedback(session: FeedbackSession,
                 doc_vectors: np.ndarray) -> FeedbackResult:
    """Ide dec-hi variant: uses only the single most non-relevant document.

    q' = q + mean(rel) − max_non_rel

    More aggressive than Rocchio on the negative side.
    """
    q = session.query_vector.copy()
    d = len(q)

    if session.relevant_indices:
        rel_centroid = doc_vectors[session.relevant_indices].mean(axis=0)
    else:
        rel_centroid = np.zeros(d)

    if session.non_relevant_indices:
        # Pick the non-relevant doc closest to the query (hardest negative)
        nrel_vecs = doc_vectors[session.non_relevant_indices]
        sims = nrel_vecs @ q
        hardest = session.non_relevant_indices[int(np.argmax(sims))]
        nrel_vec = doc_vectors[hardest]
    else:
        nrel_vec = np.zeros(d)

    new_q = q + rel_centroid - nrel_vec
    new_q = np.maximum(new_q, 0.0)
    shift = float(np.linalg.norm(new_q - q))

    return FeedbackResult(new_q, shift, rel_centroid, nrel_vec)


def pseudo_relevance_feedback(query_vector: np.ndarray,
                              doc_vectors: np.ndarray,
                              scores: np.ndarray,
                              top_k: int = 5,
                              alpha: float = 1.0,
                              beta: float = 0.5) -> FeedbackResult:
    """Blind/pseudo relevance feedback: assume top-k results are relevant.

    No user judgement needed — uses initial ranking as pseudo-labels.
    """
    ranked = np.argsort(-scores)
    pseudo_rel = ranked[:top_k].tolist()

    session = FeedbackSession(
        query_vector=query_vector,
        relevant_indices=pseudo_rel,
    )
    return rocchio_feedback(session, doc_vectors, alpha=alpha, beta=beta, gamma=0.0)


def query_expansion_by_feedback(original_terms: dict[str, float],
                                relevant_doc_terms: list[dict[str, float]],
                                top_n: int = 10,
                                weight_decay: float = 0.5) -> dict[str, float]:
    """Expand query terms based on term distributions in relevant documents.

    Selects the top-N terms (by aggregated TF-IDF across relevant docs)
    that are not already in the query, and adds them with decayed weight.
    """
    # Aggregate term scores across relevant documents
    agg: dict[str, float] = {}
    for doc_terms in relevant_doc_terms:
        for term, score in doc_terms.items():
            agg[term] = agg.get(term, 0.0) + score

    # Remove terms already in query
    for t in original_terms:
        agg.pop(t, None)

    # Select top-N expansion terms
    expansion = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Merge with original
    result = dict(original_terms)
    max_orig = max(original_terms.values()) if original_terms else 1.0
    for term, score in expansion:
        result[term] = weight_decay * score / (max(agg.values()) or 1.0) * max_orig

    return result


# --- Implicit feedback from listening behaviour ---

@dataclass
class ListeningEvent:
    track_id: int
    listen_duration: float  # seconds
    total_duration: float   # track length in seconds
    skipped: bool
    repeated: bool
    saved: bool


def implicit_relevance_score(event: ListeningEvent) -> float:
    """Convert a listening event to an implicit relevance score (0-1).

    Combines listen fraction, skip penalty, repeat/save bonus.
    """
    if event.total_duration <= 0:
        return 0.0

    listen_frac = min(event.listen_duration / event.total_duration, 1.0)

    score = listen_frac * 0.6
    if event.skipped:
        score -= 0.2
    if event.repeated:
        score += 0.25
    if event.saved:
        score += 0.15

    return float(np.clip(score, 0.0, 1.0))


def implicit_feedback_session(events: list[ListeningEvent],
                              threshold: float = 0.5) -> FeedbackSession:
    """Build a feedback session from implicit listening behaviour.

    Events scoring above threshold are treated as relevant; below as non-relevant.
    """
    relevant = []
    non_relevant = []
    for i, evt in enumerate(events):
        if implicit_relevance_score(evt) >= threshold:
            relevant.append(evt.track_id)
        else:
            non_relevant.append(evt.track_id)

    return FeedbackSession(
        query_vector=np.zeros(0),  # to be filled by caller
        relevant_indices=relevant,
        non_relevant_indices=non_relevant,
    )


def diversity_reranking(scores: np.ndarray, doc_vectors: np.ndarray,
                        lambda_: float = 0.5, top_k: int = 10) -> np.ndarray:
    """MMR-style diversity reranking (Maximal Marginal Relevance).

    Balances relevance (original score) with novelty (dissimilarity to
    already-selected documents).

    Returns indices of top-k reranked documents.
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)

    # Normalise doc vectors
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = doc_vectors / norms

    selected: list[int] = []
    remaining = set(range(n))

    for _ in range(min(top_k, n)):
        best_idx = -1
        best_mmr = -np.inf

        for idx in remaining:
            rel = scores[idx]
            if selected:
                sims = normed[idx] @ normed[selected].T
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0

            mmr = lambda_ * rel - (1 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return np.array(selected)
