"""BM25 and advanced retrieval: term weighting, Rocchio feedback, diversity reranking, evaluation."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class QueryTerm:
    term: str
    weight: float = 1.0
    is_expanded: bool = False


@dataclass
class RetrievalDocument:
    doc_id: str
    terms: Dict[str, int] = field(default_factory=dict)
    score: float = 0.0
    length: int = 0

    def term_freq(self, term: str) -> int:
        return self.terms.get(term, 0)


@dataclass
class RetrievalConfig:
    k1: float = 1.2
    b: float = 0.75
    top_k: int = 10
    expansion_terms: int = 5
    alpha: float = 1.0
    beta: float = 0.75
    gamma: float = 0.15


def tf(count: int, length: int) -> float:
    if length == 0:
        return 0.0
    return count / length


def idf(doc_freq: int, n_docs: int) -> float:
    if n_docs == 0:
        return 0.0
    return np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)


def tf_idf(count: int, length: int, doc_freq: int, n_docs: int) -> float:
    return tf(count, length) * idf(doc_freq, n_docs)


def bm25_term_score(term_freq: int, doc_length: int,
                    avg_dl: float, doc_freq: int, n_docs: int,
                    k1: float = 1.2, b: float = 0.75) -> float:
    idf_val = idf(doc_freq, n_docs)
    tf_norm = (term_freq * (k1 + 1)) / (
        term_freq + k1 * (1 - b + b * doc_length / max(avg_dl, 1e-9))
    )
    return idf_val * tf_norm


def bm25_score(query_terms: List[str], doc: RetrievalDocument,
               avg_dl: float, doc_freqs: Dict[str, int],
               n_docs: int, config: Optional[RetrievalConfig] = None) -> float:
    cfg = config or RetrievalConfig()
    total = 0.0
    for term in query_terms:
        tf_val = doc.term_freq(term)
        df = doc_freqs.get(term, 0)
        total += bm25_term_score(tf_val, doc.length, avg_dl, df, n_docs,
                                 cfg.k1, cfg.b)
    return total


def rocchio_expansion(query_vec: np.ndarray,
                      relevant_vecs: List[np.ndarray],
                      non_relevant_vecs: Optional[List[np.ndarray]] = None,
                      alpha: float = 1.0, beta: float = 0.75,
                      gamma: float = 0.15) -> np.ndarray:
    result = alpha * query_vec
    if relevant_vecs:
        rel_centroid = np.mean(relevant_vecs, axis=0)
        result += beta * rel_centroid
    if non_relevant_vecs:
        nrel_centroid = np.mean(non_relevant_vecs, axis=0)
        result -= gamma * nrel_centroid
    return np.maximum(result, 0.0)


def pseudo_relevance_expansion(query_vec: np.ndarray,
                               top_doc_vecs: List[np.ndarray],
                               n_expand: int = 5,
                               beta: float = 0.5) -> np.ndarray:
    if not top_doc_vecs:
        return query_vec.copy()
    centroid = np.mean(top_doc_vecs[:n_expand], axis=0)
    return query_vec + beta * centroid


def term_proximity(positions_a: List[int],
                   positions_b: List[int]) -> float:
    if not positions_a or not positions_b:
        return float('inf')
    min_dist = float('inf')
    for pa in positions_a:
        for pb in positions_b:
            d = abs(pa - pb)
            if d < min_dist:
                min_dist = d
    return float(min_dist)


def term_cooccurrence_score(doc_terms: Dict[str, int],
                            query_terms: List[str]) -> float:
    present = sum(1 for t in query_terms if t in doc_terms)
    if len(query_terms) == 0:
        return 0.0
    return present / len(query_terms)


def mutual_info_expansion(term_doc_matrix: np.ndarray,
                          query_indices: List[int],
                          n_expand: int = 5) -> List[int]:
    n_terms, n_docs = term_doc_matrix.shape
    if n_docs == 0:
        return []
    query_mask = np.zeros(n_docs, dtype=bool)
    for qi in query_indices:
        if qi < n_terms:
            query_mask |= term_doc_matrix[qi] > 0
    scores = []
    for t in range(n_terms):
        if t in query_indices:
            continue
        t_mask = term_doc_matrix[t] > 0
        a = np.sum(query_mask & t_mask)
        b = np.sum(~query_mask & t_mask)
        c = np.sum(query_mask & ~t_mask)
        d = np.sum(~query_mask & ~t_mask)
        n = a + b + c + d
        if n == 0 or a == 0:
            scores.append((t, 0.0))
            continue
        mi = (a / n) * np.log2((a * n) / max((a + b) * (a + c), 1e-12) + 1e-12)
        scores.append((t, mi))
    scores.sort(key=lambda x: -x[1])
    return [s[0] for s in scores[:n_expand]]


def diversity_rerank(docs: List[RetrievalDocument],
                     doc_vecs: List[np.ndarray],
                     lambda_: float = 0.5,
                     k: int = 10) -> List[RetrievalDocument]:
    if len(docs) <= 1:
        return docs[:k]
    selected = [0]
    remaining = list(range(1, len(docs)))
    while len(selected) < k and remaining:
        best_idx = -1
        best_score = -float('inf')
        for r in remaining:
            relevance = docs[r].score
            max_sim = 0.0
            for s in selected:
                sim = float(np.dot(doc_vecs[r], doc_vecs[s]) / (
                    np.linalg.norm(doc_vecs[r]) * np.linalg.norm(doc_vecs[s]) + 1e-12
                ))
                max_sim = max(max_sim, sim)
            mmr = lambda_ * relevance - (1 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = r
        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)
    return [docs[i] for i in selected]


def retrieval_average_precision(relevant: List[bool]) -> float:
    if not any(relevant):
        return 0.0
    precisions = []
    hits = 0
    for i, rel in enumerate(relevant):
        if rel:
            hits += 1
            precisions.append(hits / (i + 1))
    return float(np.mean(precisions))


def dcg(relevances: List[float], k: Optional[int] = None) -> float:
    if k is not None:
        relevances = relevances[:k]
    total = 0.0
    for i, rel in enumerate(relevances):
        total += rel / np.log2(i + 2)
    return total


def ndcg(relevances: List[float], k: Optional[int] = None) -> float:
    actual = dcg(relevances, k)
    ideal = dcg(sorted(relevances, reverse=True), k)
    if ideal < 1e-12:
        return 0.0
    return actual / ideal
