"""Query expansion for music IR: semantic expansion, relevance feedback, query reformulation.

Expands user queries in music information retrieval systems using
pseudo-relevance feedback (Rocchio), semantic similarity, and
tag co-occurrence. Pure numpy — no external NLP libraries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExpandedQuery:
    original_terms: list[str]
    expanded_terms: list[str]
    weights: np.ndarray
    method: str


def rocchio_expand(query_vec: np.ndarray,
                   relevant_vecs: np.ndarray,
                   non_relevant_vecs: np.ndarray | None = None,
                   alpha: float = 1.0,
                   beta: float = 0.75,
                   gamma: float = 0.15) -> np.ndarray:
    expanded = alpha * query_vec
    if len(relevant_vecs) > 0:
        expanded += beta * np.mean(relevant_vecs, axis=0)
    if non_relevant_vecs is not None and len(non_relevant_vecs) > 0:
        expanded -= gamma * np.mean(non_relevant_vecs, axis=0)
    expanded = np.maximum(expanded, 0.0)
    norm = np.linalg.norm(expanded)
    if norm > 1e-8:
        expanded /= norm
    return expanded


def tag_cooccurrence_matrix(tag_lists: list[list[str]],
                             vocab: list[str]) -> np.ndarray:
    n = len(vocab)
    tag_to_idx = {t: i for i, t in enumerate(vocab)}
    cooc = np.zeros((n, n))
    for tags in tag_lists:
        indices = [tag_to_idx[t] for t in tags if t in tag_to_idx]
        for i in indices:
            for j in indices:
                if i != j:
                    cooc[i, j] += 1
    row_sums = cooc.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cooc / row_sums


def expand_by_cooccurrence(query_tags: list[str],
                            cooc_matrix: np.ndarray,
                            vocab: list[str],
                            top_k: int = 5,
                            threshold: float = 0.1) -> list[tuple[str, float]]:
    tag_to_idx = {t: i for i, t in enumerate(vocab)}
    scores = np.zeros(len(vocab))
    for tag in query_tags:
        if tag in tag_to_idx:
            scores += cooc_matrix[tag_to_idx[tag]]
    for tag in query_tags:
        if tag in tag_to_idx:
            scores[tag_to_idx[tag]] = 0.0
    ranked = np.argsort(scores)[::-1]
    result = []
    for idx in ranked[:top_k]:
        if scores[idx] >= threshold:
            result.append((vocab[idx], float(scores[idx])))
    return result


@dataclass
class BM25Index:
    docs: list[list[str]] = field(default_factory=list)
    vocab: dict[str, int] = field(default_factory=dict)
    df: np.ndarray = field(default=None)
    doc_lens: np.ndarray = field(default=None)
    avg_dl: float = 0.0
    k1: float = 1.5
    b: float = 0.75

    def build(self, documents: list[list[str]]) -> None:
        self.docs = documents
        vocab_set: set[str] = set()
        for doc in documents:
            vocab_set.update(doc)
        self.vocab = {t: i for i, t in enumerate(sorted(vocab_set))}
        n_terms = len(self.vocab)
        self.df = np.zeros(n_terms)
        self.doc_lens = np.array([len(d) for d in documents], dtype=float)
        self.avg_dl = float(np.mean(self.doc_lens)) if len(self.doc_lens) > 0 else 1.0
        for doc in documents:
            seen = set()
            for term in doc:
                if term in self.vocab and term not in seen:
                    self.df[self.vocab[term]] += 1
                    seen.add(term)

    def score(self, query: list[str], doc_idx: int) -> float:
        if doc_idx >= len(self.docs):
            return 0.0
        doc = self.docs[doc_idx]
        dl = self.doc_lens[doc_idx]
        n_docs = len(self.docs)
        s = 0.0
        tf_counts: dict[str, int] = {}
        for t in doc:
            tf_counts[t] = tf_counts.get(t, 0) + 1
        for term in query:
            if term not in self.vocab:
                continue
            ti = self.vocab[term]
            df_t = self.df[ti]
            if df_t == 0:
                continue
            idf = np.log((n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)
            tf = tf_counts.get(term, 0)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
            s += idf * tf_norm
        return float(s)

    def search(self, query: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(ranked_lists: list[list[int]],
                            k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rlist in ranked_lists:
        for rank, doc_id in enumerate(rlist, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return result
