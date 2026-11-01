"""Topic modeling for music corpora.

Latent Dirichlet Allocation (collapsed Gibbs sampling), non-negative matrix
factorization, and corpus-level statistics — all pure numpy, no gensim/sklearn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


MUSIC_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "not", "no", "so", "if", "than", "too",
    "very", "just", "that", "this", "it", "its", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them", "their", "which",
    "what", "who", "whom", "each", "all", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "also", "how",
}


@dataclass
class Vocabulary:
    word2id: Dict[str, int] = field(default_factory=dict)
    id2word: List[str] = field(default_factory=list)

    def add(self, word: str) -> int:
        if word not in self.word2id:
            idx = len(self.id2word)
            self.word2id[word] = idx
            self.id2word.append(word)
            return idx
        return self.word2id[word]

    def __len__(self) -> int:
        return len(self.id2word)


@dataclass
class TopicModelResult:
    topic_word: np.ndarray  # (K, V)
    doc_topic: np.ndarray   # (D, K)
    vocab: Vocabulary
    n_topics: int
    perplexity: float


def tokenize_music_text(
    text: str,
    stopwords: Optional[Set[str]] = None,
    min_length: int = 2,
) -> List[str]:
    """Tokenize and filter music text."""
    if stopwords is None:
        stopwords = MUSIC_STOPWORDS
    tokens = re.findall(r"[a-z][a-z'-]*[a-z]|[a-z]", text.lower())
    return [t for t in tokens if t not in stopwords and len(t) >= min_length]


def build_corpus(
    documents: List[str],
    min_df: int = 2,
    max_df_ratio: float = 0.8,
) -> Tuple[List[List[int]], Vocabulary]:
    """Build integer-encoded corpus with vocabulary filtering."""
    vocab = Vocabulary()
    raw_docs: List[List[str]] = []
    df: Dict[str, int] = {}
    for doc in documents:
        tokens = tokenize_music_text(doc)
        raw_docs.append(tokens)
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    n_docs = len(documents)
    max_df = int(max_df_ratio * n_docs)
    valid = {w for w, c in df.items() if min_df <= c <= max_df}

    corpus: List[List[int]] = []
    for tokens in raw_docs:
        doc_ids = [vocab.add(t) for t in tokens if t in valid]
        corpus.append(doc_ids)
    return corpus, vocab


def lda_gibbs(
    corpus: List[List[int]],
    n_vocab: int,
    n_topics: int = 10,
    alpha: float = 0.1,
    beta: float = 0.01,
    n_iterations: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapsed Gibbs sampling for LDA.

    Returns (topic_word: K×V, doc_topic: D×K) as normalized distributions.
    """
    rng = np.random.RandomState(seed)
    D = len(corpus)
    K = n_topics
    V = n_vocab

    # topic-word count, doc-topic count
    nkw = np.zeros((K, V), dtype=float)
    ndk = np.zeros((D, K), dtype=float)
    nk = np.zeros(K, dtype=float)

    # random init
    assignments: List[List[int]] = []
    for d, doc in enumerate(corpus):
        z_doc = []
        for w in doc:
            z = rng.randint(K)
            z_doc.append(z)
            nkw[z, w] += 1
            ndk[d, z] += 1
            nk[z] += 1
        assignments.append(z_doc)

    # Gibbs iterations
    for _ in range(n_iterations):
        for d, doc in enumerate(corpus):
            for i, w in enumerate(doc):
                z_old = assignments[d][i]
                nkw[z_old, w] -= 1
                ndk[d, z_old] -= 1
                nk[z_old] -= 1

                # conditional
                p = (nkw[:, w] + beta) / (nk + V * beta) * (ndk[d, :] + alpha)
                p /= p.sum()
                z_new = rng.choice(K, p=p)

                assignments[d][i] = z_new
                nkw[z_new, w] += 1
                ndk[d, z_new] += 1
                nk[z_new] += 1

    # normalize
    topic_word = (nkw + beta) / (nk[:, None] + V * beta)
    doc_topic = (ndk + alpha) / (ndk.sum(axis=1, keepdims=True) + K * alpha)
    return topic_word, doc_topic


def nmf_topics(
    tfidf: np.ndarray,
    n_topics: int = 10,
    n_iterations: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Non-negative matrix factorization via multiplicative update rules.

    Parameters
    ----------
    tfidf : (D, V) document-term matrix
    n_topics : number of topics K
    Returns (W: D×K basis, H: K×V topic-word).
    """
    rng = np.random.RandomState(seed)
    D, V = tfidf.shape
    K = n_topics
    eps = 1e-10

    W = rng.rand(D, K).astype(float) + eps
    H = rng.rand(K, V).astype(float) + eps

    for _ in range(n_iterations):
        # update H
        numerator_h = W.T @ tfidf
        denominator_h = W.T @ W @ H + eps
        H *= numerator_h / denominator_h
        # update W
        numerator_w = tfidf @ H.T
        denominator_w = W @ H @ H.T + eps
        W *= numerator_w / denominator_w

    # normalize rows of H
    row_sums = H.sum(axis=1, keepdims=True)
    H_norm = H / np.maximum(row_sums, eps)
    return W, H_norm


def tfidf_matrix(
    corpus: List[List[int]],
    n_vocab: int,
) -> np.ndarray:
    """Build TF-IDF matrix from integer-encoded corpus."""
    D = len(corpus)
    tf = np.zeros((D, n_vocab), dtype=float)
    for d, doc in enumerate(corpus):
        for w in doc:
            tf[d, w] += 1.0
    # normalize TF
    doc_lengths = tf.sum(axis=1, keepdims=True)
    tf = tf / np.maximum(doc_lengths, 1.0)
    # IDF
    df = (tf > 0).sum(axis=0).astype(float)
    idf = np.log((D + 1) / (df + 1)) + 1
    return tf * idf


def top_words_per_topic(
    topic_word: np.ndarray,
    vocab: Vocabulary,
    n_words: int = 10,
) -> List[List[Tuple[str, float]]]:
    """Extract top-n words for each topic."""
    result = []
    for k in range(topic_word.shape[0]):
        idxs = np.argsort(topic_word[k])[::-1][:n_words]
        words = [(vocab.id2word[i], float(topic_word[k, i])) for i in idxs]
        result.append(words)
    return result


def topic_coherence(
    topic_word: np.ndarray,
    corpus: List[List[int]],
    n_top: int = 10,
) -> List[float]:
    """UMass coherence: Σ log P(w_j, w_i) / P(w_i) for top words per topic."""
    D = len(corpus)
    # document frequency
    doc_sets = [set(doc) for doc in corpus]
    n_vocab = topic_word.shape[1]
    df = np.zeros(n_vocab)
    for ds in doc_sets:
        for w in ds:
            df[w] += 1

    coherences = []
    for k in range(topic_word.shape[0]):
        top_ids = np.argsort(topic_word[k])[::-1][:n_top]
        score = 0.0
        count = 0
        for i in range(1, len(top_ids)):
            for j in range(i):
                wi, wj = top_ids[i], top_ids[j]
                co = sum(1 for ds in doc_sets if wi in ds and wj in ds)
                score += np.log((co + 1) / max(df[wj], 1))
                count += 1
        coherences.append(score / max(count, 1))
    return coherences


def compute_perplexity(
    corpus: List[List[int]],
    topic_word: np.ndarray,
    doc_topic: np.ndarray,
) -> float:
    """Compute held-out perplexity."""
    total_ll = 0.0
    total_words = 0
    for d, doc in enumerate(corpus):
        for w in doc:
            pw = np.dot(doc_topic[d], topic_word[:, w])
            total_ll += np.log(max(pw, 1e-300))
            total_words += 1
    if total_words == 0:
        return float("inf")
    return float(np.exp(-total_ll / total_words))


def fit_lda(
    documents: List[str],
    n_topics: int = 10,
    n_iterations: int = 100,
    alpha: float = 0.1,
    beta: float = 0.01,
    seed: int = 42,
) -> TopicModelResult:
    """End-to-end LDA: tokenize → build corpus → Gibbs → result."""
    corpus, vocab = build_corpus(documents)
    topic_word, doc_topic = lda_gibbs(
        corpus, len(vocab), n_topics, alpha, beta, n_iterations, seed
    )
    perp = compute_perplexity(corpus, topic_word, doc_topic)
    return TopicModelResult(
        topic_word=topic_word,
        doc_topic=doc_topic,
        vocab=vocab,
        n_topics=n_topics,
        perplexity=perp,
    )


def fit_nmf(
    documents: List[str],
    n_topics: int = 10,
    n_iterations: int = 200,
    seed: int = 42,
) -> TopicModelResult:
    """End-to-end NMF: tokenize → TF-IDF → NMF → result."""
    corpus, vocab = build_corpus(documents)
    mat = tfidf_matrix(corpus, len(vocab))
    W, H = nmf_topics(mat, n_topics, n_iterations, seed)
    # normalize W to doc-topic distribution
    row_sums = W.sum(axis=1, keepdims=True)
    doc_topic = W / np.maximum(row_sums, 1e-10)
    perp = compute_perplexity(corpus, H, doc_topic)
    return TopicModelResult(
        topic_word=H,
        doc_topic=doc_topic,
        vocab=vocab,
        n_topics=n_topics,
        perplexity=perp,
    )
