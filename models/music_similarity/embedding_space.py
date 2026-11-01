"""Music embedding space: dimensionality reduction, LSH, kNN, quality metrics.

Implements embedding-space construction and evaluation for music
similarity retrieval.  All methods are pure numpy / scipy with no
external ML frameworks.

Theoretical foundations:

* Johnson & Lindenstrauss (1984) — random projections preserve
  pairwise distances within (1 ± ε) with O(log n / ε²) dimensions.
* Indyk & Motwani (1998) — locality-sensitive hashing for approximate
  nearest-neighbour search in sub-linear time.
* Venna & Kaski (2006) — trustworthiness / continuity for evaluating
  dimensionality reduction quality.
* Radovanović, Nanopoulos & Ivanović (2010) — hubness phenomenon in
  high-dimensional spaces; k-occurrence distribution skew.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """Configuration for an embedding space."""
    dim: int
    metric: str = "cosine"
    normalize: bool = True


@dataclass
class MusicEmbedding:
    """A single track embedding with metadata."""
    vector: np.ndarray
    track_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityResult:
    """Result of a similarity search."""
    query_id: str
    neighbors: List[Tuple[str, float]]  # (id, distance)
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core distance / similarity
# ---------------------------------------------------------------------------

def es_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Returns 1.0 for identical directions, 0.0 for orthogonal, −1.0 for
    opposite.  Handles zero-norm vectors gracefully (returns 0.0).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance matrix.

    Parameters
    ----------
    embeddings : (n, d) array

    Returns
    -------
    (n, n) array with values in [0, 2].  ``D[i,j] = 1 - cos(e_i, e_j)``.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    E_norm = E / norms
    sim = E_norm @ E_norm.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def euclidean_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix via the expansion trick.

    Uses ||a - b||² = ||a||² + ||b||² − 2 a·b to avoid O(n²d) explicit
    subtraction.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    sq = np.sum(E ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * E @ E.T
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)


def es_mahalanobis_distance(
    a: np.ndarray, b: np.ndarray, cov_inv: np.ndarray
) -> float:
    """Mahalanobis distance between two vectors.

    ``d_M(a, b) = sqrt((a - b)^T Σ^{-1} (a - b))``
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    val = float(diff @ np.asarray(cov_inv, dtype=np.float64) @ diff)
    return float(np.sqrt(max(val, 0.0)))


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise each row to unit length."""
    E = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return E / norms


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def pca_reduce(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """PCA via truncated SVD (no sklearn).

    Centres the data, computes the top ``target_dim`` right singular
    vectors, and projects.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    mean = E.mean(axis=0)
    E_c = E - mean
    _, _, Vt = np.linalg.svd(E_c, full_matrices=False)
    return E_c @ Vt[:target_dim].T


def random_projection(
    embeddings: np.ndarray, target_dim: int, seed: int = 42
) -> np.ndarray:
    """Johnson-Lindenstrauss random projection.

    Projects from d to ``target_dim`` using a Gaussian random matrix
    scaled by ``1 / sqrt(target_dim)`` to approximately preserve
    pairwise distances (Johnson & Lindenstrauss 1984).
    """
    E = np.asarray(embeddings, dtype=np.float64)
    rng = np.random.RandomState(seed)
    R = rng.randn(E.shape[1], target_dim) / np.sqrt(target_dim)
    return E @ R


# ---------------------------------------------------------------------------
# Locality-sensitive hashing
# ---------------------------------------------------------------------------

def locality_sensitive_hash(
    embeddings: np.ndarray, n_planes: int, seed: int = 0
) -> List[int]:
    """Random-hyperplane LSH (Indyk & Motwani 1998).

    For each embedding, returns an integer hash code whose bits encode
    the sign of the dot product with ``n_planes`` random normal vectors.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    rng = np.random.RandomState(seed)
    planes = rng.randn(E.shape[1], n_planes)
    signs = (E @ planes > 0).astype(np.int64)
    # Encode each row as an integer
    powers = 1 << np.arange(n_planes, dtype=np.int64)
    codes = signs @ powers
    return codes.tolist()


# ---------------------------------------------------------------------------
# Nearest-neighbour search
# ---------------------------------------------------------------------------

def knn_search(
    query: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    metric: str = "cosine",
) -> List[Tuple[int, float]]:
    """Brute-force k-nearest-neighbour search.

    Parameters
    ----------
    metric : ``'cosine'`` or ``'euclidean'``

    Returns
    -------
    List of ``(index, distance)`` sorted by ascending distance.
    """
    q = np.asarray(query, dtype=np.float64).ravel()
    E = np.asarray(embeddings, dtype=np.float64)
    if metric == "cosine":
        nq = np.linalg.norm(q)
        ne = np.linalg.norm(E, axis=1)
        denom = np.maximum(nq * ne, 1e-15)
        sims = E @ q / denom
        dists = 1.0 - sims
    else:
        dists = np.linalg.norm(E - q[None, :], axis=1)
    k = min(k, len(dists))
    idx = np.argpartition(dists, k)[:k]
    idx = idx[np.argsort(dists[idx])]
    return [(int(i), float(dists[i])) for i in idx]


def approximate_nn(
    query: np.ndarray,
    embeddings: np.ndarray,
    lsh_codes: List[int],
    n_planes: int,
    k: int,
    seed: int = 0,
) -> List[Tuple[int, float]]:
    """LSH-accelerated approximate nearest-neighbour search.

    Computes the query's own hash code, then restricts brute-force
    search to embeddings whose Hamming distance in hash space is below
    ``n_planes // 2``.  Falls back to full search if the candidate set
    is smaller than *k*.
    """
    q = np.asarray(query, dtype=np.float64).ravel()
    E = np.asarray(embeddings, dtype=np.float64)
    rng = np.random.RandomState(seed)
    planes = rng.randn(E.shape[1], n_planes)
    q_signs = (q @ planes > 0).astype(np.int64)
    powers = 1 << np.arange(n_planes, dtype=np.int64)
    q_code = int(q_signs @ powers)

    codes_arr = np.asarray(lsh_codes, dtype=np.int64)
    # Hamming distance in hash space
    xor = np.bitwise_xor(codes_arr, q_code)
    hamming = np.zeros(len(codes_arr), dtype=int)
    tmp = xor.copy()
    while np.any(tmp > 0):
        hamming += (tmp & 1).astype(int)
        tmp >>= 1

    threshold = max(1, n_planes // 2)
    candidates = np.where(hamming <= threshold)[0]
    if len(candidates) < k:
        candidates = np.arange(len(E))

    # Cosine search over candidates
    nq = max(np.linalg.norm(q), 1e-15)
    ne = np.linalg.norm(E[candidates], axis=1)
    denom = np.maximum(nq * ne, 1e-15)
    sims = E[candidates] @ q / denom
    dists = 1.0 - sims

    kk = min(k, len(dists))
    local_idx = np.argpartition(dists, kk)[:kk]
    local_idx = local_idx[np.argsort(dists[local_idx])]
    return [(int(candidates[i]), float(dists[i])) for i in local_idx]


# ---------------------------------------------------------------------------
# Loss functions for embedding training
# ---------------------------------------------------------------------------

def es_triplet_loss(
    anchor: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    margin: float = 0.2,
) -> float:
    """Triplet margin loss: max(0, d(a,p) − d(a,n) + margin).

    Uses Euclidean distance.
    """
    a = np.asarray(anchor, dtype=np.float64).ravel()
    p = np.asarray(positive, dtype=np.float64).ravel()
    n = np.asarray(negative, dtype=np.float64).ravel()
    dp = float(np.linalg.norm(a - p))
    dn = float(np.linalg.norm(a - n))
    return max(0.0, dp - dn + margin)


def es_contrastive_loss(
    embed_a: np.ndarray,
    embed_b: np.ndarray,
    label: int,
    margin: float = 1.0,
) -> float:
    """Contrastive loss (Hadsell, Chopra & LeCun 2006).

    ``label = 1`` for similar pairs, ``label = 0`` for dissimilar.
    """
    a = np.asarray(embed_a, dtype=np.float64).ravel()
    b = np.asarray(embed_b, dtype=np.float64).ravel()
    d = float(np.linalg.norm(a - b))
    if label == 1:
        return 0.5 * d ** 2
    else:
        return 0.5 * max(0.0, margin - d) ** 2


# ---------------------------------------------------------------------------
# Embedding quality metrics
# ---------------------------------------------------------------------------

def embedding_quality_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Evaluate embedding quality: silhouette, cluster purity, ratio.

    Parameters
    ----------
    embeddings : (n, d)
    labels : (n,) integer cluster / class labels

    Returns
    -------
    Dict with ``silhouette``, ``cluster_purity``, ``inter_intra_ratio``.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    L = np.asarray(labels).ravel()
    n = len(E)
    unique_labels = np.unique(L)
    if len(unique_labels) < 2 or n < 3:
        return {"silhouette": 0.0, "cluster_purity": 1.0, "inter_intra_ratio": 0.0}

    D = euclidean_distance_matrix(E)

    # Silhouette (simplified: mean over all points)
    sil_vals = np.zeros(n)
    for i in range(n):
        same = L == L[i]
        same[i] = False
        if same.sum() == 0:
            sil_vals[i] = 0.0
            continue
        a_i = D[i, same].mean()
        b_i = np.inf
        for lab in unique_labels:
            if lab == L[i]:
                continue
            mask = L == lab
            if mask.sum() > 0:
                b_i = min(b_i, D[i, mask].mean())
        sil_vals[i] = (b_i - a_i) / max(max(a_i, b_i), 1e-15)
    silhouette = float(sil_vals.mean())

    # Cluster purity: 1-NN classification accuracy
    correct = 0
    for i in range(n):
        dists_i = D[i].copy()
        dists_i[i] = np.inf
        nn = np.argmin(dists_i)
        if L[nn] == L[i]:
            correct += 1
    purity = correct / n

    # Inter/intra cluster distance ratio
    intra = 0.0
    intra_count = 0
    inter = 0.0
    inter_count = 0
    for lab in unique_labels:
        mask = L == lab
        idx = np.where(mask)[0]
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                intra += D[idx[ii], idx[jj]]
                intra_count += 1
    for i_lab in range(len(unique_labels)):
        for j_lab in range(i_lab + 1, len(unique_labels)):
            mask_a = L == unique_labels[i_lab]
            mask_b = L == unique_labels[j_lab]
            sub = D[np.ix_(np.where(mask_a)[0], np.where(mask_b)[0])]
            inter += sub.sum()
            inter_count += sub.size
    avg_intra = intra / max(intra_count, 1)
    avg_inter = inter / max(inter_count, 1)
    ratio = avg_inter / max(avg_intra, 1e-15)

    return {
        "silhouette": silhouette,
        "cluster_purity": purity,
        "inter_intra_ratio": ratio,
    }


def neighborhood_overlap(
    emb_a: np.ndarray, emb_b: np.ndarray, k: int = 10
) -> float:
    """Fraction of shared k-nearest neighbours between two embeddings.

    Both embedding matrices must have the same number of rows (same
    points, different representations).  Returns a value in [0, 1].
    """
    A = np.asarray(emb_a, dtype=np.float64)
    B = np.asarray(emb_b, dtype=np.float64)
    n = len(A)
    k = min(k, n - 1)
    if k < 1:
        return 0.0

    Da = euclidean_distance_matrix(A)
    Db = euclidean_distance_matrix(B)

    overlap = 0.0
    for i in range(n):
        Da[i, i] = np.inf
        Db[i, i] = np.inf
        nn_a = set(np.argpartition(Da[i], k)[:k].tolist())
        nn_b = set(np.argpartition(Db[i], k)[:k].tolist())
        overlap += len(nn_a & nn_b) / k
    return overlap / n


def manifold_trustworthiness(
    high_dim: np.ndarray, low_dim: np.ndarray, k: int = 7
) -> float:
    """Trustworthiness metric for dimensionality reduction quality.

    Venna & Kaski (2006).  Measures whether the k-nearest neighbours in
    the low-dimensional embedding were also close in the original
    high-dimensional space.  Returns a value in [0, 1] where 1.0 is
    perfect.
    """
    X = np.asarray(high_dim, dtype=np.float64)
    Y = np.asarray(low_dim, dtype=np.float64)
    n = len(X)
    k = min(k, n - 2)
    if k < 1:
        return 1.0

    Dx = euclidean_distance_matrix(X)
    Dy = euclidean_distance_matrix(Y)

    # Rank matrices
    rank_x = np.zeros((n, n), dtype=int)
    for i in range(n):
        rank_x[i] = np.argsort(np.argsort(Dx[i]))

    penalty = 0.0
    for i in range(n):
        Dy[i, i] = np.inf
        nn_y = set(np.argpartition(Dy[i], k)[:k].tolist())
        Dx[i, i] = np.inf
        nn_x = set(np.argpartition(Dx[i], k)[:k].tolist())
        for j in nn_y:
            if j not in nn_x:
                penalty += rank_x[i, j] - k

    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        return 1.0
    return 1.0 - (2.0 / denom) * penalty


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_similarity_graph(
    embeddings: np.ndarray, k: int, threshold: float = 0.5
) -> Dict[int, List[Tuple[int, float]]]:
    """Build a k-NN similarity graph (adjacency list).

    An edge from *i* to *j* is included if *j* is among the *k*
    nearest neighbours of *i* **and** their cosine similarity exceeds
    ``threshold``.

    Returns
    -------
    Dict mapping node index → list of ``(neighbour_index, similarity)``.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    n = len(E)
    k = min(k, n - 1)
    D = cosine_distance_matrix(E)

    graph: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n):
        D[i, i] = np.inf
        idx = np.argpartition(D[i], k)[:k]
        neighbours = []
        for j in idx:
            sim = 1.0 - D[i, j]
            if sim >= threshold:
                neighbours.append((int(j), float(sim)))
        neighbours.sort(key=lambda x: -x[1])
        graph[i] = neighbours
    return graph


# ---------------------------------------------------------------------------
# Hubness analysis
# ---------------------------------------------------------------------------

def hub_score(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """k-occurrence (hubness) score for each point.

    A point's hub score is the number of times it appears in other
    points' k-nearest-neighbour lists.  High hub scores indicate
    points that are "universally close" — a known pathology in high-
    dimensional spaces (Radovanović et al. 2010).

    Returns
    -------
    (n,) integer array of k-occurrence counts.
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = len(D)
    k = min(k, n - 1)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        d_i = D[i].copy()
        d_i[i] = np.inf
        nn = np.argpartition(d_i, k)[:k]
        for j in nn:
            counts[j] += 1
    return counts


def hubness_skewness(hub_scores: np.ndarray) -> float:
    """Skewness of the hub-score distribution.

    Positive skewness indicates the presence of hubs — a small number
    of points that appear disproportionately often as nearest
    neighbours (Radovanović et al. 2010).
    """
    h = np.asarray(hub_scores, dtype=np.float64)
    mu = h.mean()
    std = h.std()
    if std < 1e-15:
        return 0.0
    return float(np.mean(((h - mu) / std) ** 3))


def anti_hub_fraction(hub_scores: np.ndarray) -> float:
    """Fraction of points that are never a nearest neighbour (anti-hubs)."""
    h = np.asarray(hub_scores, dtype=np.float64)
    return float(np.mean(h == 0))


# ---------------------------------------------------------------------------
# Embedding-space diagnostics
# ---------------------------------------------------------------------------

def intrinsic_dimensionality(
    embeddings: np.ndarray, method: str = "mle", k: int = 10
) -> float:
    """Estimate intrinsic dimensionality of the embedding manifold.

    Parameters
    ----------
    method : ``'mle'`` (Levina & Bickel 2005) or ``'pca'`` (fraction of
        variance explained above 95 %).
    """
    E = np.asarray(embeddings, dtype=np.float64)
    n = len(E)
    if method == "pca":
        cov = np.cov(E.T)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
        total = eigvals.sum()
        if total < 1e-15:
            return 1.0
        cumvar = np.cumsum(eigvals) / total
        return float(np.searchsorted(cumvar, 0.95) + 1)

    # MLE method (Levina & Bickel 2005)
    D = euclidean_distance_matrix(E)
    k = min(k, n - 1)
    estimates = []
    for i in range(n):
        d_i = np.sort(D[i])[1 : k + 1]  # skip self
        d_i = d_i[d_i > 1e-15]
        if len(d_i) < 2:
            continue
        r_k = d_i[-1]
        if r_k < 1e-15:
            continue
        log_ratios = np.log(r_k / d_i[:-1])
        est = (len(d_i) - 1) / log_ratios.sum()
        estimates.append(est)
    if not estimates:
        return 1.0
    return float(np.mean(estimates))


def isotropy_score(embeddings: np.ndarray) -> float:
    """Measure how uniformly the embeddings fill directional space.

    Returns a value in (0, 1].  1.0 = perfectly isotropic (all singular
    values equal); near 0 = degenerate / collapsed along few axes.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    E_c = E - E.mean(axis=0)
    s = np.linalg.svd(E_c, compute_uv=False)
    s = s[s > 1e-15]
    if len(s) == 0:
        return 0.0
    # Ratio of minimum to maximum singular value
    return float(s.min() / s.max())


def alignment_uniformity(
    embeddings: np.ndarray, labels: np.ndarray, t: float = 2.0
) -> Tuple[float, float]:
    """Wang & Isola (2020) alignment and uniformity metrics.

    alignment = E[ ||f(x) - f(x+)||² ] over positive pairs
    uniformity = log E[ e^{-t ||f(x) - f(y)||²} ] over all pairs

    Lower is better for both.
    """
    E = normalize_embeddings(np.asarray(embeddings, dtype=np.float64))
    L = np.asarray(labels).ravel()
    n = len(E)

    # Alignment: mean squared distance between same-class pairs
    align_sum = 0.0
    align_count = 0
    for lab in np.unique(L):
        idx = np.where(L == lab)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                d2 = float(np.sum((E[idx[i]] - E[idx[j]]) ** 2))
                align_sum += d2
                align_count += 1
    alignment = align_sum / max(align_count, 1)

    # Uniformity: log-mean-exp of pairwise Gaussian kernel
    D2 = np.sum((E[:, None, :] - E[None, :, :]) ** 2, axis=2)
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    vals = np.exp(-t * D2[mask])
    uniformity = float(np.log(vals.mean() + 1e-15))

    return alignment, uniformity
