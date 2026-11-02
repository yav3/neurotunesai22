"""Scaling optimizations: O(n³) → O(n·k²) triplet pruning, MIS, batch construction.

Pure numpy/scipy. No torch/tensorflow/jax.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# 1. KNN graph + triplet pruning
# ---------------------------------------------------------------------------

def build_knn_graph(
    embeddings: np.ndarray, k: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Build k-nearest-neighbor graph via KDTree.

    Args:
        embeddings: (n, d) array.
        k: neighbors per node.

    Returns:
        (distances, indices) each (n, k). indices[i] are the k nearest
        neighbors of embeddings[i] (excluding self).
    """
    n = embeddings.shape[0]
    k_query = min(k + 1, n)
    tree = KDTree(embeddings)
    dists, idxs = tree.query(embeddings, k=k_query)
    # first column is self (distance 0); drop it
    if k_query > 1:
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]
    return dists, idxs


def prune_triplets(
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    embeddings: np.ndarray,
    k: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only triplets whose positive AND negative are in anchor's k-neighborhood.

    Reduces O(n³) candidate set to O(n·k²).

    Returns:
        (pruned_anchors, pruned_positives, pruned_negatives) index arrays.
    """
    _, knn_idx = build_knn_graph(embeddings, k)
    n = embeddings.shape[0]
    neighbor_set = [set(knn_idx[i]) for i in range(n)]

    mask = np.array([
        (int(positives[i]) in neighbor_set[int(anchors[i])]) and
        (int(negatives[i]) in neighbor_set[int(anchors[i])])
        for i in range(len(anchors))
    ], dtype=bool)

    return anchors[mask], positives[mask], negatives[mask]


def efficient_triplet_loss(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 32,
    margin: float = 0.2,
) -> float:
    """Triplet loss using only k-neighborhood triplets.  O(n·k²)."""
    miner = TripletMiner(k=k, margin=margin, strategy="semi_hard")
    a_idx, p_idx, n_idx = miner.mine(embeddings, labels)
    if len(a_idx) == 0:
        return 0.0
    d_ap = np.linalg.norm(embeddings[a_idx] - embeddings[p_idx], axis=1)
    d_an = np.linalg.norm(embeddings[a_idx] - embeddings[n_idx], axis=1)
    losses = np.maximum(d_ap - d_an + margin, 0.0)
    return float(np.mean(losses))


@dataclass
class TripletMiner:
    """Mine informative triplets from k-neighborhood."""

    k: int = 32
    margin: float = 0.2
    strategy: str = "semi_hard"  # "semi_hard" | "hardest" | "all"

    def mine(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.strategy == "semi_hard":
            return semi_hard_triplets(embeddings, labels, self.k, self.margin)
        elif self.strategy == "hardest":
            return hardest_negative_mining(embeddings, labels, self.k)
        else:
            return self._all_knn_triplets(embeddings, labels)

    def _all_knn_triplets(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, knn_idx = build_knn_graph(embeddings, self.k)
        a_list, p_list, n_list = [], [], []
        for i in range(len(embeddings)):
            neighbors = knn_idx[i]
            pos_mask = labels[neighbors] == labels[i]
            neg_mask = labels[neighbors] != labels[i]
            pos_nbrs = neighbors[pos_mask]
            neg_nbrs = neighbors[neg_mask]
            for p in pos_nbrs:
                for n in neg_nbrs:
                    a_list.append(i)
                    p_list.append(p)
                    n_list.append(n)
        if not a_list:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
        return np.array(a_list), np.array(p_list), np.array(n_list)


# ---------------------------------------------------------------------------
# 2. Semi-hard / hardest mining
# ---------------------------------------------------------------------------

def semi_hard_triplets(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 32,
    margin: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semi-hard triplets: d(a,p) < d(a,n) < d(a,p) + margin, within k-neighborhood."""
    _, knn_idx = build_knn_graph(embeddings, k)
    a_list, p_list, n_list = [], [], []
    for i in range(len(embeddings)):
        neighbors = knn_idx[i]
        pos_mask = labels[neighbors] == labels[i]
        neg_mask = labels[neighbors] != labels[i]
        pos_nbrs = neighbors[pos_mask]
        neg_nbrs = neighbors[neg_mask]
        if len(pos_nbrs) == 0 or len(neg_nbrs) == 0:
            continue
        for p in pos_nbrs:
            d_ap = float(np.linalg.norm(embeddings[i] - embeddings[p]))
            for n in neg_nbrs:
                d_an = float(np.linalg.norm(embeddings[i] - embeddings[n]))
                if d_ap < d_an < d_ap + margin:
                    a_list.append(i)
                    p_list.append(p)
                    n_list.append(n)
    if not a_list:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    return np.array(a_list), np.array(p_list), np.array(n_list)


def hardest_negative_mining(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each (anchor, positive) pair, pick the closest negative in k-neighborhood."""
    _, knn_idx = build_knn_graph(embeddings, k)
    a_list, p_list, n_list = [], [], []
    for i in range(len(embeddings)):
        neighbors = knn_idx[i]
        pos_mask = labels[neighbors] == labels[i]
        neg_mask = labels[neighbors] != labels[i]
        pos_nbrs = neighbors[pos_mask]
        neg_nbrs = neighbors[neg_mask]
        if len(pos_nbrs) == 0 or len(neg_nbrs) == 0:
            continue
        neg_dists = np.linalg.norm(embeddings[neg_nbrs] - embeddings[i], axis=1)
        hardest_neg = neg_nbrs[np.argmin(neg_dists)]
        for p in pos_nbrs:
            a_list.append(i)
            p_list.append(p)
            n_list.append(hardest_neg)
    if not a_list:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    return np.array(a_list), np.array(p_list), np.array(n_list)


# ---------------------------------------------------------------------------
# 3. MIS (Maximum Independent Set)
# ---------------------------------------------------------------------------

def greedy_mis(adjacency: np.ndarray) -> np.ndarray:
    """Greedy maximum independent set on an adjacency matrix.

    Iteratively picks the lowest-degree unvisited node and removes its neighbors.

    Args:
        adjacency: (n, n) symmetric binary adjacency matrix.

    Returns:
        1-d boolean mask of selected nodes.
    """
    n = adjacency.shape[0]
    adj = adjacency.astype(bool).copy()
    np.fill_diagonal(adj, False)
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)

    while available.any():
        degrees = adj[available][:, available].sum(axis=1)
        available_idx = np.where(available)[0]
        pick = available_idx[np.argmin(degrees)]
        selected[pick] = True
        available[pick] = False
        neighbors = np.where(adj[pick])[0]
        available[neighbors] = False

    return selected


def benchmark_mis(
    n_nodes_list: Optional[List[int]] = None,
    density: float = 0.1,
    n_trials: int = 3,
    seed: int = 42,
) -> Dict[str, List]:
    """Benchmark greedy MIS across graph sizes.

    Returns dict with keys: n, avg_time, avg_mis_size, avg_density.
    """
    if n_nodes_list is None:
        n_nodes_list = [50, 100, 200, 500]
    rng = np.random.RandomState(seed)
    result: Dict[str, List] = {"n": [], "avg_time": [], "avg_mis_size": [], "avg_density": []}

    for n in n_nodes_list:
        times, sizes = [], []
        for _ in range(n_trials):
            adj = (rng.rand(n, n) < density).astype(int)
            adj = np.triu(adj, 1)
            adj = adj + adj.T
            t0 = time.perf_counter()
            mis_mask = greedy_mis(adj)
            times.append(time.perf_counter() - t0)
            sizes.append(int(mis_mask.sum()))
        result["n"].append(n)
        result["avg_time"].append(float(np.mean(times)))
        result["avg_mis_size"].append(float(np.mean(sizes)))
        result["avg_density"].append(density)

    return result


def graph_based_batch_selection(
    embeddings: np.ndarray, k: int = 32, batch_size: int = 64
) -> np.ndarray:
    """Select diverse batch via MIS on k-NN similarity graph.

    Nodes connected if they are mutual k-nearest neighbors (too similar).
    MIS selects maximally diverse subset. If MIS < batch_size, pad with
    random remaining points.

    Returns:
        (batch_size,) index array.
    """
    n = embeddings.shape[0]
    batch_size = min(batch_size, n)
    _, knn_idx = build_knn_graph(embeddings, k)

    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in knn_idx[i]:
            if i in knn_idx[j]:
                adj[i, j] = True
                adj[j, i] = True

    mis_mask = greedy_mis(adj.astype(int))
    selected = np.where(mis_mask)[0]

    if len(selected) >= batch_size:
        return selected[:batch_size]

    remaining = np.setdiff1d(np.arange(n), selected)
    np.random.shuffle(remaining)
    extra = remaining[: batch_size - len(selected)]
    return np.concatenate([selected, extra])


# ---------------------------------------------------------------------------
# 4. Batch construction
# ---------------------------------------------------------------------------

class StratifiedBatchSampler:
    """Ensure each batch has balanced class representation."""

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int = 64,
        n_classes_per_batch: int = 8,
        seed: int = 42,
    ):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.n_classes_per_batch = n_classes_per_batch
        self.rng = np.random.RandomState(seed)

        self.unique_labels = np.unique(self.labels)
        self.label_to_idx: Dict[int, np.ndarray] = {}
        for lbl in self.unique_labels:
            self.label_to_idx[int(lbl)] = np.where(self.labels == lbl)[0]

    def sample(self) -> np.ndarray:
        n_classes = min(self.n_classes_per_batch, len(self.unique_labels))
        chosen_labels = self.rng.choice(self.unique_labels, size=n_classes, replace=False)
        per_class = max(1, self.batch_size // n_classes)
        indices = []
        for lbl in chosen_labels:
            pool = self.label_to_idx[int(lbl)]
            n_pick = min(per_class, len(pool))
            picked = self.rng.choice(pool, size=n_pick, replace=False)
            indices.extend(picked.tolist())
        indices = np.array(indices[: self.batch_size])
        self.rng.shuffle(indices)
        return indices


class DiversityBatchSampler:
    """Farthest-point sampling for maximum diversity batches."""

    def __init__(self, embeddings: np.ndarray, batch_size: int = 64, seed: int = 42):
        self.embeddings = embeddings
        self.batch_size = min(batch_size, len(embeddings))
        self.rng = np.random.RandomState(seed)

    def sample(self) -> np.ndarray:
        n = len(self.embeddings)
        bs = self.batch_size
        selected = [self.rng.randint(n)]
        min_dists = np.full(n, np.inf)

        for _ in range(bs - 1):
            last = self.embeddings[selected[-1]]
            dists = np.linalg.norm(self.embeddings - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected] = -1.0
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)

        return np.array(selected)


class CurriculumSampler:
    """Easy-to-hard curriculum: at low epochs, sample easy (low-loss) examples."""

    def __init__(
        self,
        losses_per_sample: np.ndarray,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.losses = np.asarray(losses_per_sample, dtype=np.float64)
        self.batch_size = min(batch_size, len(self.losses))
        self.rng = np.random.RandomState(seed)

    def sample(self, epoch: int, total_epochs: int = 100) -> np.ndarray:
        """Sample with curriculum weighting.

        At epoch 0: strongly prefer easy samples.
        At epoch total_epochs: uniform.
        """
        progress = min(epoch / max(total_epochs, 1), 1.0)
        # temperature: low early (sharp toward easy), high late (uniform)
        temperature = 0.1 + 0.9 * progress
        # invert losses so low loss → high weight
        max_loss = self.losses.max() + 1e-8
        easiness = max_loss - self.losses
        logits = easiness / (temperature * max_loss + 1e-8)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return self.rng.choice(len(self.losses), size=self.batch_size, replace=False, p=probs)


# ---------------------------------------------------------------------------
# 5. Scaling benchmarks
# ---------------------------------------------------------------------------

def _naive_triplet_loss(embeddings: np.ndarray, labels: np.ndarray, margin: float = 0.2) -> float:
    """O(n³) brute-force triplet loss for benchmarking."""
    n = len(embeddings)
    total_loss = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if labels[j] != labels[i]:
                continue
            if j == i:
                continue
            for m in range(n):
                if labels[m] == labels[i]:
                    continue
                d_ap = float(np.linalg.norm(embeddings[i] - embeddings[j]))
                d_an = float(np.linalg.norm(embeddings[i] - embeddings[m]))
                total_loss += max(d_ap - d_an + margin, 0.0)
                count += 1
    return total_loss / max(count, 1)


def benchmark_scaling(
    n_list: Optional[List[int]] = None,
    d: int = 64,
    k: int = 32,
    margin: float = 0.2,
    seed: int = 42,
) -> Dict[str, List]:
    """Time naive O(n³) vs pruned O(n·k²) triplet computation.

    Returns dict with keys: n, naive_time, pruned_time, speedup.
    """
    if n_list is None:
        n_list = [100, 500, 1000, 5000]
    rng = np.random.RandomState(seed)
    result: Dict[str, List] = {"n": [], "naive_time": [], "pruned_time": [], "speedup": []}

    for n in n_list:
        emb = rng.randn(n, d).astype(np.float64)
        labels = rng.randint(0, max(2, n // 10), size=n)
        k_eff = min(k, n - 1)

        # naive (skip for large n to avoid timeout)
        if n <= 500:
            t0 = time.perf_counter()
            _naive_triplet_loss(emb, labels, margin)
            naive_t = time.perf_counter() - t0
        else:
            # extrapolate from O(n³) scaling
            naive_t = float("inf")

        # pruned
        t0 = time.perf_counter()
        efficient_triplet_loss(emb, labels, k=k_eff, margin=margin)
        pruned_t = time.perf_counter() - t0

        speedup = naive_t / pruned_t if pruned_t > 0 and np.isfinite(naive_t) else float("inf")

        result["n"].append(n)
        result["naive_time"].append(round(naive_t, 4))
        result["pruned_time"].append(round(pruned_t, 4))
        result["speedup"].append(round(speedup, 1))

    return result
