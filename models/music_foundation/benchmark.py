"""Evaluation and benchmarking harness for the frontier music foundation model.

Implements linear-probe classification, retrieval evaluation, clustering metrics,
representation analysis (CKA, intrinsic dimension, isotropy), and training
dynamics analysis. All from-scratch — no sklearn, no torch.

References:
  - Kornblith et al. (2019), "Similarity of Neural Network Representations Revisited"
  - Levina & Bickel (2004), "Maximum Likelihood Estimation of Intrinsic Dimension"
  - Mu et al. (2018), "All-but-the-Top: Simple and Effective Postprocessing for Word Representations"
  - Lloyd (1982), "Least squares quantization in PCM"
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BenchmarkConfig:
    tasks: List[str] = field(default_factory=lambda: [
        "probe", "retrieval", "clustering", "representation",
    ])
    n_trials: int = 3
    seed: int = 42
    batch_size: int = 32
    max_samples: int = 5000
    probe_epochs: int = 100
    probe_lr: float = 0.01
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    n_clusters: Optional[int] = None


# ---------- linear probe classifier ----------

class ProbeClassifier:
    """Logistic regression via gradient descent for evaluating frozen representations."""

    def __init__(self, n_features: int, n_classes: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        self.n_classes = n_classes

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-10)

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        oh = np.zeros((len(y), self.n_classes))
        oh[np.arange(len(y)), y.astype(int)] = 1.0
        return oh

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
    ) -> List[float]:
        """Train the linear probe. Returns per-epoch loss history."""
        n = len(X)
        y_oh = self._one_hot(y)
        losses = []
        rng = np.random.RandomState(42)

        for epoch in range(epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = X[idx]
                yb = y_oh[idx]
                logits = xb @ self.W + self.b
                probs = self._softmax(logits)
                loss = -np.mean(np.sum(yb * np.log(probs + 1e-10), axis=1))
                epoch_loss += loss
                n_batches += 1
                grad_logits = (probs - yb) / len(xb)
                grad_W = xb.T @ grad_logits + weight_decay * self.W
                grad_b = grad_logits.sum(axis=0)
                self.W -= lr * grad_W
                self.b -= lr * grad_b
            losses.append(epoch_loss / max(n_batches, 1))
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        return np.argmax(logits, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        return self._softmax(logits)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
        preds = self.predict(X)
        acc = float(np.mean(preds == y.astype(int)))
        classes = np.unique(y.astype(int))
        per_class = {}
        for c in classes:
            mask = y.astype(int) == c
            if mask.sum() > 0:
                per_class[int(c)] = float(np.mean(preds[mask] == c))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        class_map = {c: i for i, c in enumerate(classes)}
        for true, pred in zip(y.astype(int), preds):
            cm[class_map[true], class_map[pred]] += 1
        return {
            "accuracy": acc,
            "per_class_accuracy": per_class,
            "confusion_matrix": cm,
            "n_classes": len(classes),
        }


# ---------- retrieval benchmark ----------

class RetrievalBenchmark:
    """Evaluate representation quality via nearest-neighbor retrieval."""

    def __init__(self):
        self._embeddings: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def build_index(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._embeddings = embeddings / (norms + 1e-8)
        self._labels = labels

    def query(self, embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, similarities) of k nearest neighbors."""
        q = embedding / (np.linalg.norm(embedding) + 1e-8)
        sims = self._embeddings @ q
        top_k = np.argsort(sims)[::-1][:k]
        return top_k, sims[top_k]

    def precision_at_k(self, query_emb: np.ndarray, query_label: int, k: int) -> float:
        indices, _ = self.query(query_emb, k)
        relevant = (self._labels[indices] == query_label).astype(float)
        return float(relevant.sum() / k)

    def average_precision(self, query_emb: np.ndarray, query_label: int, k: int) -> float:
        indices, _ = self.query(query_emb, k)
        relevant = (self._labels[indices] == query_label).astype(float)
        if relevant.sum() == 0:
            return 0.0
        precisions = np.cumsum(relevant) / np.arange(1, k + 1)
        return float((precisions * relevant).sum() / relevant.sum())

    def evaluate_retrieval(
        self,
        query_embeddings: np.ndarray,
        query_labels: np.ndarray,
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        results = {}
        for k in k_values:
            p_at_k = [self.precision_at_k(query_embeddings[i], query_labels[i], k)
                       for i in range(len(query_embeddings))]
            results[f"precision@{k}"] = float(np.mean(p_at_k))
        aps = [self.average_precision(query_embeddings[i], query_labels[i], max(k_values))
               for i in range(len(query_embeddings))]
        results["mAP"] = float(np.mean(aps))
        ranks = []
        for i in range(len(query_embeddings)):
            indices, _ = self.query(query_embeddings[i], len(self._labels))
            match_positions = np.where(self._labels[indices] == query_labels[i])[0]
            if len(match_positions) > 0:
                ranks.append(1.0 / (match_positions[0] + 1))
            else:
                ranks.append(0.0)
        results["MRR"] = float(np.mean(ranks))
        return results


# ---------- clustering (k-means from scratch) ----------

def _kmeans(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm. Returns (labels, centroids)."""
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.choice(n, size=k, replace=False)
    centroids = X[indices].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = X[mask].mean(axis=0)

    return labels, centroids


def _entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -float(np.sum(probs * np.log(probs + 1e-10)))


def _mutual_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    n = len(labels_a)
    unique_a = np.unique(labels_a)
    unique_b = np.unique(labels_b)
    mi = 0.0
    for a in unique_a:
        for b in unique_b:
            n_ab = np.sum((labels_a == a) & (labels_b == b))
            if n_ab == 0:
                continue
            n_a = np.sum(labels_a == a)
            n_b = np.sum(labels_b == b)
            mi += (n_ab / n) * np.log(n * n_ab / (n_a * n_b) + 1e-10)
    return mi


def normalized_mutual_information(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    mi = _mutual_information(true_labels, pred_labels)
    h_true = _entropy(true_labels)
    h_pred = _entropy(pred_labels)
    denom = 0.5 * (h_true + h_pred)
    if denom < 1e-10:
        return 0.0
    return mi / denom


def adjusted_rand_index(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Adjusted Rand Index (Hubert & Arabie, 1985)."""
    n = len(true_labels)
    unique_t = np.unique(true_labels)
    unique_p = np.unique(pred_labels)
    contingency = np.zeros((len(unique_t), len(unique_p)))
    t_map = {v: i for i, v in enumerate(unique_t)}
    p_map = {v: i for i, v in enumerate(unique_p)}
    for i in range(n):
        contingency[t_map[true_labels[i]], p_map[pred_labels[i]]] += 1

    def comb2(x):
        return x * (x - 1) / 2

    sum_comb_nij = sum(comb2(contingency[i, j]) for i in range(len(unique_t)) for j in range(len(unique_p)))
    sum_comb_ai = sum(comb2(contingency[i].sum()) for i in range(len(unique_t)))
    sum_comb_bj = sum(comb2(contingency[:, j].sum()) for j in range(len(unique_p)))
    comb_n = comb2(n)

    expected = sum_comb_ai * sum_comb_bj / comb_n if comb_n > 0 else 0
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)
    denom = max_index - expected
    if abs(denom) < 1e-10:
        return 0.0
    return (sum_comb_nij - expected) / denom


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Average silhouette coefficient (Rousseeuw, 1987)."""
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    scores = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        diff_labels = unique_labels[unique_labels != labels[i]]
        a_i = np.mean(np.linalg.norm(X[same] - X[i], axis=1)) if same.sum() > 1 else 0
        b_i = np.inf
        for dl in diff_labels:
            mask = labels == dl
            d = np.mean(np.linalg.norm(X[mask] - X[i], axis=1))
            b_i = min(b_i, d)
        scores[i] = (b_i - a_i) / (max(a_i, b_i) + 1e-10)
    return float(np.mean(scores))


class ClusteringBenchmark:
    def evaluate(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> Dict[str, float]:
        k = n_clusters or len(np.unique(true_labels))
        pred_labels, centroids = _kmeans(embeddings, k)
        return {
            "NMI": normalized_mutual_information(true_labels, pred_labels),
            "ARI": adjusted_rand_index(true_labels, pred_labels),
            "silhouette": silhouette_score(embeddings, pred_labels),
            "n_clusters": k,
        }


# ---------- representation analysis ----------

def centered_kernel_alignment(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA with linear kernel (Kornblith et al., 2019)."""
    def _hsic(K, L, n):
        H = np.eye(n) - np.ones((n, n)) / n
        return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)

    n = X.shape[0]
    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = _hsic(K, L, n)
    hsic_kk = _hsic(K, K, n)
    hsic_ll = _hsic(L, L, n)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return float(hsic_kl / denom)


def intrinsic_dimensionality_mle(embeddings: np.ndarray, k: int = 5) -> float:
    """MLE estimator of intrinsic dimension (Levina & Bickel, 2004)."""
    n, d = embeddings.shape
    estimates = []
    for i in range(n):
        dists = np.linalg.norm(embeddings - embeddings[i], axis=1)
        dists[i] = np.inf
        nn_dists = np.sort(dists)[:k]
        nn_dists = nn_dists[nn_dists > 1e-10]
        if len(nn_dists) < 2:
            continue
        T_k = nn_dists[-1]
        log_ratios = np.log(T_k / nn_dists[:-1])
        if len(log_ratios) > 0:
            estimates.append(len(log_ratios) / np.sum(log_ratios))
    return float(np.mean(estimates)) if estimates else 0.0


def isotropy_score(embeddings: np.ndarray) -> float:
    """Isotropy: ratio of min to max eigenvalue of covariance (Mu et al., 2018).

    1.0 = perfectly isotropic, 0.0 = degenerate.
    """
    centered = embeddings - embeddings.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) < 2:
        return 0.0
    return float(eigvals.min() / eigvals.max())


def effective_rank(embeddings: np.ndarray) -> float:
    """Effective rank via normalized singular value entropy."""
    centered = embeddings - embeddings.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    sv = sv[sv > 1e-10]
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def singular_value_spectrum(embeddings: np.ndarray) -> np.ndarray:
    """Normalized singular value spectrum."""
    centered = embeddings - embeddings.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    return sv / (sv[0] + 1e-10)


def representation_similarity_analysis(X: np.ndarray, Y: np.ndarray) -> float:
    """RSA: Spearman correlation between pairwise distance matrices."""
    def _pdist(M):
        n = len(M)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(M[i] - M[j]))
        return np.array(dists)

    dx = _pdist(X)
    dy = _pdist(Y)
    rx = np.argsort(np.argsort(dx)).astype(float)
    ry = np.argsort(np.argsort(dy)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------- training dynamics ----------

def loss_landscape_1d(
    loss_fn,
    params: Dict[str, np.ndarray],
    direction: Dict[str, np.ndarray],
    n_points: int = 41,
    alpha_range: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """1D cross-section of the loss landscape along a direction."""
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    losses = []
    for alpha in alphas:
        perturbed = {k: params[k] + alpha * direction[k] for k in params}
        losses.append(loss_fn(perturbed))
    return alphas, np.array(losses)


def gradient_signal_to_noise(gradient_history: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """Gradient SNR per parameter group (higher = more stable learning signal)."""
    if len(gradient_history) < 2:
        return {}
    param_names = list(gradient_history[0].keys())
    results = {}
    for name in param_names:
        grads = np.stack([g[name].ravel() for g in gradient_history])
        mean_grad = grads.mean(axis=0)
        std_grad = grads.std(axis=0) + 1e-10
        snr = np.mean(np.abs(mean_grad) / std_grad)
        results[name] = float(snr)
    return results


def convergence_analysis(loss_history: List[float], window: int = 50) -> Dict[str, float]:
    """Analyze training convergence from loss history."""
    arr = np.array(loss_history)
    n = len(arr)
    if n < window:
        return {"converged": False, "final_loss": float(arr[-1]) if n > 0 else 0.0}
    smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
    recent_slope = 0.0
    if len(smoothed) >= 2:
        x = np.arange(min(window, len(smoothed)))
        y = smoothed[-len(x):]
        recent_slope = float(np.polyfit(x, y, 1)[0])
    oscillation = float(np.std(arr[-window:])) if n >= window else 0.0
    return {
        "converged": abs(recent_slope) < 1e-5,
        "final_loss": float(arr[-1]),
        "recent_slope": recent_slope,
        "oscillation": oscillation,
        "best_loss": float(arr.min()),
        "best_step": int(np.argmin(arr)),
    }


# ---------- benchmark suite ----------

@dataclass
class BenchmarkReport:
    probe_results: Optional[Dict] = None
    retrieval_results: Optional[Dict] = None
    clustering_results: Optional[Dict] = None
    representation_results: Optional[Dict] = None
    convergence_results: Optional[Dict] = None
    model_info: Dict = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        s = {}
        if self.probe_results:
            s["probe_accuracy"] = self.probe_results.get("accuracy", 0)
        if self.retrieval_results:
            s["mAP"] = self.retrieval_results.get("mAP", 0)
            s["MRR"] = self.retrieval_results.get("MRR", 0)
        if self.clustering_results:
            s["NMI"] = self.clustering_results.get("NMI", 0)
            s["ARI"] = self.clustering_results.get("ARI", 0)
        if self.representation_results:
            s["isotropy"] = self.representation_results.get("isotropy", 0)
            s["effective_rank"] = self.representation_results.get("effective_rank", 0)
        return s


class BenchmarkSuite:
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

    def run_probe(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict:
        n_classes = len(np.unique(train_labels))
        probe = ProbeClassifier(train_embeddings.shape[1], n_classes, self.config.seed)
        probe.fit(train_embeddings, train_labels,
                  lr=self.config.probe_lr, epochs=self.config.probe_epochs)
        return probe.evaluate(test_embeddings, test_labels)

    def run_retrieval(
        self,
        ref_embeddings: np.ndarray,
        ref_labels: np.ndarray,
        query_embeddings: np.ndarray,
        query_labels: np.ndarray,
    ) -> Dict:
        bench = RetrievalBenchmark()
        bench.build_index(ref_embeddings, ref_labels)
        return bench.evaluate_retrieval(query_embeddings, query_labels, self.config.k_values)

    def run_clustering(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        bench = ClusteringBenchmark()
        return bench.evaluate(embeddings, labels, self.config.n_clusters)

    def run_representation_analysis(self, embeddings: np.ndarray) -> Dict:
        return {
            "isotropy": isotropy_score(embeddings),
            "effective_rank": effective_rank(embeddings),
            "intrinsic_dim": intrinsic_dimensionality_mle(embeddings, k=5),
            "sv_spectrum": singular_value_spectrum(embeddings)[:10].tolist(),
        }

    def run_all(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        loss_history: Optional[List[float]] = None,
    ) -> BenchmarkReport:
        report = BenchmarkReport()
        tasks = self.config.tasks
        if "probe" in tasks:
            report.probe_results = self.run_probe(
                train_embeddings, train_labels, test_embeddings, test_labels)
        if "retrieval" in tasks:
            report.retrieval_results = self.run_retrieval(
                train_embeddings, train_labels, test_embeddings, test_labels)
        if "clustering" in tasks:
            report.clustering_results = self.run_clustering(
                np.vstack([train_embeddings, test_embeddings]),
                np.concatenate([train_labels, test_labels]),
            )
        if "representation" in tasks:
            report.representation_results = self.run_representation_analysis(
                np.vstack([train_embeddings, test_embeddings]))
        if loss_history:
            report.convergence_results = convergence_analysis(loss_history)
        return report

    @staticmethod
    def compare_models(reports: Dict[str, BenchmarkReport]) -> Dict[str, Dict[str, float]]:
        return {name: report.summary() for name, report in reports.items()}
