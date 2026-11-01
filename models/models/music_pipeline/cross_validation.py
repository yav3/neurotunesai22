"""Cross-validation and evaluation for music ML pipelines."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FoldResult:
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_score: float = 0.0
    val_score: float = 0.0
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class CVResult:
    folds: list[FoldResult]
    metric_name: str = "accuracy"

    @property
    def mean_train(self) -> float:
        return float(np.mean([f.train_score for f in self.folds]))

    @property
    def mean_val(self) -> float:
        return float(np.mean([f.val_score for f in self.folds]))

    @property
    def std_val(self) -> float:
        return float(np.std([f.val_score for f in self.folds]))

    @property
    def overfitting_gap(self) -> float:
        return self.mean_train - self.mean_val


def k_fold_split(n_samples: int, k: int = 5, shuffle: bool = True,
                 seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    fold_size = n_samples // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, val_idx))
    return folds


def stratified_k_fold(labels: np.ndarray, k: int = 5,
                      seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    class_indices: dict[int, np.ndarray] = {}
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        class_indices[int(lab)] = idx
    folds: list[tuple[list[int], list[int]]] = [([], []) for _ in range(k)]
    for lab_idx in class_indices.values():
        fold_size = len(lab_idx) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(lab_idx)
            for j in range(len(lab_idx)):
                if start <= j < end:
                    folds[i][1].append(int(lab_idx[j]))
                else:
                    folds[i][0].append(int(lab_idx[j]))
    return [(np.array(sorted(t)), np.array(sorted(v))) for t, v in folds]


def leave_one_out(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.delete(np.arange(n_samples), i), np.array([i]))
            for i in range(n_samples)]


def group_k_fold(groups: np.ndarray, k: int = 5,
                 seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_groups = len(unique_groups)
    fold_size = n_groups // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_groups
        val_groups = set(unique_groups[start:end].tolist())
        val_idx = np.array([j for j in range(len(groups)) if groups[j] in val_groups])
        train_idx = np.array([j for j in range(len(groups)) if groups[j] not in val_groups])
        folds.append((train_idx, val_idx))
    return folds


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall < 1e-10:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int = 0) -> np.ndarray:
    if n_classes == 0:
        n_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        cm[int(t), int(p)] += 1
    return cm


def learning_curve_points(train_sizes: list[float],
                          n_total: int,
                          k: int = 5) -> list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]]:
    result = []
    for frac in train_sizes:
        n = max(int(n_total * frac), k)
        splits = k_fold_split(n, k=k)
        result.append((n, splits))
    return result
