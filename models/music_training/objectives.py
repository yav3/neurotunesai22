"""Training objectives: contrastive, reconstruction, multi-task."""
from __future__ import annotations

import numpy as np


class ContrastiveLoss:
    """InfoNCE-style contrastive loss on embedding pairs."""

    def __init__(self, temperature: float = 0.07) -> None:
        self.temperature = temperature

    def __call__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        n = len(embeddings)
        if n < 2:
            return 0.0

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = embeddings / norms

        sim = normed @ normed.T / self.temperature

        mask_pos = labels[:, None] == labels[None, :]
        np.fill_diagonal(mask_pos, False)

        if not mask_pos.any():
            return 0.0

        exp_sim = np.exp(sim - np.max(sim, axis=1, keepdims=True))
        np.fill_diagonal(exp_sim, 0.0)

        denom = exp_sim.sum(axis=1)
        denom = np.maximum(denom, 1e-8)

        loss = 0.0
        count = 0
        for i in range(n):
            pos_indices = np.where(mask_pos[i])[0]
            if len(pos_indices) == 0:
                continue
            for j in pos_indices:
                loss -= np.log(exp_sim[i, j] / denom[i] + 1e-8)
                count += 1

        return float(loss / max(count, 1))


class ReconstructionLoss:
    """MSE reconstruction loss on feature vectors."""

    def __call__(self, predicted: np.ndarray, target: np.ndarray) -> float:
        diff = predicted - target
        return float(np.mean(diff ** 2))

    def per_feature(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        diff = predicted - target
        return np.mean(diff ** 2, axis=0)


class MultiTaskHead:
    """Combines classification cross-entropy + regression MSE losses."""

    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 1.0,
        n_classes: int = 10,
    ) -> None:
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.n_classes = n_classes

    def classification_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        n = len(labels)
        log_probs = np.log(probs[np.arange(n), labels.astype(int)] + 1e-8)
        return float(-np.mean(log_probs))

    def regression_loss(self, predicted: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((predicted - target) ** 2))

    def __call__(
        self,
        logits: np.ndarray,
        class_labels: np.ndarray,
        reg_predicted: np.ndarray,
        reg_target: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        cls_loss = self.classification_loss(logits, class_labels)
        reg_loss = self.regression_loss(reg_predicted, reg_target)
        total = self.classification_weight * cls_loss + self.regression_weight * reg_loss
        return total, {
            "classification": cls_loss,
            "regression": reg_loss,
            "total": total,
        }
