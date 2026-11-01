"""Pre-training objectives for the music foundation model.

Implements InfoNCE contrastive learning, masked feature prediction,
and cross-modal alignment objectives.
"""
from __future__ import annotations

import numpy as np

from .encoder import MusicFoundationModel, EncoderConfig, layer_norm


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


class InfoNCELoss:
    def __init__(self, temperature: float = 0.07) -> None:
        self.temperature = temperature

    def __call__(self, anchors: np.ndarray, positives: np.ndarray) -> float:
        sim = cosine_similarity_matrix(anchors, positives) / self.temperature
        n = sim.shape[0]
        sim_max = sim.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(sim - sim_max), axis=1)) + sim_max.flatten()
        positive_sim = np.diag(sim)
        loss = -positive_sim + log_sum_exp
        return float(loss.mean())


class MaskedFeaturePrediction:
    def __init__(self, mask_ratio: float = 0.15, seed: int = 42) -> None:
        self.mask_ratio = mask_ratio
        self.rng = np.random.RandomState(seed)

    def mask_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        masked = features.copy()
        mask = self.rng.random(features.shape) < self.mask_ratio
        masked[mask] = 0.0
        return masked, mask

    def loss(self, predicted: np.ndarray, original: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        diff = (predicted - original) ** 2
        return float(diff[mask].mean())


class CrossModalAlignment:
    def __init__(self, temperature: float = 0.1) -> None:
        self.info_nce = InfoNCELoss(temperature)

    def __call__(self, audio_embs: np.ndarray, semantic_embs: np.ndarray) -> float:
        loss_a2s = self.info_nce(audio_embs, semantic_embs)
        loss_s2a = self.info_nce(semantic_embs, audio_embs)
        return (loss_a2s + loss_s2a) / 2


class PretrainingObjective:
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        masked_weight: float = 0.5,
        alignment_weight: float = 0.5,
        temperature: float = 0.07,
        mask_ratio: float = 0.15,
    ) -> None:
        self.contrastive_weight = contrastive_weight
        self.masked_weight = masked_weight
        self.alignment_weight = alignment_weight
        self.info_nce = InfoNCELoss(temperature)
        self.masked_pred = MaskedFeaturePrediction(mask_ratio)
        self.alignment = CrossModalAlignment(temperature)

    def __call__(
        self,
        audio_embs: np.ndarray,
        semantic_embs: np.ndarray,
        fused_embs: np.ndarray,
        original_features: np.ndarray | None = None,
        predicted_features: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        contrastive = self.info_nce(fused_embs, fused_embs)

        masked = 0.0
        if original_features is not None and predicted_features is not None and mask is not None:
            masked = self.masked_pred.loss(predicted_features, original_features, mask)

        alignment = self.alignment(audio_embs, semantic_embs)

        total = (
            self.contrastive_weight * contrastive
            + self.masked_weight * masked
            + self.alignment_weight * alignment
        )

        return {
            "total": total,
            "contrastive": contrastive,
            "masked": masked,
            "alignment": alignment,
        }
