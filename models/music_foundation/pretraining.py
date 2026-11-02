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


# ---------------------------------------------------------------------------
# VICReg loss (Bardes et al., 2022)
# ---------------------------------------------------------------------------

class VICRegLoss:
    """Variance–Invariance–Covariance Regularization."""

    def __init__(
        self,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
    ) -> None:
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def __call__(self, z1: np.ndarray, z2: np.ndarray) -> dict[str, float]:
        n, d = z1.shape
        inv = float(np.mean((z1 - z2) ** 2))
        std1 = np.sqrt(z1.var(axis=0) + 1e-4)
        std2 = np.sqrt(z2.var(axis=0) + 1e-4)
        var = float(np.mean(np.maximum(0, 1.0 - std1)) + np.mean(np.maximum(0, 1.0 - std2))) / 2
        z1c = z1 - z1.mean(axis=0)
        z2c = z2 - z2.mean(axis=0)
        cov1 = (z1c.T @ z1c) / max(n - 1, 1)
        cov2 = (z2c.T @ z2c) / max(n - 1, 1)
        off1 = cov1 ** 2
        np.fill_diagonal(off1, 0)
        off2 = cov2 ** 2
        np.fill_diagonal(off2, 0)
        cov_loss = float(off1.sum() + off2.sum()) / (2 * d)
        total = self.lambda_inv * inv + self.lambda_var * var + self.lambda_cov * cov_loss
        return {"total": total, "invariance": inv, "variance": var, "covariance": cov_loss}


# ---------------------------------------------------------------------------
# Barlow Twins loss (Zbontar et al., 2021)
# ---------------------------------------------------------------------------

class BarlowTwinsLoss:
    """Cross-correlation matrix loss with off-diagonal penalty."""

    def __init__(self, lambda_bt: float = 0.005) -> None:
        self.lambda_bt = lambda_bt

    def __call__(self, z1: np.ndarray, z2: np.ndarray) -> dict[str, float]:
        n, d = z1.shape
        z1n = (z1 - z1.mean(axis=0)) / (z1.std(axis=0) + 1e-8)
        z2n = (z2 - z2.mean(axis=0)) / (z2.std(axis=0) + 1e-8)
        cc = (z1n.T @ z2n) / max(n, 1)
        diag = np.diag(cc)
        on_diag = float(np.sum((diag - 1) ** 2))
        off_diag_mask = ~np.eye(d, dtype=bool)
        off_diag = float(np.sum(cc[off_diag_mask] ** 2))
        total = on_diag + self.lambda_bt * off_diag
        return {"total": total, "on_diagonal": on_diag, "off_diagonal": off_diag}


# ---------------------------------------------------------------------------
# BYOL-style prediction loss (Grill et al., 2020)
# ---------------------------------------------------------------------------

def _batch_norm_1d(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    var = x.var(axis=0) + 1e-5
    return (x - mean) / np.sqrt(var)


class BYOLLoss:
    """Asymmetric prediction: online → target, no negatives."""

    def __init__(self, input_dim: int, predictor_dim: int = 128, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, predictor_dim).astype(np.float64) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(predictor_dim)
        self.W2 = rng.randn(predictor_dim, input_dim).astype(np.float64) * np.sqrt(2.0 / predictor_dim)
        self.b2 = np.zeros(input_dim)

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = _batch_norm_1d(x @ self.W1 + self.b1)
        h = np.maximum(h, 0)  # ReLU
        return h @ self.W2 + self.b2

    def __call__(self, online: np.ndarray, target: np.ndarray) -> float:
        pred = self.predict(online)
        pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-12)
        tgt_norm = target / (np.linalg.norm(target, axis=-1, keepdims=True) + 1e-12)
        return float(2 - 2 * np.mean(np.sum(pred_norm * tgt_norm, axis=-1)))


# ---------------------------------------------------------------------------
# Masked Acoustic Model — wav2vec 2.0 span masking + L1 loss
# ---------------------------------------------------------------------------

class MaskedAcousticModel:
    """Span masking strategy following Baevski et al. (2020)."""

    def __init__(
        self,
        mask_length: int = 10,
        mask_prob: float = 0.065,
        min_masks: int = 1,
        seed: int = 42,
    ) -> None:
        self.mask_length = mask_length
        self.mask_prob = mask_prob
        self.min_masks = min_masks
        self.rng = np.random.RandomState(seed)

    def generate_mask(self, seq_len: int) -> np.ndarray:
        n_masks = max(int(seq_len * self.mask_prob / self.mask_length), self.min_masks)
        starts = self.rng.choice(max(seq_len - self.mask_length, 1), size=n_masks, replace=True)
        mask = np.zeros(seq_len, dtype=bool)
        for s in starts:
            mask[s:s + self.mask_length] = True
        return mask

    def loss(self, predicted: np.ndarray, original: np.ndarray, mask: np.ndarray) -> float:
        if mask.ndim == 1 and predicted.ndim == 2:
            mask_2d = mask[:, None].repeat(predicted.shape[1], axis=1)
        else:
            mask_2d = mask
        if mask_2d.sum() == 0:
            return 0.0
        return float(np.abs(predicted[mask_2d] - original[mask_2d]).mean())


# ---------------------------------------------------------------------------
# Multi-task pretraining with uncertainty weighting (Kendall et al., 2018)
# ---------------------------------------------------------------------------

class MultiTaskPretraining:
    """Combines named objectives with learnable uncertainty weights.

    total = Σ_i  (1 / (2 σ_i²)) L_i  +  log σ_i
    where log_sigma is the learned parameter (initialized to 0 → σ=1).
    """

    def __init__(
        self,
        objective_names: list[str],
        objectives: list[Callable],
        fixed_weights: dict[str, float] | None = None,
        learnable: bool = True,
    ) -> None:
        self.names = objective_names
        self.objectives = objectives
        self.learnable = learnable
        self.log_sigma = {name: 0.0 for name in objective_names}
        self.fixed_weights = fixed_weights or {n: 1.0 for n in objective_names}

    def __call__(self, *args, **kwargs) -> dict[str, float]:
        results: dict[str, float] = {}
        total = 0.0
        for name, obj in zip(self.names, self.objectives):
            val = obj(*args, **kwargs)
            if isinstance(val, dict):
                val = val.get("total", 0.0)
            results[name] = float(val)
            if self.learnable:
                sigma2 = np.exp(2 * self.log_sigma[name])
                total += float(val) / (2 * sigma2) + self.log_sigma[name]
            else:
                total += self.fixed_weights.get(name, 1.0) * float(val)
        results["total"] = total
        return results


# ---------------------------------------------------------------------------
# Codebook diversity loss (entropy maximisation)
# ---------------------------------------------------------------------------

class CodebookDiversityLoss:
    """Encourage uniform codebook usage via soft-assignment entropy."""

    def __init__(self, n_codebooks: int = 1, codebook_size: int = 320) -> None:
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size

    def __call__(self, soft_assignments: np.ndarray) -> float:
        if soft_assignments.ndim == 1:
            soft_assignments = soft_assignments.reshape(1, -1)
        avg = soft_assignments.mean(axis=0) + 1e-10
        avg = avg / avg.sum()
        entropy = -np.sum(avg * np.log(avg))
        max_entropy = np.log(float(self.codebook_size))
        return float(max_entropy - entropy)


# ---------------------------------------------------------------------------
# Temporal order prediction (pretext task)
# ---------------------------------------------------------------------------

class TemporalOrderPrediction:
    """Predict correct permutation of shuffled segments."""

    def __init__(self, n_segments: int = 4, dim: int = 64, seed: int = 0) -> None:
        self.n_segments = n_segments
        from math import factorial
        self.n_perms = min(factorial(n_segments), 24)
        rng = np.random.RandomState(seed)
        self.W = rng.randn(n_segments * dim, self.n_perms).astype(np.float64) * 0.02
        self.b = np.zeros(self.n_perms)

    def predict_logits(self, segment_embeddings: np.ndarray) -> np.ndarray:
        flat = segment_embeddings.flatten()
        return flat @ self.W + self.b

    def loss(self, segment_embeddings: np.ndarray, correct_perm_idx: int) -> float:
        logits = self.predict_logits(segment_embeddings)
        log_max = logits.max()
        log_sum_exp = np.log(np.sum(np.exp(logits - log_max))) + log_max
        return float(-logits[correct_perm_idx] + log_sum_exp)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    rng = np.random.RandomState(0)
    n, d = 16, 32

    z1 = rng.randn(n, d)
    z2 = z1 + rng.randn(n, d) * 0.1

    # VICReg
    vic = VICRegLoss()
    r = vic(z1, z2)
    assert "total" in r and "invariance" in r, "VICReg keys"
    assert r["invariance"] < 1.0, "VICReg invariance small for similar inputs"

    # Barlow Twins
    bt = BarlowTwinsLoss()
    r = bt(z1, z2)
    assert "total" in r, "BT keys"
    assert r["total"] >= 0, "BT non-negative"

    # BYOL
    byol = BYOLLoss(d, 16)
    loss = byol(z1, z2)
    assert 0 <= loss <= 4.0, f"BYOL loss range: {loss}"

    # MAM
    mam = MaskedAcousticModel(mask_length=3, mask_prob=0.3)
    mask = mam.generate_mask(50)
    assert mask.shape == (50,), "MAM mask shape"
    assert mask.any(), "MAM should mask something"
    pred = rng.randn(50, d)
    orig = rng.randn(50, d)
    l = mam.loss(pred, orig, mask)
    assert l > 0, "MAM loss positive"

    # MultiTask
    obj1 = lambda *a, **k: 1.5
    obj2 = lambda *a, **k: 0.3
    mt = MultiTaskPretraining(["a", "b"], [obj1, obj2])
    r = mt()
    assert "total" in r and "a" in r and "b" in r, "MT keys"

    # Codebook diversity
    cdl = CodebookDiversityLoss(codebook_size=8)
    uniform = np.ones(8) / 8
    peaked = np.zeros(8)
    peaked[0] = 1.0
    assert cdl(uniform) < cdl(peaked), "Uniform should have lower diversity loss"

    # Temporal order
    top = TemporalOrderPrediction(n_segments=3, dim=d)
    segs = rng.randn(3, d)
    l = top.loss(segs, 0)
    assert l > 0, "TOP loss positive"

    parts = [
        "VICReg", "BarlowTwins", "BYOL", "MAM", "MultiTask",
        "CodebookDiversity", "TemporalOrder",
    ]
    print(f"Self-test passed: {', '.join(parts)}.")
