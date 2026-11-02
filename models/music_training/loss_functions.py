"""Specialized loss functions for music ML training."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class LossResult:
    value: float
    grad: Optional[np.ndarray] = None


def spectral_convergence(
    pred_mag: np.ndarray, target_mag: np.ndarray,
) -> float:
    diff_norm = np.linalg.norm(target_mag - pred_mag, ord='fro')
    target_norm = np.linalg.norm(target_mag, ord='fro')
    if target_norm < 1e-12:
        return 0.0
    return float(diff_norm / target_norm)


def log_stft_magnitude_loss(
    pred_mag: np.ndarray, target_mag: np.ndarray,
    eps: float = 1e-7,
) -> float:
    log_pred = np.log(pred_mag + eps)
    log_target = np.log(target_mag + eps)
    return float(np.mean(np.abs(log_target - log_pred)))


def multi_resolution_stft_loss(
    pred: np.ndarray, target: np.ndarray,
    fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
    hop_ratio: float = 0.25,
) -> float:
    total = 0.0
    for n_fft in fft_sizes:
        hop = int(n_fft * hop_ratio)
        win = np.hanning(n_fft)
        n_frames_p = (len(pred) - n_fft) // hop + 1
        n_frames_t = (len(target) - n_fft) // hop + 1
        n_frames = min(n_frames_p, n_frames_t)
        if n_frames < 1:
            continue
        pred_mag = np.zeros((n_fft // 2 + 1, n_frames))
        tgt_mag = np.zeros_like(pred_mag)
        for i in range(n_frames):
            start = i * hop
            p_frame = pred[start:start + n_fft] * win
            t_frame = target[start:start + n_fft] * win
            pred_mag[:, i] = np.abs(np.fft.rfft(p_frame))
            tgt_mag[:, i] = np.abs(np.fft.rfft(t_frame))
        sc = spectral_convergence(pred_mag, tgt_mag)
        lm = log_stft_magnitude_loss(pred_mag, tgt_mag)
        total += sc + lm
    return total / len(fft_sizes)


def mel_weighted_loss(
    pred_mag: np.ndarray, target_mag: np.ndarray,
    mel_weights: np.ndarray,
) -> float:
    pred_mel = mel_weights @ pred_mag
    tgt_mel = mel_weights @ target_mag
    return float(np.mean((tgt_mel - pred_mel) ** 2))


def reconstruction_loss(
    pred: np.ndarray, target: np.ndarray,
    loss_type: str = "mse",
) -> float:
    if loss_type == "mse":
        return float(np.mean((target - pred) ** 2))
    elif loss_type == "mae":
        return float(np.mean(np.abs(target - pred)))
    elif loss_type == "huber":
        delta = 1.0
        diff = np.abs(target - pred)
        quad = np.minimum(diff, delta)
        linear = diff - quad
        return float(np.mean(0.5 * quad ** 2 + delta * linear))
    return float(np.mean((target - pred) ** 2))


def kl_divergence_gaussian(
    mu: np.ndarray, log_var: np.ndarray,
) -> float:
    return float(-0.5 * np.mean(1.0 + log_var - mu ** 2 - np.exp(log_var)))


def vae_loss(
    pred: np.ndarray, target: np.ndarray,
    mu: np.ndarray, log_var: np.ndarray,
    beta: float = 1.0,
) -> Tuple[float, float, float]:
    recon = reconstruction_loss(pred, target, "mse")
    kl = kl_divergence_gaussian(mu, log_var)
    return recon + beta * kl, recon, kl


def contrastive_loss(
    anchor: np.ndarray, positive: np.ndarray,
    negatives: np.ndarray, temperature: float = 0.07,
) -> float:
    a_norm = anchor / (np.linalg.norm(anchor) + 1e-12)
    p_norm = positive / (np.linalg.norm(positive) + 1e-12)
    n_norms = negatives / (np.linalg.norm(negatives, axis=1, keepdims=True) + 1e-12)
    pos_sim = np.dot(a_norm, p_norm) / temperature
    neg_sims = n_norms @ a_norm / temperature
    max_val = max(pos_sim, np.max(neg_sims))
    log_sum_exp = max_val + np.log(
        np.exp(pos_sim - max_val) + np.sum(np.exp(neg_sims - max_val))
    )
    return float(-pos_sim + log_sum_exp)


def triplet_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray,
    margin: float = 1.0,
) -> float:
    d_pos = float(np.sum((anchor - positive) ** 2))
    d_neg = float(np.sum((anchor - negative) ** 2))
    return max(d_pos - d_neg + margin, 0.0)


def feature_matching_loss(
    real_features: np.ndarray, fake_features: np.ndarray,
) -> float:
    return float(np.mean((np.mean(real_features, axis=0) - np.mean(fake_features, axis=0)) ** 2))


def adversarial_loss_generator(fake_scores: np.ndarray) -> float:
    return float(np.mean((fake_scores - 1.0) ** 2))


def adversarial_loss_discriminator(
    real_scores: np.ndarray, fake_scores: np.ndarray,
) -> float:
    real_loss = float(np.mean((real_scores - 1.0) ** 2))
    fake_loss = float(np.mean(fake_scores ** 2))
    return 0.5 * (real_loss + fake_loss)


def hard_negative_mining(
    anchor: np.ndarray, negatives: np.ndarray, k: int = 5,
) -> np.ndarray:
    dists = np.sum((negatives - anchor) ** 2, axis=1)
    indices = np.argsort(dists)[:k]
    return negatives[indices]


def weighted_mse(
    pred: np.ndarray, target: np.ndarray, weights: np.ndarray,
) -> float:
    return float(np.mean(weights * (target - pred) ** 2))


def focal_loss(
    probs: np.ndarray, targets: np.ndarray,
    gamma: float = 2.0, alpha: float = 0.25,
) -> float:
    eps = 1e-7
    p_t = np.where(targets == 1, probs, 1.0 - probs)
    p_t = np.clip(p_t, eps, 1.0 - eps)
    a_t = np.where(targets == 1, alpha, 1.0 - alpha)
    return float(-np.mean(a_t * (1.0 - p_t) ** gamma * np.log(p_t)))
