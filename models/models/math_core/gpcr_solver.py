"""GPCR Fiedler spectral bias — functional selectivity in 6D pathway space.

6×6 pathway affinity → normalized Laplacian → Fiedler partition →
bias factor ∈ [0,1] and log₂ functional selectivity ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PATHWAY_NAMES = ["Gs", "Gi/o", "Gq/11", "G12/13", "beta-arr1", "beta-arr2"]
N_PATHWAY = 6
IDX_BARR1, IDX_BARR2 = 4, 5


@dataclass
class BiasConfig:
    epsilon: float = 1e-7
    cross_talk_strength: float = 0.15
    barr_correlation: float = 0.75
    fsr_clip: float = 6.0


@dataclass
class BiasResult:
    bias_factor: float
    delta_phi_primary: float
    delta_phi_arrestin: float
    fiedler_vector: list[float]
    primary_channels: list[int]
    arrestin_channels: list[int]
    fiedler_gap: float
    confidence: float
    functional_selectivity_ratio: float


def build_pathway_affinity_matrix(x_base: np.ndarray,
                                  cross_talk: float = 0.15,
                                  barr_correlation: float = 0.75) -> np.ndarray:
    x = np.asarray(x_base, dtype=float)
    assert x.shape == (N_PATHWAY,)
    A = np.diag(x)
    for i in range(N_PATHWAY):
        for j in range(i + 1, N_PATHWAY):
            if i == IDX_BARR1 and j == IDX_BARR2:
                val = barr_correlation * np.sqrt(max(x[i], 0) * max(x[j], 0))
            else:
                val = cross_talk * np.sqrt(max(x[i], 0) * max(x[j], 0))
            A[i, j] = val
            A[j, i] = val
    return A


def compute_fiedler_partition(affinity: np.ndarray) -> tuple[np.ndarray, list[int], list[int], float]:
    D = np.diag(affinity.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))
    L = D_inv_sqrt @ (D - affinity) @ D_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler = eigenvectors[:, 1]
    primary = [i for i in range(N_PATHWAY) if fiedler[i] < 0]
    arrestin = [i for i in range(N_PATHWAY) if fiedler[i] >= 0]
    gap = float(eigenvalues[1] - eigenvalues[0])
    return fiedler, primary, arrestin, gap


def compute_bias(x_base: np.ndarray, x_complex: np.ndarray,
                 config: BiasConfig | None = None) -> BiasResult:
    cfg = config or BiasConfig()
    x_b = np.asarray(x_base, dtype=float)
    x_c = np.asarray(x_complex, dtype=float)
    A = build_pathway_affinity_matrix(x_b, cfg.cross_talk_strength, cfg.barr_correlation)
    fiedler, primary, arrestin, gap = compute_fiedler_partition(A)
    delta = x_c - x_b
    dp = float(np.mean(delta[primary])) if primary else 0.0
    da = float(np.mean(delta[arrestin])) if arrestin else 0.0
    denom = abs(dp) + abs(da) + cfg.epsilon
    bias_raw = (dp - da) / denom
    bf = 0.5 + 0.5 * bias_raw
    fsr = float(np.clip(np.log2((abs(dp) + cfg.epsilon) / (abs(da) + cfg.epsilon)),
                        -cfg.fsr_clip, cfg.fsr_clip))
    eigenvalues = np.linalg.eigvalsh(A)
    confidence = float(eigenvalues[1] / (eigenvalues[-1] + cfg.epsilon)) if eigenvalues[-1] > 0 else 0.0
    return BiasResult(
        bias_factor=float(np.clip(bf, 0, 1)),
        delta_phi_primary=dp,
        delta_phi_arrestin=da,
        fiedler_vector=fiedler.tolist(),
        primary_channels=primary,
        arrestin_channels=arrestin,
        fiedler_gap=gap,
        confidence=confidence,
        functional_selectivity_ratio=fsr,
    )
