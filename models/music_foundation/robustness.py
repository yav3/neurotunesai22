"""Robustness verification layer for the music foundation model.

Pure numpy/scipy. Implements:
  - Adversarial `would_flip` test (SPSA-based)
  - Metamorphic invariance/equivariance tests
  - Calibration metrics (ECE, Brier, reliability diagram)
  - Conformal prediction (split-conformal, coverage guarantee)
  - Regime-shift evaluation (temporal split, MMD drift, sliding-window)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# 1. Adversarial robustness — would_flip
# ---------------------------------------------------------------------------

def _embed_signal(orchestrator: Any, signal: np.ndarray) -> np.ndarray:
    """Single-signal embedding via the orchestrator's pipeline."""
    return orchestrator.get_embeddings([signal])[0]


def would_flip(
    orchestrator: Any,
    signal: np.ndarray,
    eps: float = 0.01,
    n_iters: int = 20,
    seed: int = 0,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if an ε-ball perturbation flips the nearest-codebook assignment.

    Uses SPSA to find an adversarial direction in signal space, then projects
    onto the ε-ball.  Returns (flipped: bool, adversarial_signal | None).
    """
    rng = np.random.RandomState(seed)
    base_emb = _embed_signal(orchestrator, signal)

    q = orchestrator.model.quantizer
    base_proj = base_emb @ q.proj_W
    base_proj_g = base_proj.reshape(q.n_codebooks, q.codebook_dim)
    base_codes = []
    for g in range(q.n_codebooks):
        dists = np.sum((q.codebooks[g] - base_proj_g[g]) ** 2, axis=1)
        base_codes.append(int(np.argmin(dists)))

    best_adv = None
    for _ in range(n_iters):
        delta = rng.choice([-1.0, 1.0], size=signal.shape)
        c = eps * 0.1
        sig_plus = np.clip(signal + c * delta, -1.0, 1.0)
        sig_minus = np.clip(signal - c * delta, -1.0, 1.0)

        emb_p = _embed_signal(orchestrator, sig_plus)
        emb_m = _embed_signal(orchestrator, sig_minus)

        # gradient of embedding distance w.r.t. signal perturbation
        diff = np.linalg.norm(emb_p - base_emb) - np.linalg.norm(emb_m - base_emb)
        grad_approx = (diff / (2 * c)) * delta

        # step in the adversarial direction, project to ε-ball
        perturbation = eps * grad_approx / (np.linalg.norm(grad_approx) + 1e-12)
        adv_signal = np.clip(signal + perturbation, -1.0, 1.0)

        adv_emb = _embed_signal(orchestrator, adv_signal)
        adv_proj = adv_emb @ q.proj_W
        adv_proj_g = adv_proj.reshape(q.n_codebooks, q.codebook_dim)
        adv_codes = []
        for g in range(q.n_codebooks):
            dists = np.sum((q.codebooks[g] - adv_proj_g[g]) ** 2, axis=1)
            adv_codes.append(int(np.argmin(dists)))

        if adv_codes != base_codes:
            return True, adv_signal

    return False, None


# ---------------------------------------------------------------------------
# 2. Metamorphic tests
# ---------------------------------------------------------------------------

def test_permutation_invariance(
    orchestrator: Any,
    signal: np.ndarray,
    tolerance: float = 0.1,
    seed: int = 0,
) -> Tuple[bool, float]:
    """Permuting mel frequency bins should not change embedding beyond tolerance.

    Returns (passed, max_cosine_distance).
    """
    rng = np.random.RandomState(seed)
    base_emb = _embed_signal(orchestrator, signal)
    base_norm = np.linalg.norm(base_emb) + 1e-12

    features = orchestrator.feature_extractor.extract(signal)
    mel = features["mel"]  # (n_mels, T)
    perm = rng.permutation(mel.shape[0])
    mel_perm = mel[perm]

    features_perm = dict(features)
    features_perm["mel"] = mel_perm
    fused = orchestrator.integrator.forward(features_perm)
    enc = orchestrator.model.transformer
    x = fused.T
    x = x + enc.pos_enc[: x.shape[0]]
    for layer in enc.layers:
        x, _ = layer.forward(x)
    x = x * enc.final_gamma + enc.final_beta
    perm_emb = x.mean(axis=0)

    cos_dist = 1.0 - float(
        np.dot(base_emb, perm_emb) / (base_norm * (np.linalg.norm(perm_emb) + 1e-12))
    )
    return cos_dist <= tolerance, cos_dist


def test_rescaling_invariance(
    orchestrator: Any,
    signal: np.ndarray,
    scales: Sequence[float] = (0.5, 0.8, 1.2, 2.0),
    tolerance: float = 0.15,
) -> Tuple[bool, Dict[float, float]]:
    """Amplitude rescaling should preserve relative embedding direction.

    Returns (all_passed, {scale: cosine_distance}).
    """
    base_emb = _embed_signal(orchestrator, signal)
    base_norm = np.linalg.norm(base_emb) + 1e-12
    results: Dict[float, float] = {}
    all_ok = True
    for s in scales:
        scaled = np.clip(signal * s, -1.0, 1.0)
        emb_s = _embed_signal(orchestrator, scaled)
        cos_d = 1.0 - float(
            np.dot(base_emb, emb_s)
            / (base_norm * (np.linalg.norm(emb_s) + 1e-12))
        )
        results[s] = cos_d
        if cos_d > tolerance:
            all_ok = False
    return all_ok, results


def test_pitch_shift_equivariance(
    orchestrator: Any,
    signal: np.ndarray,
    semitones: float = 2.0,
) -> Tuple[bool, float]:
    """Pitch shift should rotate chroma features by the expected amount.

    Returns (equivariant, chroma_rotation_error_in_semitones).
    """
    sr = orchestrator.config.sr
    features_orig = orchestrator.feature_extractor.extract(signal)
    chroma_orig = features_orig["chroma"]  # (12, T)

    # resample to shift pitch
    ratio = 2.0 ** (semitones / 12.0)
    n_out = int(len(signal) / ratio)
    if n_out < 2:
        return True, 0.0
    x_old = np.linspace(0, len(signal) - 1, len(signal))
    x_new = np.linspace(0, len(signal) - 1, n_out)
    shifted = np.interp(x_new, x_old, signal)

    features_shift = orchestrator.feature_extractor.extract(shifted)
    chroma_shift = features_shift["chroma"]  # (12, T')

    # compare mean chroma profiles
    profile_orig = chroma_orig.mean(axis=1)  # (12,)
    profile_shift = chroma_shift.mean(axis=1)

    # expected rotation: semitones bins (mod 12)
    expected_shift = int(round(semitones)) % 12
    profile_expected = np.roll(profile_orig, expected_shift)

    # correlation-based error
    corr_expected = float(
        np.dot(profile_expected, profile_shift)
        / (np.linalg.norm(profile_expected) * np.linalg.norm(profile_shift) + 1e-12)
    )
    # convert correlation gap to approximate semitone error
    error_semitones = float(np.arccos(np.clip(corr_expected, -1, 1)) * 12 / np.pi)
    return error_semitones < 2.0, error_semitones


# ---------------------------------------------------------------------------
# 3. Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Naeini et al. 2015).

    Args:
        probs: (N,) predicted probabilities for the positive class.
        labels: (N,) binary ground-truth {0,1}.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(probs[mask].mean())
        avg_acc = float(labels[mask].mean())
        ece += (count / n) * abs(avg_conf - avg_acc)
    return ece


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error of probability forecasts."""
    return float(np.mean((np.asarray(probs) - np.asarray(labels)) ** 2))


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability diagram data.

    Returns:
        (mean_predicted, fraction_positive, count) — each length n_bins.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        c = int(mask.sum())
        counts[i] = c
        if c > 0:
            mean_pred[i] = float(probs[mask].mean())
            frac_pos[i] = float(labels[mask].mean())
    return mean_pred, frac_pos, counts


# ---------------------------------------------------------------------------
# 4. Conformal prediction
# ---------------------------------------------------------------------------

def calibrate_conformal(
    scores: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute conformal quantile threshold from calibration nonconformity scores.

    Split-conformal: threshold = ceil((1-α)(n+1))/n quantile of scores.
    Guarantees marginal coverage ≥ 1-α on exchangeable test data.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("inf")
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level))


def conformal_predict(
    nonconformity_fn: Callable[[np.ndarray, int], float],
    x: np.ndarray,
    classes: Sequence[int],
    threshold: float,
) -> List[int]:
    """Return prediction set C(x) = {y : s(x,y) ≤ threshold}.

    Args:
        nonconformity_fn: (x, y_candidate) → nonconformity score.
        x: input signal.
        classes: candidate label set.
        threshold: from calibrate_conformal.
    """
    return [y for y in classes if nonconformity_fn(x, y) <= threshold]


def conformal_coverage(
    nonconformity_fn: Callable[[np.ndarray, int], float],
    test_signals: List[np.ndarray],
    test_labels: List[int],
    classes: Sequence[int],
    threshold: float,
) -> float:
    """Empirical coverage: fraction of test points where true label ∈ C(x)."""
    if not test_signals:
        return 0.0
    covered = 0
    for sig, y_true in zip(test_signals, test_labels):
        pred_set = conformal_predict(nonconformity_fn, sig, classes, threshold)
        if y_true in pred_set:
            covered += 1
    return covered / len(test_signals)


# ---------------------------------------------------------------------------
# 5. Regime-shift evaluation
# ---------------------------------------------------------------------------

def temporal_split(
    clips: List[Any],
    train_frac: float = 0.7,
) -> Tuple[List[Any], List[Any]]:
    """Split clips by temporal order (NOT random). Preserves distribution shift."""
    n = len(clips)
    split = int(n * train_frac)
    return clips[:split], clips[split:]


def drift_detector(
    embeddings_t1: np.ndarray,
    embeddings_t2: np.ndarray,
    method: str = "mmd",
) -> Tuple[float, float]:
    """Detect distribution shift between two embedding populations.

    Args:
        embeddings_t1: (N1, D) embeddings from period 1.
        embeddings_t2: (N2, D) embeddings from period 2.
        method: "mmd" (Maximum Mean Discrepancy) or "ks" (Kolmogorov-Smirnov).

    Returns:
        (statistic, p_value).
    """
    e1 = np.asarray(embeddings_t1, dtype=np.float64)
    e2 = np.asarray(embeddings_t2, dtype=np.float64)

    if method == "ks":
        # per-dimension KS, Bonferroni-corrected
        d = e1.shape[1]
        max_stat = 0.0
        min_p = 1.0
        for j in range(d):
            stat_j, p_j = sp_stats.ks_2samp(e1[:, j], e2[:, j])
            if stat_j > max_stat:
                max_stat = stat_j
            min_p = min(min_p, p_j * d)  # Bonferroni
        return float(max_stat), float(min(min_p, 1.0))

    # MMD with Gaussian kernel (median heuristic for bandwidth)
    combined = np.vstack([e1, e2])
    dists_all = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=2)
    median_dist = float(np.median(dists_all[dists_all > 0]))
    sigma2 = median_dist if median_dist > 0 else 1.0

    n1, n2 = len(e1), len(e2)

    def _kernel_mean(a: np.ndarray, b: np.ndarray) -> float:
        d2 = np.sum((a[:, None] - b[None, :]) ** 2, axis=2)
        return float(np.mean(np.exp(-d2 / (2 * sigma2))))

    mmd2 = _kernel_mean(e1, e1) + _kernel_mean(e2, e2) - 2 * _kernel_mean(e1, e2)
    mmd_stat = float(np.sqrt(max(mmd2, 0.0)))

    # permutation test for p-value
    rng = np.random.RandomState(42)
    n_perm = 200
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n1 + n2)
        e1p = combined[perm[:n1]]
        e2p = combined[perm[n1:]]
        mmd2_p = _kernel_mean(e1p, e1p) + _kernel_mean(e2p, e2p) - 2 * _kernel_mean(e1p, e2p)
        if np.sqrt(max(mmd2_p, 0.0)) >= mmd_stat:
            count_ge += 1
    p_val = (count_ge + 1) / (n_perm + 1)
    return mmd_stat, p_val


def evaluate_under_drift(
    orchestrator: Any,
    clips: List[Any],
    window_size: int = 20,
    step: int = 10,
) -> List[Dict[str, Any]]:
    """Sliding-window evaluation showing performance under temporal drift.

    For each window: compute mean embedding, isotropy, intra-window variance,
    and drift statistic vs. the first window.
    """
    if len(clips) < window_size:
        return []

    signals = [c.signal for c in clips]

    # first window as reference
    ref_emb = orchestrator.get_embeddings(signals[:window_size])

    results = []
    for start in range(0, len(signals) - window_size + 1, step):
        end = start + window_size
        window_emb = orchestrator.get_embeddings(signals[start:end])
        intra_var = float(np.mean(np.var(window_emb, axis=0)))

        norms = np.linalg.norm(window_emb, axis=1, keepdims=True) + 1e-12
        normed = window_emb / norms
        cos_sim = normed @ normed.T
        iso = float(np.std(cos_sim[np.triu_indices(len(cos_sim), k=1)]))

        if start > 0:
            mmd_stat, mmd_p = drift_detector(ref_emb, window_emb, method="mmd")
        else:
            mmd_stat, mmd_p = 0.0, 1.0

        results.append({
            "window_start": start,
            "window_end": end,
            "intra_variance": intra_var,
            "isotropy_std": iso,
            "mmd_vs_first": mmd_stat,
            "mmd_p_value": mmd_p,
        })

    return results


# ---------------------------------------------------------------------------
# 6. RobustnessReport
# ---------------------------------------------------------------------------

@dataclass
class RobustnessReport:
    """Aggregated robustness verification results."""
    # adversarial
    would_flip_rate: float = 0.0
    n_flip_tested: int = 0

    # metamorphic
    permutation_invariance_pass: bool = True
    permutation_cos_dist: float = 0.0
    rescaling_invariance_pass: bool = True
    rescaling_distances: Dict[float, float] = field(default_factory=dict)
    pitch_equivariance_pass: bool = True
    pitch_error_semitones: float = 0.0

    # calibration (populated if class labels available)
    ece: Optional[float] = None
    brier: Optional[float] = None

    # conformal
    conformal_threshold: Optional[float] = None
    conformal_coverage: Optional[float] = None

    # drift
    drift_windows: List[Dict[str, Any]] = field(default_factory=list)
    max_mmd: float = 0.0
    drift_detected: bool = False

    def summary(self) -> str:
        lines = ["=== Robustness Report ==="]
        lines.append(f"Adversarial flip rate: {self.would_flip_rate:.2%} ({self.n_flip_tested} tested)")
        lines.append(f"Permutation invariance: {'PASS' if self.permutation_invariance_pass else 'FAIL'} (cos_dist={self.permutation_cos_dist:.4f})")
        lines.append(f"Rescaling invariance: {'PASS' if self.rescaling_invariance_pass else 'FAIL'} {self.rescaling_distances}")
        lines.append(f"Pitch equivariance: {'PASS' if self.pitch_equivariance_pass else 'FAIL'} (err={self.pitch_error_semitones:.2f} st)")
        if self.ece is not None:
            lines.append(f"ECE: {self.ece:.4f}")
        if self.brier is not None:
            lines.append(f"Brier: {self.brier:.4f}")
        if self.conformal_threshold is not None:
            lines.append(f"Conformal threshold: {self.conformal_threshold:.4f}, coverage: {self.conformal_coverage:.2%}")
        lines.append(f"Drift detected: {self.drift_detected} (max MMD={self.max_mmd:.4f})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Full suite runner
# ---------------------------------------------------------------------------

def run_robustness_suite(
    orchestrator: Any,
    n_flip_samples: int = 5,
    flip_eps: float = 0.01,
    drift_window: int = 10,
    seed: int = 42,
) -> RobustnessReport:
    """Run the complete robustness verification battery.

    Args:
        orchestrator: a trained PretrainOrchestrator with .clips populated.
        n_flip_samples: number of signals to test for adversarial flips.
        flip_eps: perturbation budget for would_flip.
        drift_window: window size for drift evaluation.
        seed: random seed.
    """
    rng = np.random.RandomState(seed)
    report = RobustnessReport()

    clips = orchestrator.clips
    if not clips:
        return report

    signals = [c.signal for c in clips]

    # --- adversarial ---
    n_test = min(n_flip_samples, len(signals))
    indices = rng.choice(len(signals), n_test, replace=False)
    flips = 0
    for idx in indices:
        flipped, _ = would_flip(orchestrator, signals[idx], eps=flip_eps, seed=seed + idx)
        if flipped:
            flips += 1
    report.would_flip_rate = flips / n_test
    report.n_flip_tested = n_test

    # --- metamorphic ---
    test_sig = signals[rng.randint(len(signals))]
    perm_pass, perm_dist = test_permutation_invariance(orchestrator, test_sig, seed=seed)
    report.permutation_invariance_pass = perm_pass
    report.permutation_cos_dist = perm_dist

    resc_pass, resc_dists = test_rescaling_invariance(orchestrator, test_sig)
    report.rescaling_invariance_pass = resc_pass
    report.rescaling_distances = resc_dists

    pitch_pass, pitch_err = test_pitch_shift_equivariance(orchestrator, test_sig)
    report.pitch_equivariance_pass = pitch_pass
    report.pitch_error_semitones = pitch_err

    # --- drift ---
    if len(clips) >= drift_window * 2:
        windows = evaluate_under_drift(orchestrator, clips, window_size=drift_window, step=max(1, drift_window // 2))
        report.drift_windows = windows
        if windows:
            mmds = [w["mmd_vs_first"] for w in windows]
            report.max_mmd = max(mmds)
            p_vals = [w["mmd_p_value"] for w in windows if w["mmd_vs_first"] > 0]
            report.drift_detected = any(p < 0.05 for p in p_vals)

    return report
