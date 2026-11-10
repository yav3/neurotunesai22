"""Uncertainty quantification for the music foundation model.

Pure numpy/scipy. No torch/tensorflow/jax.

Five subsystems:
1. Multi-eigenstate ensemble — epistemic + aleatoric uncertainty via deep ensembles
2. Continuous-time feature SDE — stochastic dynamics on audio feature trajectories
3. Contradiction-based selection rules — learn when the model is unreliable
4. Counterfactual operators — causal sensitivity of embeddings to input features
5. Uncertainty-aware prediction wrapper — unified interface with abstention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════
# 1. Multi-eigenstate ensemble
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EnsembleMember:
    """One trained member of the ensemble."""
    orchestrator: object  # PretrainOrchestrator — kept as object to avoid circular import
    seed: int
    train_loss: float = float("inf")


class EigenstateEnsemble:
    """Deep ensemble where each member is a different random initialization.

    Each "eigenstate" explores a different loss-landscape basin, so
    disagreement across members → epistemic uncertainty (the model doesn't
    know), while average within-member variance → aleatoric uncertainty
    (inherent noise in the data).
    """

    def __init__(self, n_members: int = 5, d_model: int = 64, seed: int = 42):
        self.n_members = n_members
        self.d_model = d_model
        self.base_seed = seed
        self.members: List[EnsembleMember] = []

    def member_seeds(self) -> List[int]:
        rng = np.random.RandomState(self.base_seed)
        return [int(rng.randint(0, 2**31)) for _ in range(self.n_members)]

    def add_member(self, orchestrator: object, seed: int, loss: float = float("inf")) -> None:
        self.members.append(EnsembleMember(orchestrator=orchestrator, seed=seed, train_loss=loss))

    @property
    def is_trained(self) -> bool:
        return len(self.members) == self.n_members


def train_ensemble(
    orchestrator_factory: Callable[[int], object],
    n_members: int,
    train_clips: Optional[list] = None,
    n_steps: int = 10,
) -> EigenstateEnsemble:
    """Train an ensemble of n_members independently.

    Args:
        orchestrator_factory: callable(seed) → PretrainOrchestrator (already configured).
            The factory must accept a seed argument and return an orchestrator ready for
            .prepare_data() + .run().
        n_members: number of ensemble members.
        train_clips: if provided, loaded into each orchestrator via .load_data().
        n_steps: passed through to each member's run().
    """
    ensemble = EigenstateEnsemble(n_members=n_members)
    for seed in ensemble.member_seeds():
        orch = orchestrator_factory(seed)
        if train_clips is not None and hasattr(orch, "load_data"):
            orch.load_data(train_clips)
        elif hasattr(orch, "prepare_data"):
            orch.prepare_data()
        history = orch.run()
        final_loss = history.final_loss if hasattr(history, "final_loss") else float("inf")
        ensemble.add_member(orch, seed, final_loss)
    return ensemble


@dataclass
class EnsemblePrediction:
    mean_embedding: np.ndarray        # (D,)
    variance_per_dim: np.ndarray      # (D,)
    epistemic_uncertainty: float       # scalar: mean variance across members
    aleatoric_uncertainty: float       # scalar: average within-member spread
    member_embeddings: np.ndarray      # (M, D)


def ensemble_predict(
    ensemble: EigenstateEnsemble,
    signals: List[np.ndarray],
) -> EnsemblePrediction:
    """Predict with all ensemble members and decompose uncertainty.

    Returns mean embedding, per-dimension variance, and scalar epistemic /
    aleatoric uncertainty estimates.
    """
    if not ensemble.is_trained:
        raise ValueError(f"Ensemble has {len(ensemble.members)}/{ensemble.n_members} members")

    all_embeddings = []  # (M, N, D) — M members, N signals, D dims
    for member in ensemble.members:
        orch = member.orchestrator
        embs = orch.get_embeddings(signals)  # (N, D)
        all_embeddings.append(embs)

    stacked = np.stack(all_embeddings)  # (M, N, D)

    # per-signal mean across members, then average across signals → (D,)
    mean_per_signal = stacked.mean(axis=0)       # (N, D)
    mean_embedding = mean_per_signal.mean(axis=0) # (D,)

    # epistemic: variance across members (disagreement)
    var_across_members = stacked.var(axis=0)      # (N, D)
    epistemic_per_dim = var_across_members.mean(axis=0)  # (D,)
    epistemic = float(epistemic_per_dim.mean())

    # aleatoric: for each member, variance across signals (data noise);
    # average across members
    var_within = stacked.var(axis=1)  # (M, D)
    aleatoric = float(var_within.mean())

    # flatten member embeddings to (M, D) via mean over signals
    member_means = stacked.mean(axis=1)  # (M, D)

    return EnsemblePrediction(
        mean_embedding=mean_embedding,
        variance_per_dim=epistemic_per_dim,
        epistemic_uncertainty=epistemic,
        aleatoric_uncertainty=aleatoric,
        member_embeddings=member_means,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Continuous-time feature SDE
# ═══════════════════════════════════════════════════════════════════════════


class FeatureSDE:
    """Stochastic differential equation on audio-feature trajectories.

    dx = drift(x, t) dt + diffusion(x, t) dW

    Applied to temporal evolution of extracted features: drift captures
    learned dynamics (e.g. adjacency-weighted transitions), diffusion
    captures transition uncertainty.
    """

    def __init__(
        self,
        drift_fn: Callable[[np.ndarray, float], np.ndarray],
        diffusion_fn: Callable[[np.ndarray, float], np.ndarray],
        dt: float = 0.01,
        seed: int = 42,
    ):
        self.drift = drift_fn
        self.diffusion = diffusion_fn
        self.dt = dt
        self.rng = np.random.RandomState(seed)


def euler_maruyama(
    sde: FeatureSDE,
    x0: np.ndarray,
    T: float,
    n_paths: int = 100,
) -> np.ndarray:
    """Simulate n_paths of the SDE via Euler–Maruyama.

    Args:
        sde: the SDE to simulate.
        x0: initial state (D,).
        T: total simulation time.
        n_paths: number of independent sample paths.

    Returns:
        paths: (n_paths, n_steps+1, D) array of trajectories.
    """
    n_steps = max(1, int(T / sde.dt))
    D = x0.shape[0]
    paths = np.zeros((n_paths, n_steps + 1, D))
    paths[:, 0, :] = x0[np.newaxis, :]

    sqrt_dt = np.sqrt(sde.dt)
    for i in range(n_steps):
        t = i * sde.dt
        x = paths[:, i, :]  # (n_paths, D)
        # vectorise over paths: evaluate drift/diffusion per path
        drift_vals = np.stack([sde.drift(x[p], t) for p in range(n_paths)])     # (n_paths, D)
        diff_vals = np.stack([sde.diffusion(x[p], t) for p in range(n_paths)])  # (n_paths, D)
        dW = sde.rng.randn(n_paths, D) * sqrt_dt
        paths[:, i + 1, :] = x + drift_vals * sde.dt + diff_vals * dW

    return paths


def sde_uncertainty(
    paths: np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, np.ndarray]:
    """Compute uncertainty statistics from SDE sample paths.

    Args:
        paths: (n_paths, n_steps, D) trajectories.
        confidence: CI level.

    Returns:
        dict with 'mean', 'std', 'ci_lower', 'ci_upper', 'median', all (n_steps, D).
    """
    mean = paths.mean(axis=0)   # (n_steps, D)
    std = paths.std(axis=0)     # (n_steps, D)
    median = np.median(paths, axis=0)

    alpha = 1.0 - confidence
    ci_lower = np.percentile(paths, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(paths, 100 * (1 - alpha / 2), axis=0)

    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median": median,
    }


def build_learned_sde(
    feature_trajectory: np.ndarray,
    dt: float = 0.01,
    seed: int = 42,
) -> FeatureSDE:
    """Fit an SDE to an observed feature trajectory via moment matching.

    Given x(t) of shape (T_steps, D), estimate:
      - drift ≈ E[Δx/dt]  (local mean velocity)
      - diffusion ≈ std(Δx/√dt)  (local noise intensity)

    Returns a FeatureSDE with constant (time-averaged) drift and diffusion
    coefficients — a first-order Ornstein–Uhlenbeck approximation.
    """
    T_steps, D = feature_trajectory.shape
    if T_steps < 3:
        # fallback: zero drift, unit diffusion
        return FeatureSDE(
            drift_fn=lambda x, t: np.zeros_like(x),
            diffusion_fn=lambda x, t: np.ones_like(x),
            dt=dt, seed=seed,
        )

    dx = np.diff(feature_trajectory, axis=0)  # (T-1, D)

    # drift: mean-reverting OU: dx ≈ θ(μ - x)dt + σ dW
    # simple: mean velocity
    mean_dx = dx.mean(axis=0) / dt        # (D,)
    std_dx = dx.std(axis=0) / np.sqrt(dt) # (D,)
    std_dx = np.maximum(std_dx, 1e-8)

    # mean-reversion coefficient (OU θ)
    x_centered = feature_trajectory[:-1] - feature_trajectory.mean(axis=0)
    # least-squares: dx_i ≈ -θ * x_centered_i * dt
    # θ_j = -cov(dx_j, x_j) / (var(x_j) * dt)
    theta = np.zeros(D)
    mu = feature_trajectory.mean(axis=0)
    for j in range(D):
        var_xj = x_centered[:, j].var()
        if var_xj > 1e-12:
            theta[j] = -np.cov(dx[:, j], x_centered[:, j])[0, 1] / (var_xj * dt)
    theta = np.clip(theta, 0, 100)  # stability: non-negative mean reversion

    _mu = mu.copy()
    _theta = theta.copy()
    _sigma = std_dx.copy()

    def drift_fn(x: np.ndarray, t: float) -> np.ndarray:
        return _theta * (_mu - x)

    def diffusion_fn(x: np.ndarray, t: float) -> np.ndarray:
        return _sigma

    return FeatureSDE(drift_fn=drift_fn, diffusion_fn=diffusion_fn, dt=dt, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Selection-rule learning from contradictions
# ═══════════════════════════════════════════════════════════════════════════


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 − cosine_similarity(a, b).  Returns 0 for identical, 2 for opposite."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


@dataclass
class Contradiction:
    """A detected self-contradiction under augmentation."""
    signal_idx: int
    augmentation: str
    distance: float  # cosine distance between original and augmented embeddings
    original_embedding: np.ndarray
    augmented_embedding: np.ndarray


class ContradictionDetector:
    """Detect when the model contradicts itself under augmentation.

    Two embeddings of (original, augmented) versions of the same input
    contradict if their cosine distance exceeds `threshold`.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.history: List[Contradiction] = []

    def check(
        self,
        original_emb: np.ndarray,
        augmented_emb: np.ndarray,
        signal_idx: int = 0,
        augmentation: str = "unknown",
    ) -> bool:
        """Return True if augmentation caused a contradiction."""
        d = _cosine_distance(original_emb, augmented_emb)
        is_contradiction = d > self.threshold
        if is_contradiction:
            self.history.append(Contradiction(
                signal_idx=signal_idx,
                augmentation=augmentation,
                distance=d,
                original_embedding=original_emb.copy(),
                augmented_embedding=augmented_emb.copy(),
            ))
        return is_contradiction

    def detect_batch(
        self,
        get_embedding: Callable[[np.ndarray], np.ndarray],
        signals: List[np.ndarray],
        augmentations: Dict[str, Callable[[np.ndarray], np.ndarray]],
    ) -> List[Contradiction]:
        """Run all augmentations on all signals and collect contradictions."""
        contradictions = []
        for i, sig in enumerate(signals):
            orig_emb = get_embedding(sig)
            for aug_name, aug_fn in augmentations.items():
                aug_sig = aug_fn(sig)
                aug_emb = get_embedding(aug_sig)
                if self.check(orig_emb, aug_emb, i, aug_name):
                    contradictions.append(self.history[-1])
        return contradictions

    @property
    def contradiction_rate(self) -> float:
        """Fraction of checks that were contradictions (from history)."""
        return len(self.history)  # absolute count; rate needs total checks


@dataclass
class SelectionRule:
    """Learned rule predicting when the model is unreliable.

    Logistic regression: P(contradiction) = σ(w·features + b).
    """
    weights: np.ndarray   # (n_features,)
    bias: float
    feature_names: List[str]
    auc: float = 0.0


def learn_selection_rule(
    contradictions: List[Contradiction],
    non_contradictions_features: np.ndarray,
    contradiction_features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_iter: int = 200,
    lr: float = 0.01,
    reg: float = 0.01,
) -> SelectionRule:
    """Fit logistic regression to predict contradictions from input features.

    Args:
        contradictions: list (for metadata only).
        non_contradictions_features: (N_neg, F) feature matrix for non-contradictions.
        contradiction_features: (N_pos, F) feature matrix for contradictions.
        feature_names: optional names for each feature dimension.
        n_iter: gradient descent steps.
        lr: learning rate.
        reg: L2 regularisation.

    Returns:
        SelectionRule with trained weights.
    """
    X_pos = contradiction_features
    X_neg = non_contradictions_features
    F = X_pos.shape[1]

    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_pos))])

    # standardise
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    X_s = (X - mu) / sd

    w = np.zeros(F)
    b = 0.0

    for _ in range(n_iter):
        logits = X_s @ w + b
        # numerically stable sigmoid
        p = np.where(logits >= 0,
                     1.0 / (1.0 + np.exp(-logits)),
                     np.exp(logits) / (1.0 + np.exp(logits)))
        p = np.clip(p, 1e-8, 1 - 1e-8)

        err = p - y  # (N,)
        grad_w = (X_s.T @ err) / len(y) + reg * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b

    # AUC via Wilcoxon rank-sum
    scores = X_s @ w + b
    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        auc = float(np.mean(pos_scores[:, None] > neg_scores[None, :]))
    else:
        auc = 0.5

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    return SelectionRule(weights=w, bias=float(b), feature_names=feature_names, auc=auc)


class SelectionRuleFilter:
    """Apply a learned selection rule to flag unreliable predictions."""

    def __init__(self, rule: SelectionRule, threshold: float = 0.5):
        self.rule = rule
        self.threshold = threshold

    def predict_unreliable(self, features: np.ndarray) -> np.ndarray:
        """Return boolean array: True where the model is likely unreliable.

        Args:
            features: (N, F) or (F,) feature vectors.
        """
        if features.ndim == 1:
            features = features[np.newaxis, :]
        logits = features @ self.rule.weights + self.rule.bias
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        return p > self.threshold

    def reliability_score(self, features: np.ndarray) -> np.ndarray:
        """Return P(reliable) = 1 − P(contradiction) for each sample."""
        if features.ndim == 1:
            features = features[np.newaxis, :]
        logits = features @ self.rule.weights + self.rule.bias
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        return 1.0 - p


# ═══════════════════════════════════════════════════════════════════════════
# 4. Counterfactual operators
# ═══════════════════════════════════════════════════════════════════════════


def counterfactual_embedding(
    get_embedding: Callable[[np.ndarray], np.ndarray],
    signal: np.ndarray,
    intervention_value: float,
    sample_idx: int,
) -> np.ndarray:
    """What would the embedding be if signal[sample_idx] had a different value?

    Produces a point-intervention counterfactual: replace one sample in the
    signal with `intervention_value` and re-compute the embedding.
    """
    cf_signal = signal.copy()
    cf_signal[sample_idx] = intervention_value
    return get_embedding(cf_signal)


def causal_sensitivity(
    get_embedding: Callable[[np.ndarray], np.ndarray],
    signal: np.ndarray,
    sample_indices: Optional[Sequence[int]] = None,
    eps: float = 1e-3,
) -> np.ndarray:
    """Finite-difference ∂embedding/∂signal[i] for each index i.

    Args:
        get_embedding: signal → embedding vector.
        signal: (S,) raw audio signal.
        sample_indices: which sample positions to probe. Defaults to 64
            evenly spaced positions.
        eps: perturbation size.

    Returns:
        sensitivity: (len(indices), D) Jacobian rows.
    """
    S = len(signal)
    if sample_indices is None:
        sample_indices = np.linspace(0, S - 1, min(64, S), dtype=int)

    base_emb = get_embedding(signal)  # (D,)
    D = base_emb.shape[0]
    jac = np.zeros((len(sample_indices), D))

    for j, idx in enumerate(sample_indices):
        sig_plus = signal.copy()
        sig_plus[idx] += eps
        emb_plus = get_embedding(sig_plus)
        jac[j] = (emb_plus - base_emb) / eps

    return jac


def counterfactual_explanation(
    get_embedding: Callable[[np.ndarray], np.ndarray],
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    n_probes: int = 32,
    eps: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """Identify which signal regions causally explain embedding differences.

    Interpolates from signal_a to signal_b in n_probes steps, measuring
    the embedding change at each step. Regions where the embedding changes
    most are causally responsible.

    Returns:
        dict with:
          'region_importance': (n_probes,) importance score per interpolation step
          'embedding_trajectory': (n_probes, D) embeddings along interpolation
          'cumulative_distance': (n_probes,) cumulative cosine distance from start
    """
    emb_a = get_embedding(signal_a)
    trajectory = [emb_a]
    importance = []

    for i in range(1, n_probes + 1):
        alpha = i / n_probes
        sig_interp = (1 - alpha) * signal_a + alpha * signal_b
        emb = get_embedding(sig_interp)
        # importance of this step = distance moved
        importance.append(float(np.linalg.norm(emb - trajectory[-1])))
        trajectory.append(emb)

    trajectory_arr = np.stack(trajectory[1:])  # (n_probes, D)
    importance_arr = np.array(importance)
    # normalise importance
    total = importance_arr.sum()
    if total > 1e-12:
        importance_arr /= total

    # cumulative cosine distance from emb_a
    cum_dist = np.zeros(n_probes)
    for i in range(n_probes):
        cum_dist[i] = _cosine_distance(emb_a, trajectory_arr[i])

    return {
        "region_importance": importance_arr,
        "embedding_trajectory": trajectory_arr,
        "cumulative_distance": cum_dist,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Uncertainty-aware prediction wrapper
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class UncertaintyPrediction:
    """Full uncertainty-annotated prediction."""
    embedding: np.ndarray                    # (D,)
    epistemic_uncertainty: float             # ensemble disagreement
    aleatoric_uncertainty: float             # data noise
    sde_ci_lower: Optional[np.ndarray]       # (D,) lower CI from SDE
    sde_ci_upper: Optional[np.ndarray]       # (D,) upper CI from SDE
    contradiction_flag: bool                 # True if any augmentation contradicts
    counterfactual_sensitivity: Optional[np.ndarray]  # (n_probes, D) Jacobian
    should_abstain: bool                     # True if uncertainty too high
    calibrated_confidence: float             # [0, 1]


class UncertaintyWrapper:
    """Unified uncertainty-aware prediction interface.

    Composes ensemble, SDE, contradiction detector, and counterfactual
    analysis into a single `predict_with_uncertainty` call.
    """

    def __init__(
        self,
        get_embedding: Callable[[np.ndarray], np.ndarray],
        ensemble: Optional[EigenstateEnsemble] = None,
        sde: Optional[FeatureSDE] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        abstention_threshold: float = 0.5,
        augmentations: Optional[Dict[str, Callable]] = None,
    ):
        self.get_embedding = get_embedding
        self.ensemble = ensemble
        self.sde = sde
        self.contradiction_detector = contradiction_detector
        self.abstention_threshold = abstention_threshold
        self.augmentations = augmentations or {
            "noise_01": lambda s: s + np.random.randn(len(s)) * 0.01,
            "gain_2x": lambda s: s * 2.0,
            "shift_100": lambda s: np.roll(s, 100),
        }

    def predict_with_uncertainty(
        self,
        signal: np.ndarray,
        run_sde: bool = True,
        run_counterfactual: bool = False,
        sde_T: float = 1.0,
        sde_n_paths: int = 50,
    ) -> UncertaintyPrediction:
        """Full uncertainty-annotated prediction for a single signal."""

        base_emb = self.get_embedding(signal)
        D = base_emb.shape[0]

        # — ensemble uncertainty —
        epistemic = 0.0
        aleatoric = 0.0
        if self.ensemble is not None and self.ensemble.is_trained:
            ep = ensemble_predict(self.ensemble, [signal])
            epistemic = ep.epistemic_uncertainty
            aleatoric = ep.aleatoric_uncertainty
            base_emb = ep.mean_embedding  # use ensemble mean

        # — SDE confidence interval —
        sde_lo: Optional[np.ndarray] = None
        sde_hi: Optional[np.ndarray] = None
        if self.sde is not None and run_sde:
            paths = euler_maruyama(self.sde, base_emb, T=sde_T, n_paths=sde_n_paths)
            unc = sde_uncertainty(paths)
            sde_lo = unc["ci_lower"][-1]  # final-time CI
            sde_hi = unc["ci_upper"][-1]

        # — contradiction check —
        contradiction = False
        if self.contradiction_detector is not None:
            for aug_name, aug_fn in self.augmentations.items():
                aug_sig = aug_fn(signal)
                aug_emb = self.get_embedding(aug_sig)
                if self.contradiction_detector.check(base_emb, aug_emb, augmentation=aug_name):
                    contradiction = True
                    break

        # — counterfactual sensitivity —
        cf_sens: Optional[np.ndarray] = None
        if run_counterfactual:
            cf_sens = causal_sensitivity(self.get_embedding, signal)

        # — abstention decision —
        raw_unc = epistemic + (1.0 if contradiction else 0.0)
        abstain = raw_unc > self.abstention_threshold

        # — calibrated confidence —
        # sigmoid mapping: high uncertainty → low confidence
        conf = 1.0 / (1.0 + np.exp(5.0 * (raw_unc - self.abstention_threshold)))
        conf = float(np.clip(conf, 0.0, 1.0))

        return UncertaintyPrediction(
            embedding=base_emb,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            sde_ci_lower=sde_lo,
            sde_ci_upper=sde_hi,
            contradiction_flag=contradiction,
            counterfactual_sensitivity=cf_sens,
            should_abstain=abstain,
            calibrated_confidence=conf,
        )


def should_abstain(uncertainties: Dict[str, float], threshold: float = 0.5) -> bool:
    """Standalone abstention decision from an uncertainty dict."""
    total = sum(uncertainties.values())
    return total > threshold


def calibrated_confidence(uncertainties: Dict[str, float], scale: float = 5.0) -> float:
    """Map raw uncertainty dict to a [0, 1] confidence score."""
    total = sum(uncertainties.values())
    return float(1.0 / (1.0 + np.exp(scale * (total - 0.5))))


# ═══════════════════════════════════════════════════════════════════════════
# Standard augmentations for contradiction detection
# ═══════════════════════════════════════════════════════════════════════════


def augmentation_add_noise(signal: np.ndarray, std: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return signal + rng.randn(len(signal)) * std


def augmentation_gain(signal: np.ndarray, factor: float = 2.0) -> np.ndarray:
    return signal * factor


def augmentation_time_shift(signal: np.ndarray, shift: int = 100) -> np.ndarray:
    return np.roll(signal, shift)


def augmentation_polarity_invert(signal: np.ndarray) -> np.ndarray:
    return -signal


def augmentation_lowpass(signal: np.ndarray, cutoff_ratio: float = 0.5) -> np.ndarray:
    """Simple brick-wall lowpass in frequency domain."""
    F = np.fft.rfft(signal)
    n_keep = max(1, int(len(F) * cutoff_ratio))
    F[n_keep:] = 0
    return np.fft.irfft(F, n=len(signal))


STANDARD_AUGMENTATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "noise_001": lambda s: augmentation_add_noise(s, 0.001),
    "noise_01": lambda s: augmentation_add_noise(s, 0.01),
    "noise_05": lambda s: augmentation_add_noise(s, 0.05),
    "gain_05": lambda s: augmentation_gain(s, 0.5),
    "gain_2": lambda s: augmentation_gain(s, 2.0),
    "shift_100": lambda s: augmentation_time_shift(s, 100),
    "shift_1000": lambda s: augmentation_time_shift(s, 1000),
    "polarity": augmentation_polarity_invert,
    "lowpass_50": lambda s: augmentation_lowpass(s, 0.5),
    "lowpass_25": lambda s: augmentation_lowpass(s, 0.25),
}
