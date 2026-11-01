"""Adaptive filtering for music signals.

LMS, NLMS, RLS adaptive filters for noise cancellation,
echo removal, and system identification in audio contexts.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class FilterType(enum.Enum):
    LMS = "lms"
    NLMS = "nlms"
    RLS = "rls"
    KALMAN = "kalman"


@dataclass
class AdaptiveFilterState:
    weights: np.ndarray
    step_size: float = 0.01
    error_history: list[float] = field(default_factory=list)


def lms_update(
    state: AdaptiveFilterState,
    x: np.ndarray,
    desired: float,
) -> float:
    y = float(np.dot(state.weights, x))
    e = desired - y
    state.weights = state.weights + state.step_size * e * x
    state.error_history.append(e)
    return y


def nlms_update(
    state: AdaptiveFilterState,
    x: np.ndarray,
    desired: float,
    eps: float = 1e-8,
) -> float:
    y = float(np.dot(state.weights, x))
    e = desired - y
    norm_sq = float(np.dot(x, x)) + eps
    state.weights = state.weights + (state.step_size / norm_sq) * e * x
    state.error_history.append(e)
    return y


def rls_update(
    state: AdaptiveFilterState,
    x: np.ndarray,
    desired: float,
    P: np.ndarray,
    lam: float = 0.99,
) -> tuple[float, np.ndarray]:
    x_col = x.reshape(-1, 1)
    k = P @ x_col / (lam + float(x_col.T @ P @ x_col))
    y = float(state.weights @ x)
    e = desired - y
    state.weights = state.weights + (k.ravel() * e)
    P = (P - k @ x_col.T @ P) / lam
    state.error_history.append(e)
    return y, P


def kalman_filter_step(
    x_est: np.ndarray,
    P_est: np.ndarray,
    z: float,
    H: np.ndarray,
    Q: np.ndarray,
    R: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_pred = x_est.copy()
    P_pred = P_est + Q
    S = float(H @ P_pred @ H.T) + R
    K = (P_pred @ H.T) / S
    innovation = z - float(H @ x_pred)
    x_new = x_pred + K.ravel() * innovation
    P_new = (np.eye(len(x_est)) - np.outer(K.ravel(), H.ravel())) @ P_pred
    return x_new, P_new


def noise_cancellation(
    primary: np.ndarray,
    reference: np.ndarray,
    filter_order: int = 32,
    step_size: float = 0.01,
    method: FilterType = FilterType.NLMS,
) -> np.ndarray:
    n = len(primary)
    state = AdaptiveFilterState(
        weights=np.zeros(filter_order),
        step_size=step_size,
    )
    output = np.zeros(n)
    for i in range(filter_order, n):
        x = reference[i - filter_order:i][::-1]
        if method == FilterType.LMS:
            y = lms_update(state, x, primary[i])
        else:
            y = nlms_update(state, x, primary[i])
        output[i] = primary[i] - y
    return output


def echo_cancellation(
    signal: np.ndarray,
    echo_ref: np.ndarray,
    filter_order: int = 64,
    step_size: float = 0.005,
) -> np.ndarray:
    return noise_cancellation(signal, echo_ref, filter_order, step_size, FilterType.NLMS)


def system_identification(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    filter_order: int = 32,
    step_size: float = 0.01,
) -> np.ndarray:
    state = AdaptiveFilterState(
        weights=np.zeros(filter_order),
        step_size=step_size,
    )
    n = len(input_signal)
    for i in range(filter_order, n):
        x = input_signal[i - filter_order:i][::-1]
        nlms_update(state, x, output_signal[i])
    return state.weights.copy()


def compute_misalignment(
    estimated: np.ndarray,
    true_weights: np.ndarray,
) -> float:
    diff = estimated[:len(true_weights)] - true_weights[:len(estimated)]
    norm_true = float(np.linalg.norm(true_weights))
    if norm_true < 1e-12:
        return float(np.linalg.norm(diff))
    return float(np.linalg.norm(diff) / norm_true)
