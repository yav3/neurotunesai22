"""Rhythmic entrainment for music therapy: Kuramoto coupling, PLV, ISO principle."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EntrainmentState:
    phase_coherence: float
    frequency_ratio: float
    coupling_strength: float
    stability: float


def kuramoto_phase_coupling(
    theta_stim: np.ndarray,
    theta_response: np.ndarray,
    coupling_k: float = 1.0,
    dt: float = 0.01,
) -> np.ndarray:
    n = min(len(theta_stim), len(theta_response))
    if n == 0:
        return np.array([])
    out = np.zeros(n)
    out[0] = theta_response[0]
    for i in range(1, n):
        d_theta = coupling_k * np.sin(theta_stim[i - 1] - out[i - 1])
        omega = (theta_stim[i] - theta_stim[i - 1]) / dt if i > 0 else 0.0
        out[i] = out[i - 1] + (omega + d_theta) * dt
    return out


def phase_locking_value(phase1: np.ndarray, phase2: np.ndarray) -> float:
    n = min(len(phase1), len(phase2))
    if n == 0:
        return 0.0
    diff = phase1[:n] - phase2[:n]
    return float(np.abs(np.mean(np.exp(1j * diff))))


def adaptive_tempo_prescription(
    resting_hr: float,
    target_hr: float,
    steps: int = 10,
) -> np.ndarray:
    if steps < 1:
        return np.array([resting_hr])
    resting_bpm = resting_hr
    target_bpm = target_hr
    tempos = np.linspace(resting_bpm, target_bpm, steps)
    return tempos


def rhythmic_complexity_for_therapy(ioi_sequence: np.ndarray) -> dict:
    if len(ioi_sequence) < 2:
        return {"syncopation_index": 0.0, "regularity": 1.0, "predictability": 1.0}
    mean_ioi = float(np.mean(ioi_sequence))
    std_ioi = float(np.std(ioi_sequence))
    cv = std_ioi / max(mean_ioi, 1e-12)
    regularity = max(1.0 - cv, 0.0)

    diffs = np.diff(ioi_sequence)
    sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0) if len(diffs) > 1 else 0
    syncopation = float(sign_changes / max(len(diffs) - 1, 1))

    autocorr = np.correlate(ioi_sequence - mean_ioi, ioi_sequence - mean_ioi, mode="full")
    mid = len(autocorr) // 2
    norm = autocorr[mid] if autocorr[mid] > 0 else 1.0
    predictability = float(autocorr[mid + 1] / norm) if len(autocorr) > mid + 1 else 0.0
    predictability = max(min(predictability, 1.0), 0.0)

    return {
        "syncopation_index": syncopation,
        "regularity": regularity,
        "predictability": predictability,
    }


def entrainment_trajectory(
    stim_freq: float,
    response_freq_init: float,
    coupling: float = 0.5,
    n_steps: int = 100,
) -> EntrainmentState:
    dt = 0.01
    omega_s = 2 * np.pi * stim_freq
    omega_r = 2 * np.pi * response_freq_init
    theta_s = 0.0
    theta_r = 0.0
    phase_diffs = []
    for _ in range(n_steps):
        diff = theta_s - theta_r
        d_theta_r = omega_r + coupling * np.sin(diff)
        theta_s += omega_s * dt
        theta_r += d_theta_r * dt
        phase_diffs.append(diff % (2 * np.pi))

    phase_diffs_arr = np.array(phase_diffs)
    coherence = float(np.abs(np.mean(np.exp(1j * phase_diffs_arr[-max(n_steps // 4, 1):]))))
    final_ratio = stim_freq / max(response_freq_init, 1e-12)

    diffs_late = phase_diffs_arr[-max(n_steps // 4, 1):]
    stability = 1.0 - min(float(np.std(diffs_late) / np.pi), 1.0)

    return EntrainmentState(
        phase_coherence=coherence,
        frequency_ratio=final_ratio,
        coupling_strength=coupling,
        stability=stability,
    )


def iso_principle_trajectory(
    patient_tempo: float,
    target_tempo: float,
    session_minutes: float = 20.0,
    update_interval_sec: float = 30.0,
) -> np.ndarray:
    n_updates = max(int(session_minutes * 60 / update_interval_sec), 2)
    alpha = 3.0 / n_updates
    tempos = np.zeros(n_updates)
    tempos[0] = patient_tempo
    for i in range(1, n_updates):
        tempos[i] = tempos[i - 1] + alpha * (target_tempo - tempos[i - 1])
    return tempos
