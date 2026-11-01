"""Reverb parameter estimation and impulse response synthesis."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ReverbParameters:
    rt60: float
    pre_delay_ms: float = 10.0
    early_reflection_density: float = 0.5
    late_diffusion: float = 0.7
    damping: float = 0.3
    room_size_m3: float = 100.0


def estimate_rt60(impulse_response: np.ndarray, sr: int) -> float:
    ir2 = impulse_response ** 2
    if np.sum(ir2) < 1e-30:
        return 0.0
    schroeder = np.cumsum(ir2[::-1])[::-1]
    schroeder = schroeder / (schroeder[0] + 1e-30)
    db = 10.0 * np.log10(schroeder + 1e-30)
    idx_5 = np.searchsorted(-db, 5.0)
    idx_25 = np.searchsorted(-db, 25.0)
    if idx_25 <= idx_5 or idx_25 >= len(db):
        idx_25 = min(len(db) - 1, idx_5 + 10)
    if idx_25 <= idx_5:
        return 0.0
    t5 = idx_5 / sr
    t25 = idx_25 / sr
    slope = (db[idx_25] - db[idx_5]) / (t25 - t5 + 1e-30)
    if slope >= 0:
        return 0.0
    return -60.0 / slope


def generate_impulse_response(params: ReverbParameters, sr: int,
                              duration: float = 1.0) -> np.ndarray:
    n = int(sr * duration)
    ir = np.zeros(n)
    pre_samples = int(params.pre_delay_ms * sr / 1000.0)
    if pre_samples < n:
        ir[pre_samples] = 1.0
    if params.rt60 > 0:
        decay_rate = 6.91 / params.rt60
        t = np.arange(n) / sr
        env = np.exp(-decay_rate * t) * params.late_diffusion
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(n)
        n_early = min(int(0.08 * sr), n)
        n_reflections = max(1, int(params.early_reflection_density * 20))
        for i in range(n_reflections):
            idx = pre_samples + int((i + 1) * n_early / (n_reflections + 1))
            if idx < n:
                ir[idx] += (1.0 - params.damping) * (0.8 ** i)
        ir += noise * env * 0.1
    ir /= (np.max(np.abs(ir)) + 1e-30)
    return ir


def apply_reverb(signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
    if len(signal) == 0 or len(ir) == 0:
        return signal.copy()
    return np.convolve(signal, ir, mode="full")


def clarity_c80(ir: np.ndarray, sr: int) -> float:
    ir2 = ir ** 2
    boundary = int(0.08 * sr)
    early = np.sum(ir2[:boundary]) if boundary <= len(ir2) else np.sum(ir2)
    late = np.sum(ir2[boundary:]) if boundary < len(ir2) else 1e-30
    if late < 1e-30:
        return 30.0
    return 10.0 * np.log10(early / late)


def definition_d50(ir: np.ndarray, sr: int) -> float:
    ir2 = ir ** 2
    boundary = int(0.05 * sr)
    early = np.sum(ir2[:boundary]) if boundary <= len(ir2) else np.sum(ir2)
    total = np.sum(ir2)
    if total < 1e-30:
        return 0.0
    return float(np.clip(early / total, 0.0, 1.0))


def early_decay_time(ir: np.ndarray, sr: int) -> float:
    ir2 = ir ** 2
    if np.sum(ir2) < 1e-30:
        return 0.0
    schroeder = np.cumsum(ir2[::-1])[::-1]
    schroeder = schroeder / (schroeder[0] + 1e-30)
    db = 10.0 * np.log10(schroeder + 1e-30)
    idx_0 = np.searchsorted(-db, 0.0)
    idx_10 = np.searchsorted(-db, 10.0)
    if idx_10 <= idx_0 or idx_10 >= len(db):
        return 0.0
    t0 = idx_0 / sr
    t10 = idx_10 / sr
    slope = (db[idx_10] - db[idx_0]) / (t10 - t0 + 1e-30)
    if slope >= 0:
        return 0.0
    return -60.0 / slope
