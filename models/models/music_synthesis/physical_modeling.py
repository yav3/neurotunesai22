"""Physical modeling synthesis: Karplus-Strong, digital waveguide, bowed string."""
from __future__ import annotations

import numpy as np


def karplus_strong(frequency: float, sr: float, duration: float,
                   decay: float = 0.996, seed: int = 42) -> np.ndarray:
    n_samples = int(sr * duration)
    delay_len = max(2, int(sr / frequency))
    rng = np.random.default_rng(seed)
    buffer = rng.uniform(-1, 1, delay_len)
    output = np.zeros(n_samples)
    for i in range(n_samples):
        idx = i % delay_len
        output[i] = buffer[idx]
        next_idx = (idx + 1) % delay_len
        buffer[idx] = decay * 0.5 * (buffer[idx] + buffer[next_idx])
    return output


def damped_string(frequency: float, sr: float, duration: float,
                  brightness: float = 0.5, decay: float = 0.998,
                  seed: int = 42) -> np.ndarray:
    n_samples = int(sr * duration)
    delay_len = max(2, int(sr / frequency))
    rng = np.random.default_rng(seed)
    buffer = rng.uniform(-1, 1, delay_len)
    output = np.zeros(n_samples)
    a = 1.0 - brightness
    prev_filtered = 0.0
    for i in range(n_samples):
        idx = i % delay_len
        output[i] = buffer[idx]
        next_idx = (idx + 1) % delay_len
        avg = 0.5 * (buffer[idx] + buffer[next_idx])
        filtered = (1.0 - a) * avg + a * prev_filtered
        prev_filtered = filtered
        buffer[idx] = decay * filtered
    return output


def bowed_string(frequency: float, sr: float, duration: float,
                 bow_velocity: float = 0.2, bow_force: float = 0.5,
                 seed: int = 42) -> np.ndarray:
    n_samples = int(sr * duration)
    delay_len = max(2, int(sr / (2.0 * frequency)))
    rng = np.random.default_rng(seed)
    nut_delay = np.zeros(delay_len)
    bridge_delay = np.zeros(delay_len)
    nut_delay[:] = rng.uniform(-0.001, 0.001, delay_len)
    output = np.zeros(n_samples)
    reflection_coeff = -0.95
    for i in range(n_samples):
        idx = i % delay_len
        v_nut = nut_delay[idx]
        v_bridge = bridge_delay[idx]
        v_string = v_nut + v_bridge
        delta_v = bow_velocity - v_string
        friction = bow_force * delta_v / (1.0 + abs(delta_v) * 10.0)
        nut_delay[idx] = friction + v_bridge
        bridge_delay[idx] = reflection_coeff * (friction + v_nut)
        output[i] = bridge_delay[idx]
    peak = np.max(np.abs(output))
    if peak > 1e-10:
        output /= peak
    return output


def flute_waveguide(frequency: float, sr: float, duration: float,
                    jet_velocity: float = 0.3, noise_gain: float = 0.05,
                    seed: int = 42) -> np.ndarray:
    n_samples = int(sr * duration)
    bore_len = max(2, int(sr / frequency))
    rng = np.random.default_rng(seed)
    bore = np.zeros(bore_len)
    output = np.zeros(n_samples)
    embouchure_delay = 0.0
    open_end_reflection = -0.7
    tone_hole_loss = 0.98
    prev = 0.0
    for i in range(n_samples):
        idx = i % bore_len
        noise = noise_gain * rng.normal()
        jet = jet_velocity + noise
        excitation = np.tanh(jet - embouchure_delay)
        bore[idx] = tone_hole_loss * (excitation + open_end_reflection * bore[idx])
        lp = 0.7 * bore[idx] + 0.3 * prev
        prev = lp
        embouchure_delay = lp
        output[i] = bore[idx]
    peak = np.max(np.abs(output))
    if peak > 1e-10:
        output /= peak
    return output


def drum_membrane(frequency: float, sr: float, duration: float,
                  inharmonicity: float = 0.5, decay: float = 0.995,
                  seed: int = 42) -> np.ndarray:
    n_samples = int(sr * duration)
    n_modes = 4
    mode_ratios = [1.0, 1.593, 2.136, 2.653]
    mode_ratios = [r * (1.0 + inharmonicity * (r - 1.0) * 0.1) for r in mode_ratios]
    rng = np.random.default_rng(seed)
    output = np.zeros(n_samples)
    for j, ratio in enumerate(mode_ratios):
        f_mode = frequency * ratio
        delay_len = max(2, int(sr / f_mode))
        buf = rng.uniform(-1, 1, delay_len)
        amplitude = 1.0 / (j + 1)
        mode_decay = decay ** (1.0 + j * 0.3)
        for i in range(n_samples):
            idx = i % delay_len
            next_idx = (idx + 1) % delay_len
            output[i] += amplitude * buf[idx]
            buf[idx] = mode_decay * 0.5 * (buf[idx] + buf[next_idx])
    peak = np.max(np.abs(output))
    if peak > 1e-10:
        output /= peak
    return output
