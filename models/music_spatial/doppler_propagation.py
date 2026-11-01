"""Doppler shift, air absorption, and sound propagation modeling.

Models frequency-dependent atmospheric attenuation (ISO 9613-1),
Doppler pitch shifting from source velocity, geometric spreading,
and ground reflection interference for outdoor propagation.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

SPEED_OF_SOUND = 343.0  # m/s at 20°C
REF_PRESSURE = 20e-6    # 20 µPa reference

# --- Air absorption (ISO 9613-1 simplified) ---

@dataclass
class AtmosphericConditions:
    temperature_c: float = 20.0
    humidity_pct: float = 50.0
    pressure_kpa: float = 101.325

    @property
    def temperature_k(self) -> float:
        return self.temperature_c + 273.15


def _relaxation_frequencies(atm: AtmosphericConditions) -> Tuple[float, float]:
    """Oxygen and nitrogen relaxation frequencies (Hz) per ISO 9613-1."""
    h = atm.humidity_pct
    T = atm.temperature_k
    T_ref = 293.15
    p_ref = 101.325
    p_ratio = atm.pressure_kpa / p_ref

    # Molar concentration of water vapour
    C = -6.8346 * (273.16 / T) ** 1.261 + 4.6151
    h_mol = h * 10.0 ** C * p_ratio

    # Oxygen relaxation
    fr_O = p_ratio * (24.0 + 4.04e4 * h_mol * (0.02 + h_mol) / (0.391 + h_mol))

    # Nitrogen relaxation
    fr_N = p_ratio * (T / T_ref) ** (-0.5) * (
        9.0 + 280.0 * h_mol * np.exp(-4.170 * ((T / T_ref) ** (-1.0 / 3.0) - 1.0))
    )
    return float(fr_O), float(fr_N)


def air_absorption_coefficient(
    freq: float | NDArray,
    atm: AtmosphericConditions | None = None,
) -> NDArray:
    """Attenuation coefficient α (dB/m) per ISO 9613-1.

    Parameters
    ----------
    freq : scalar or array of frequencies (Hz)
    atm : atmospheric conditions; defaults to 20°C / 50% RH / 101.325 kPa

    Returns
    -------
    alpha : same shape as *freq*, in dB per metre
    """
    if atm is None:
        atm = AtmosphericConditions()
    f = np.asarray(freq, dtype=np.float64)
    T = atm.temperature_k
    T_ref = 293.15
    p_ratio = atm.pressure_kpa / 101.325
    fr_O, fr_N = _relaxation_frequencies(atm)

    f_sq = f ** 2
    # Classical + rotational absorption
    alpha = 1.84e-11 * (1.0 / p_ratio) * (T / T_ref) ** 0.5 * f_sq

    # Vibrational: oxygen
    alpha += 0.01275 * np.exp(-2239.1 / T) * (
        fr_O / (fr_O ** 2 + f_sq)
    ) * f_sq

    # Vibrational: nitrogen
    alpha += 0.10680 * np.exp(-3352.0 / T) * (
        fr_N / (fr_N ** 2 + f_sq)
    ) * f_sq

    return alpha * 8.686  # Np/m → dB/m


def apply_air_absorption(
    spectrum: NDArray,
    freqs: NDArray,
    distance_m: float,
    atm: AtmosphericConditions | None = None,
) -> NDArray:
    """Attenuate a magnitude spectrum by air absorption over *distance_m*."""
    alpha = air_absorption_coefficient(freqs, atm)
    atten_db = alpha * distance_m
    return spectrum * 10.0 ** (-atten_db / 20.0)


# --- Doppler effect ---

def doppler_factor(
    source_velocity: float,
    receiver_velocity: float = 0.0,
    angle_rad: float = 0.0,
    c: float = SPEED_OF_SOUND,
) -> float:
    """Frequency scaling factor f_received / f_emitted.

    Uses radial velocity components along the source→receiver axis.
    Positive velocity = moving toward the other party.
    *angle_rad*: angle between velocity vector and source→receiver line.
    """
    v_s = source_velocity * np.cos(angle_rad)
    v_r = receiver_velocity * np.cos(angle_rad)
    return (c + v_r) / (c + v_s)


def doppler_shift_signal(
    signal: NDArray,
    sr: int,
    source_positions: NDArray,
    receiver_pos: NDArray,
    c: float = SPEED_OF_SOUND,
) -> NDArray:
    """Resample *signal* with per-sample Doppler from a moving source.

    Parameters
    ----------
    signal : (N,) mono source signal
    sr : sample rate
    source_positions : (N, 3) xyz positions of source at each sample
    receiver_pos : (3,) static receiver position
    c : speed of sound

    Returns
    -------
    doppler_signal : (N,) resampled with variable delay
    """
    sig = np.asarray(signal, dtype=np.float64)
    src = np.asarray(source_positions, dtype=np.float64)
    rec = np.asarray(receiver_pos, dtype=np.float64)

    distances = np.linalg.norm(src - rec, axis=1)
    delay_samples = distances / c * sr

    # Variable-delay interpolation (linear)
    read_indices = np.arange(len(sig)) - delay_samples
    read_indices = np.clip(read_indices, 0, len(sig) - 1)

    idx_floor = np.floor(read_indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(sig) - 1)
    frac = read_indices - idx_floor

    out = sig[idx_floor] * (1.0 - frac) + sig[idx_ceil] * frac

    # Geometric spreading (inverse-square)
    ref_dist = max(distances[0], 0.1)
    gain = ref_dist / np.maximum(distances, 0.1)
    return out * gain


# --- Geometric spreading ---

def spherical_spreading_db(distance_m: float, ref_distance: float = 1.0) -> float:
    """SPL loss (dB) from inverse-square spherical spreading."""
    if distance_m <= 0 or ref_distance <= 0:
        return 0.0
    return -20.0 * np.log10(distance_m / ref_distance)


def cylindrical_spreading_db(distance_m: float, ref_distance: float = 1.0) -> float:
    """SPL loss (dB) for cylindrical spreading (e.g. traffic noise corridor)."""
    if distance_m <= 0 or ref_distance <= 0:
        return 0.0
    return -10.0 * np.log10(distance_m / ref_distance)


# --- Ground reflection ---

@dataclass
class GroundReflection:
    """Coherent ground reflection model (flat, homogeneous ground).

    The interference between the direct and ground-reflected paths
    produces a comb-filter effect whose notch frequencies depend on
    path-length difference and ground impedance.
    """
    source_height_m: float = 1.5
    receiver_height_m: float = 1.5
    ground_flow_resistivity: float = 200e3  # Pa·s/m² (grass ≈ 200–300 kPa s/m²)

    def path_length_difference(self, horizontal_distance: float) -> float:
        d = horizontal_distance
        hs, hr = self.source_height_m, self.receiver_height_m
        direct = np.sqrt(d ** 2 + (hs - hr) ** 2)
        reflected = np.sqrt(d ** 2 + (hs + hr) ** 2)
        return float(reflected - direct)

    def ground_impedance(self, freq: float | NDArray) -> NDArray:
        """Delany-Bazley ground impedance model (normalised to ρc)."""
        f = np.asarray(freq, dtype=np.float64)
        sigma = self.ground_flow_resistivity
        X = f / sigma
        Z_real = 1.0 + 9.08 * X ** (-0.75)
        Z_imag = -11.9 * X ** (-0.73)
        return Z_real + 1j * Z_imag

    def reflection_coefficient(
        self, freq: float | NDArray, horizontal_distance: float
    ) -> NDArray:
        """Plane-wave reflection coefficient R_p for the ground path."""
        f = np.asarray(freq, dtype=np.float64)
        d = horizontal_distance
        Z_g = self.ground_impedance(f)
        theta = np.arctan2(
            self.source_height_m + self.receiver_height_m, d
        )
        cos_theta = np.cos(theta)
        R_p = (Z_g * cos_theta - 1.0) / (Z_g * cos_theta + 1.0)
        return R_p

    def excess_attenuation(
        self, freq: float | NDArray, horizontal_distance: float
    ) -> NDArray:
        """Excess attenuation (dB) from ground reflection interference."""
        f = np.asarray(freq, dtype=np.float64)
        delta_r = self.path_length_difference(horizontal_distance)
        R_p = self.reflection_coefficient(f, horizontal_distance)
        phase = 2.0 * np.pi * f * delta_r / SPEED_OF_SOUND
        # Total = direct + reflected with phase shift
        total = 1.0 + R_p * np.exp(1j * phase)
        return 20.0 * np.log10(np.maximum(np.abs(total), 1e-12))


# --- Combined propagation loss ---

def total_propagation_loss(
    freq: float | NDArray,
    distance_m: float,
    atm: AtmosphericConditions | None = None,
    ground: GroundReflection | None = None,
) -> NDArray:
    """Total frequency-dependent propagation loss (dB).

    Combines spherical spreading, air absorption, and optionally ground effect.
    """
    f = np.asarray(freq, dtype=np.float64)
    loss = np.full_like(f, spherical_spreading_db(distance_m))
    loss -= air_absorption_coefficient(f, atm) * distance_m
    if ground is not None:
        loss += ground.excess_attenuation(f, distance_m)
    return loss
