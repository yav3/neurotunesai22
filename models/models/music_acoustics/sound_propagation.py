"""Sound propagation models: attenuation, Doppler, diffraction, reflection."""
import numpy as np


def inverse_square_attenuation(distance: float,
                               reference_distance: float = 1.0) -> float:
    if distance <= 0 or reference_distance <= 0:
        return 0.0
    return 20.0 * np.log10(reference_distance / distance)


def air_absorption_coefficient(frequency: float,
                               temperature: float = 20.0,
                               humidity: float = 50.0) -> float:
    f_khz = frequency / 1000.0
    h = max(humidity, 1.0)
    T = temperature + 273.15
    T_ref = 293.15
    f_rO = (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    f_rN = (T_ref / T) ** 0.5 * (9.0 + 280.0 * h *
            np.exp(-4.170 * ((T_ref / T) ** (1.0 / 3.0) - 1.0)))
    alpha = 8.686 * f_khz ** 2 * (
        1.84e-11 * (T / T_ref) ** 0.5 +
        (T / T_ref) ** (-5.0 / 2.0) * (
            0.01275 * np.exp(-2239.1 / T) / (f_rO + f_khz ** 2 / f_rO) +
            0.1068 * np.exp(-3352.0 / T) / (f_rN + f_khz ** 2 / f_rN)
        )
    )
    return float(max(alpha, 0.0))


def doppler_shift(source_freq: float,
                  source_velocity: float,
                  observer_velocity: float = 0.0,
                  speed_of_sound: float = 343.0) -> float:
    denom = speed_of_sound - source_velocity
    if abs(denom) < 1e-6:
        return source_freq
    return source_freq * (speed_of_sound + observer_velocity) / denom


def interference_pattern(freq: float, d1: float, d2: float,
                         speed_of_sound: float = 343.0) -> float:
    wavelength = speed_of_sound / freq if freq > 0 else 1e10
    delta = abs(d2 - d1)
    phase = 2.0 * np.pi * delta / wavelength
    return float(abs(np.cos(phase / 2.0)))


def diffraction_coefficient(freq: float, barrier_height: float,
                            source_dist: float,
                            receiver_dist: float) -> float:
    if freq <= 0 or barrier_height <= 0:
        return 1.0
    wavelength = 343.0 / freq
    d_s = max(source_dist, 0.01)
    d_r = max(receiver_dist, 0.01)
    path_diff = np.sqrt(d_s ** 2 + barrier_height ** 2) + \
                np.sqrt(d_r ** 2 + barrier_height ** 2) - (d_s + d_r)
    N = 2.0 * path_diff / wavelength
    if N < -0.2:
        return 1.0
    att_db = 10.0 * np.log10(3.0 + 20.0 * N)
    att_db = max(att_db, 0.0)
    return float(np.clip(10.0 ** (-att_db / 20.0), 0.0, 1.0))


def ground_reflection_coefficient(freq: float,
                                  flow_resistivity: float = 200000.0) -> complex:
    if freq <= 0:
        return complex(1.0, 0.0)
    f_sigma = freq / flow_resistivity
    Z_real = 1.0 + 9.08 * f_sigma ** (-0.75)
    Z_imag = -11.9 * f_sigma ** (-0.73)
    Z = complex(Z_real, Z_imag)
    R = (Z - 1.0) / (Z + 1.0)
    return R
