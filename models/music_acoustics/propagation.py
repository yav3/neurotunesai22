"""Sound propagation models: atmospheric absorption, diffraction, ground reflection."""
from __future__ import annotations

import numpy as np


class SoundPropagation:
    """Outdoor and indoor sound propagation physics."""

    @staticmethod
    def speed_of_sound(temperature: float = 20.0, humidity: float = 50.0) -> float:
        c = 331.3 * np.sqrt(1 + temperature / 273.15)
        h_corr = 0.0016 * humidity * 0.01
        return float(c + h_corr)

    @staticmethod
    def wavelength(freq: float, temperature: float = 20.0) -> float:
        if freq <= 0:
            return 0.0
        c = SoundPropagation.speed_of_sound(temperature)
        return c / freq

    @staticmethod
    def inverse_square_law(
        distance: float, reference_spl: float = 94.0, ref_distance: float = 1.0
    ) -> float:
        if distance <= 0:
            return reference_spl
        return reference_spl - 20 * np.log10(distance / ref_distance)

    @staticmethod
    def atmospheric_absorption(
        freq: float,
        temperature: float = 20.0,
        humidity: float = 50.0,
        distance: float = 1.0,
    ) -> float:
        if freq <= 0 or distance <= 0:
            return 0.0
        T_K = temperature + 273.15
        T_rel = T_K / 293.15
        h = humidity * np.exp(4.6151 - 6.8346 * (273.16 / T_K) ** 1.261) / 100
        h = max(h, 1e-10)
        fr_O = 24 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
        fr_N = T_rel ** (-0.5) * (9 + 280 * h * np.exp(-4.170 * (T_rel ** (-1.0 / 3.0) - 1)))
        f2 = freq ** 2
        alpha = (
            8.686 * f2 * (
                1.84e-11 * T_rel ** 0.5
                + T_rel ** (-2.5) * (
                    0.01275 * np.exp(-2239.1 / T_K) / (fr_O + f2 / fr_O)
                    + 0.1068 * np.exp(-3352.0 / T_K) / (fr_N + f2 / fr_N)
                )
            )
        )
        return float(alpha * distance)

    @staticmethod
    def diffraction_coefficient(
        freq: float,
        barrier_height: float,
        source_dist: float,
        receiver_dist: float,
        temperature: float = 20.0,
    ) -> float:
        if freq <= 0:
            return 0.0
        c = SoundPropagation.speed_of_sound(temperature)
        lam = c / freq
        d_s = np.sqrt(source_dist ** 2 + barrier_height ** 2)
        d_r = np.sqrt(receiver_dist ** 2 + barrier_height ** 2)
        delta = d_s + d_r - (source_dist + receiver_dist)
        N = 2 * delta / lam
        if N <= -0.2:
            return 0.0
        att = 5 + 20 * np.log10(np.sqrt(2 * np.pi * max(N, 0.001)) / (np.tanh(np.sqrt(2 * np.pi * max(N, 0.001)))))
        return float(max(att, 0.0))

    @staticmethod
    def ground_reflection(
        freq: float,
        source_height: float,
        receiver_height: float,
        distance: float,
        flow_resistivity: float = 200000,
    ) -> float:
        if freq <= 0 or distance <= 0:
            return 0.0
        c = 343.0
        lam = c / freq
        d_direct = np.sqrt(distance ** 2 + (source_height - receiver_height) ** 2)
        d_reflect = np.sqrt(distance ** 2 + (source_height + receiver_height) ** 2)
        delta_d = d_reflect - d_direct
        phase = 2 * np.pi * delta_d / lam
        sigma_ratio = freq / flow_resistivity
        R = 1 - 2 * sigma_ratio ** 0.5
        R = max(min(R, 1.0), -1.0)
        interference = 20 * np.log10(max(abs(1 + R * np.exp(1j * phase)), 1e-10))
        return float(interference)

    @staticmethod
    def doppler_shift(
        f0: float,
        v_source: float,
        v_receiver: float = 0.0,
        c: float = 343.0,
    ) -> float:
        if f0 <= 0:
            return 0.0
        denom = c - v_source
        if abs(denom) < 1e-6:
            return f0 * 1e6
        return float(f0 * (c + v_receiver) / denom)


class RoomModeAnalyzer:
    """Room acoustic mode analysis."""

    @staticmethod
    def axial_modes(
        length: float,
        width: float,
        height: float,
        max_freq: float = 500.0,
        c: float = 343.0,
    ) -> list[tuple[float, int, int, int, str]]:
        modes = []
        max_n = int(2 * max_freq * max(length, width, height) / c) + 1
        for nx in range(max_n + 1):
            for ny in range(max_n + 1):
                for nz in range(max_n + 1):
                    if nx == ny == nz == 0:
                        continue
                    f = (c / 2) * np.sqrt(
                        (nx / length) ** 2 + (ny / width) ** 2 + (nz / height) ** 2
                    )
                    if f > max_freq:
                        continue
                    count = (nx > 0) + (ny > 0) + (nz > 0)
                    if count == 1:
                        mtype = "axial"
                    elif count == 2:
                        mtype = "tangential"
                    else:
                        mtype = "oblique"
                    modes.append((f, nx, ny, nz, mtype))
        modes.sort(key=lambda m: m[0])
        return modes

    @staticmethod
    def schroeder_frequency(volume: float, rt60: float) -> float:
        if volume <= 0 or rt60 <= 0:
            return 0.0
        return 2000 * np.sqrt(rt60 / volume)

    @staticmethod
    def sabine_rt60(
        volume: float, surface_area: float, avg_absorption: float
    ) -> float:
        denom = surface_area * avg_absorption
        if denom <= 0:
            return 0.0
        return 0.161 * volume / denom

    @staticmethod
    def eyring_rt60(
        volume: float, surface_area: float, avg_absorption: float
    ) -> float:
        if surface_area <= 0 or avg_absorption <= 0 or avg_absorption >= 1.0:
            return 0.0
        return -0.161 * volume / (surface_area * np.log(1 - avg_absorption))

    @staticmethod
    def modal_density(freq: float, volume: float, c: float = 343.0) -> float:
        if freq <= 0 or volume <= 0:
            return 0.0
        return 4 * np.pi * freq ** 2 * volume / c ** 3

    @staticmethod
    def optimal_room_ratios() -> list[tuple[float, float, float, str]]:
        return [
            (1.0, 1.4, 1.9, "Bolt"),
            (1.0, 1.28, 1.54, "Sepmeyer A"),
            (1.0, 1.6, 2.33, "Bonello"),
        ]

    @staticmethod
    def early_reflection_times(
        source_pos: tuple[float, float, float],
        receiver_pos: tuple[float, float, float],
        room_dims: tuple[float, float, float],
    ) -> list[float]:
        c = 343.0
        s = np.array(source_pos)
        r = np.array(receiver_pos)
        dims = np.array(room_dims)
        direct = np.linalg.norm(s - r) / c
        times = []
        for axis in range(3):
            for wall in [0, dims[axis]]:
                image = s.copy()
                image[axis] = 2 * wall - s[axis]
                d = np.linalg.norm(image - r)
                times.append(d / c - direct)
        times.sort()
        return times
