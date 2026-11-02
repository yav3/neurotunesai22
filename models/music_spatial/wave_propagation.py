"""Wave propagation: spherical spreading, air absorption, diffraction, waveguide mesh."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

SPEED_OF_SOUND = 343.0  # m/s at 20°C
AIR_DENSITY = 1.204      # kg/m³


@dataclass
class Medium:
    speed: float = SPEED_OF_SOUND
    density: float = AIR_DENSITY
    temperature_c: float = 20.0
    humidity_pct: float = 50.0

    def adjusted_speed(self) -> float:
        return 331.3 * np.sqrt(1.0 + self.temperature_c / 273.15)


@dataclass
class WavefrontSample:
    time: float
    pressure: float
    x: float
    y: float
    z: float = 0.0


def spherical_spreading(distance: float, ref_distance: float = 1.0) -> float:
    if distance <= 0:
        return 0.0
    return ref_distance / max(distance, 1e-12)


def cylindrical_spreading(distance: float, ref_distance: float = 1.0) -> float:
    if distance <= 0:
        return 0.0
    return np.sqrt(ref_distance / max(distance, 1e-12))


def air_absorption_coefficient(freq: float, temperature_c: float = 20.0,
                                humidity_pct: float = 50.0) -> float:
    T = temperature_c + 273.15
    T_ref = 293.15
    T_01 = T_ref
    h = humidity_pct
    f = freq
    p_sat = 10.0 ** (-6.8346 * (T_01 / T) ** 1.261 + 4.6151)
    h_mol = h * p_sat / 100.0
    f_rO = 24.0 + 4.04e4 * h_mol * (0.02 + h_mol) / (0.391 + h_mol)
    f_rN = (T / T_ref) ** (-0.5) * (9.0 + 280.0 * h_mol *
            np.exp(-4.170 * ((T / T_ref) ** (-1.0 / 3.0) - 1.0)))
    alpha = 8.686 * f * f * (
        1.84e-11 * (T / T_ref) ** 0.5 +
        (T / T_ref) ** (-2.5) * (
            0.01275 * np.exp(-2239.1 / T) / (f_rO + f * f / f_rO) +
            0.10680 * np.exp(-3352.0 / T) / (f_rN + f * f / f_rN)
        )
    )
    return float(alpha)


def apply_air_absorption(signal: np.ndarray, distance: float, sr: int,
                         medium: Optional[Medium] = None) -> np.ndarray:
    if medium is None:
        medium = Medium()
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spectrum = np.fft.rfft(signal)
    alphas = np.array([air_absorption_coefficient(f, medium.temperature_c,
                       medium.humidity_pct) for f in freqs])
    atten = np.exp(-alphas * distance)
    spectrum *= atten
    return np.fft.irfft(spectrum, n=n)


def delay_samples(distance: float, sr: int, speed: float = SPEED_OF_SOUND) -> float:
    return distance / speed * sr


def fractional_delay(signal: np.ndarray, delay: float) -> np.ndarray:
    n_int = int(np.floor(delay))
    frac = delay - n_int
    out = np.zeros(len(signal) + n_int + 1)
    for i in range(len(signal)):
        out[i + n_int] += signal[i] * (1.0 - frac)
        out[i + n_int + 1] += signal[i] * frac
    return out[:len(signal)]


@dataclass
class WaveguideMesh2D:
    nx: int
    ny: int
    damping: float = 0.001
    _pressure: np.ndarray = field(init=False, repr=False)
    _pressure_prev: np.ndarray = field(init=False, repr=False)
    _boundary: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._pressure = np.zeros((self.nx, self.ny))
        self._pressure_prev = np.zeros((self.nx, self.ny))
        self._boundary = np.ones((self.nx, self.ny))
        self._boundary[0, :] = 0
        self._boundary[-1, :] = 0
        self._boundary[:, 0] = 0
        self._boundary[:, -1] = 0

    def set_boundary(self, mask: np.ndarray) -> None:
        self._boundary = mask.astype(float)

    def excite(self, x: int, y: int, amplitude: float) -> None:
        self._pressure[x, y] += amplitude

    def step(self) -> np.ndarray:
        p = self._pressure
        pp = self._pressure_prev
        laplacian = (
            np.roll(p, 1, 0) + np.roll(p, -1, 0) +
            np.roll(p, 1, 1) + np.roll(p, -1, 1) - 4.0 * p
        )
        p_new = 2.0 * p - pp + 0.5 * laplacian
        p_new *= self._boundary
        p_new *= (1.0 - self.damping)
        self._pressure_prev = p.copy()
        self._pressure = p_new
        return p_new.copy()

    def run(self, n_steps: int, source_x: int, source_y: int,
            source_signal: np.ndarray, listen_x: int, listen_y: int) -> np.ndarray:
        output = np.zeros(n_steps)
        for t in range(n_steps):
            if t < len(source_signal):
                self.excite(source_x, source_y, source_signal[t])
            self.step()
            output[t] = self._pressure[listen_x, listen_y]
        return output


def diffraction_edge(freq: float, distance_source: float,
                     distance_receiver: float, wedge_angle: float = np.pi) -> float:
    wavelength = SPEED_OF_SOUND / max(freq, 1.0)
    path_diff = distance_source + distance_receiver
    N = 2.0 * path_diff / wavelength
    nu = wedge_angle / np.pi
    attenuation = 1.0 / (1.0 + np.sqrt(N) * nu)
    return float(np.clip(attenuation, 0, 1))


def reflection_coefficient(Z1: float, Z2: float) -> float:
    return (Z2 - Z1) / (Z2 + Z1) if (Z2 + Z1) != 0 else 0.0


def absorption_to_impedance(alpha: float) -> float:
    r = np.sqrt(1.0 - np.clip(alpha, 0, 0.999))
    Z_ratio = (1.0 + r) / (1.0 - r + 1e-12)
    return float(Z_ratio * AIR_DENSITY * SPEED_OF_SOUND)


def inverse_square_law(power: float, distance: float) -> float:
    return power / (4.0 * np.pi * max(distance, 1e-12) ** 2)


def sound_intensity_to_spl(intensity: float) -> float:
    I_ref = 1e-12
    if intensity <= 0:
        return -np.inf
    return 10.0 * np.log10(intensity / I_ref)


def spl_to_intensity(spl_db: float) -> float:
    return 1e-12 * 10.0 ** (spl_db / 10.0)


def doppler_shift(f0: float, v_source: float, v_listener: float = 0.0,
                  speed: float = SPEED_OF_SOUND) -> float:
    return f0 * (speed + v_listener) / (speed + v_source)


def fresnel_zone_radius(freq: float, distance: float) -> float:
    wavelength = SPEED_OF_SOUND / max(freq, 1.0)
    return float(np.sqrt(wavelength * distance))


@dataclass
class RayPath:
    points: List[Tuple[float, float, float]]
    total_distance: float
    reflections: int
    energy: float = 1.0


def trace_ray_2d(origin: Tuple[float, float], angle: float,
                 room_width: float, room_length: float,
                 max_reflections: int = 10) -> RayPath:
    x, y = origin
    dx = np.cos(angle)
    dy = np.sin(angle)
    points = [(x, y, 0.0)]
    total_dist = 0.0
    energy = 1.0
    wall_absorption = 0.1
    for _ in range(max_reflections):
        t_right = (room_width - x) / dx if dx > 0 else np.inf
        t_left = -x / dx if dx < 0 else np.inf
        t_top = (room_length - y) / dy if dy > 0 else np.inf
        t_bottom = -y / dy if dy < 0 else np.inf
        t_min = min(t for t in [t_right, t_left, t_top, t_bottom] if t > 1e-10)
        x += dx * t_min
        y += dy * t_min
        total_dist += t_min
        energy *= (1.0 - wall_absorption)
        points.append((x, y, 0.0))
        if abs(x) < 1e-8 or abs(x - room_width) < 1e-8:
            dx = -dx
        if abs(y) < 1e-8 or abs(y - room_length) < 1e-8:
            dy = -dy
        x = np.clip(x, 0, room_width)
        y = np.clip(y, 0, room_length)
    return RayPath(points=points, total_distance=total_dist,
                   reflections=max_reflections, energy=energy)
