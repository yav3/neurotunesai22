"""Room modal analysis, image-source method, and speech intelligibility.

Extends room_acoustics with detailed modal decomposition (axial, tangential,
oblique), higher-order image-source reflections, Schroeder frequency,
critical distance, lateral energy fraction, and a modulation-transfer-function
based Speech Transmission Index (STI).

Theory: Kuttruff (2000) *Room Acoustics* Ch. 3 (modes) / Ch. 9 (room
simulation); ISO 3382-1:2009 (C80, D50, Ts, EDT, LF); IEC 60268-16 (STI);
Barron & Lee (1988) image-source energy model; Schroeder (1962) frequency
limit.  Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence

SPEED_OF_SOUND = 343.0  # m/s at 20 °C


@dataclass
class RoomDimensionsMA:
    length: float   # m (x)
    width: float    # m (y)
    height: float   # m (z)

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def surface_area(self) -> float:
        l, w, h = self.length, self.width, self.height
        return 2.0 * (l * w + l * h + w * h)


@dataclass
class RoomMode:
    nx: int
    ny: int
    nz: int
    frequency: float
    mode_type: str   # "axial", "tangential", "oblique"


def room_modes(room: RoomDimensionsMA,
               max_freq: float = 500.0,
               c: float = SPEED_OF_SOUND) -> List[RoomMode]:
    """Compute rectangular room modes up to *max_freq* Hz.

    f(nx,ny,nz) = c/2 * sqrt((nx/Lx)^2 + (ny/Ly)^2 + (nz/Lz)^2)
    """
    modes: List[RoomMode] = []
    max_n = int(2 * max_freq * max(room.length, room.width, room.height) / c) + 1
    for nx in range(max_n + 1):
        for ny in range(max_n + 1):
            for nz in range(max_n + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                f = (c / 2.0) * np.sqrt(
                    (nx / room.length) ** 2
                    + (ny / room.width) ** 2
                    + (nz / room.height) ** 2
                )
                if f > max_freq:
                    continue
                nonzero = (nx > 0) + (ny > 0) + (nz > 0)
                if nonzero == 1:
                    mt = "axial"
                elif nonzero == 2:
                    mt = "tangential"
                else:
                    mt = "oblique"
                modes.append(RoomMode(nx, ny, nz, float(f), mt))
    modes.sort(key=lambda m: m.frequency)
    return modes


def mode_density(room: RoomDimensionsMA, freq: float,
                 c: float = SPEED_OF_SOUND) -> float:
    """Approximate number of modes per Hz at a given frequency (Kuttruff)."""
    V = room.volume
    S = room.surface_area
    L_total = 4.0 * (room.length + room.width + room.height)
    return (4.0 * np.pi * freq ** 2 * V / c ** 3
            + np.pi * freq * S / (2.0 * c ** 2)
            + L_total / (8.0 * c))


def schroeder_frequency(rt60: float, volume: float) -> float:
    """Transition frequency from modal to statistical behaviour (Schroeder 1962)."""
    if volume <= 0 or rt60 <= 0:
        return 0.0
    return 2000.0 * np.sqrt(rt60 / volume)


def critical_distance(volume: float, rt60: float,
                      directivity_q: float = 1.0) -> float:
    """Distance where direct field equals reverberant field.

    d_c = sqrt(Q * V / (16 * pi * c * T60)) — simplified Kuttruff.
    For standard Sabine rooms: d_c ≈ 0.057 * sqrt(Q * V / T60).
    """
    if rt60 <= 0 or volume <= 0:
        return 0.0
    return 0.057 * np.sqrt(directivity_q * volume / rt60)


@dataclass
class ImageSource:
    position: Tuple[float, float, float]
    delay_s: float
    attenuation: float
    order: int
    wall_sequence: Tuple[str, ...]


def image_source_reflections(
    room: RoomDimensionsMA,
    source: Tuple[float, float, float],
    receiver: Tuple[float, float, float],
    max_order: int = 2,
    wall_absorption: float = 0.1,
    c: float = SPEED_OF_SOUND,
) -> List[ImageSource]:
    """Compute image sources for a shoebox room up to *max_order*.

    Wall reflection coefficient r = sqrt(1 - alpha). Each image's
    attenuation is the product of r for every wall it reflects off.
    """
    sx, sy, sz = source
    rx, ry, rz = receiver
    Lx, Ly, Lz = room.length, room.width, room.height
    r_coeff = np.sqrt(1.0 - wall_absorption)
    images: List[ImageSource] = []

    rng = range(-max_order, max_order + 1)
    for ix in rng:
        for iy in rng:
            for iz in rng:
                order = abs(ix) + abs(iy) + abs(iz)
                if order == 0 or order > max_order:
                    continue
                # image position via mirroring formula
                if ix % 2 == 0:
                    img_x = ix * Lx + sx
                else:
                    img_x = ix * Lx + (Lx - sx) if ix > 0 else ix * Lx + (Lx - sx)
                if iy % 2 == 0:
                    img_y = iy * Ly + sy
                else:
                    img_y = iy * Ly + (Ly - sy) if iy > 0 else iy * Ly + (Ly - sy)
                if iz % 2 == 0:
                    img_z = iz * Lz + sz
                else:
                    img_z = iz * Lz + (Lz - sz) if iz > 0 else iz * Lz + (Lz - sz)

                dist = np.sqrt((img_x - rx) ** 2 + (img_y - ry) ** 2 + (img_z - rz) ** 2)
                delay = dist / c
                att = (r_coeff ** order) / max(dist, 1e-6)
                walls: List[str] = []
                if ix != 0:
                    walls.extend(["x+"] * ((abs(ix) + 1) // 2) + ["x-"] * (abs(ix) // 2))
                if iy != 0:
                    walls.extend(["y+"] * ((abs(iy) + 1) // 2) + ["y-"] * (abs(iy) // 2))
                if iz != 0:
                    walls.extend(["z+"] * ((abs(iz) + 1) // 2) + ["z-"] * (abs(iz) // 2))
                images.append(ImageSource(
                    position=(float(img_x), float(img_y), float(img_z)),
                    delay_s=float(delay),
                    attenuation=float(att),
                    order=order,
                    wall_sequence=tuple(walls[:order]),
                ))
    images.sort(key=lambda s: s.delay_s)
    return images


def image_source_rir(
    room: RoomDimensionsMA,
    source: Tuple[float, float, float],
    receiver: Tuple[float, float, float],
    duration_s: float = 1.0,
    sample_rate: int = 44100,
    max_order: int = 3,
    wall_absorption: float = 0.1,
    c: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """Synthesise a room impulse response via image-source method."""
    n_samples = int(duration_s * sample_rate)
    rir = np.zeros(n_samples)
    # direct sound
    dx = source[0] - receiver[0]
    dy = source[1] - receiver[1]
    dz = source[2] - receiver[2]
    dist_direct = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    idx_direct = int(dist_direct / c * sample_rate)
    if 0 <= idx_direct < n_samples:
        rir[idx_direct] += 1.0 / max(dist_direct, 1e-6)
    imgs = image_source_reflections(room, source, receiver, max_order,
                                    wall_absorption, c)
    for img in imgs:
        idx = int(img.delay_s * sample_rate)
        if 0 <= idx < n_samples:
            rir[idx] += img.attenuation
    return rir


def lateral_energy_fraction(
    impulse_response: np.ndarray,
    sample_rate: int,
    early_ms: float = 80.0,
) -> float:
    """Estimate Lateral Energy Fraction (LEF / LF80) from a mono IR.

    True LF requires a figure-of-eight mic; this approximation uses the
    comb-filtered alternating-sign proxy: LF ≈ Σ|h_odd|² / Σ|h_early|²
    over the 5–80 ms window (ISO 3382-1 simplified).
    """
    ir = np.asarray(impulse_response, dtype=float)
    n5 = int(0.005 * sample_rate)
    n_early = int(early_ms * 0.001 * sample_rate)
    n_early = min(n_early, len(ir))
    early = ir[n5:n_early]
    if len(early) < 2:
        return 0.0
    e_total = np.sum(early ** 2)
    if e_total < 1e-30:
        return 0.0
    odd_samples = early[1::2]
    e_lateral = np.sum(odd_samples ** 2)
    return float(np.clip(e_lateral / e_total, 0.0, 1.0))


def speech_transmission_index(
    impulse_response: np.ndarray,
    sample_rate: int,
    noise_floor_db: float = -60.0,
) -> float:
    """Simplified Speech Transmission Index from an impulse response.

    Computes the modulation transfer function (MTF) at 14 modulation
    frequencies (0.63–12.5 Hz) across octave bands, converts each to an
    apparent SNR, clips to ±15 dB, normalises to [0, 1], and averages
    (IEC 60268-16 simplified, single-channel).
    """
    ir = np.asarray(impulse_response, dtype=float)
    ir2 = ir ** 2
    e_total = np.sum(ir2)
    if e_total < 1e-30:
        return 0.0

    mod_freqs = np.array([0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5,
                          3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5])
    t = np.arange(len(ir)) / sample_rate
    noise_power = e_total * 10.0 ** (noise_floor_db / 10.0)

    snr_values = []
    for fm in mod_freqs:
        # MTF magnitude at modulation frequency fm
        cos_part = np.sum(ir2 * np.cos(2.0 * np.pi * fm * t))
        sin_part = np.sum(ir2 * np.sin(2.0 * np.pi * fm * t))
        mtf = np.sqrt(cos_part ** 2 + sin_part ** 2) / e_total
        # noise reduction
        mtf_eff = mtf / (1.0 + noise_power / max(e_total, 1e-30))
        mtf_eff = float(np.clip(mtf_eff, 1e-10, 1.0 - 1e-10))
        snr_app = 10.0 * np.log10(mtf_eff / (1.0 - mtf_eff))
        snr_values.append(float(np.clip(snr_app, -15.0, 15.0)))

    ti_values = [(s + 15.0) / 30.0 for s in snr_values]
    return float(np.mean(ti_values))


def bass_ratio(rt60_125: float, rt60_250: float,
               rt60_500: float, rt60_1000: float) -> float:
    """Bass Ratio (BR) — low-frequency warmth indicator (Beranek 1962)."""
    denom = rt60_500 + rt60_1000
    if denom <= 0:
        return 0.0
    return (rt60_125 + rt60_250) / denom


def brilliance_ratio(rt60_2000: float, rt60_4000: float,
                     rt60_500: float, rt60_1000: float) -> float:
    """Brilliance Ratio (Br) — high-frequency clarity (Beranek)."""
    denom = rt60_500 + rt60_1000
    if denom <= 0:
        return 0.0
    return (rt60_2000 + rt60_4000) / denom


def strength_g(impulse_response: np.ndarray,
               reference_ir: Optional[np.ndarray] = None) -> float:
    """Sound strength G (dB) — ISO 3382-1.

    If no reference IR is given, uses free-field at 10 m approximation.
    """
    ir = np.asarray(impulse_response, dtype=float)
    e_room = np.sum(ir ** 2)
    if reference_ir is not None:
        e_ref = np.sum(np.asarray(reference_ir, dtype=float) ** 2)
    else:
        e_ref = 1.0  # normalised free-field
    if e_ref < 1e-30:
        return 0.0
    return float(10.0 * np.log10(max(e_room, 1e-30) / e_ref))


def interaural_cross_correlation(
    ir_left: np.ndarray,
    ir_right: np.ndarray,
    sample_rate: int,
    early_ms: float = 80.0,
) -> float:
    """Early Interaural Cross-Correlation Coefficient (IACC_E).

    Measures spatial impression from binaural impulse responses.
    """
    n_early = int(early_ms * 0.001 * sample_rate)
    left = np.asarray(ir_left, dtype=float)[:n_early]
    right = np.asarray(ir_right, dtype=float)[:n_early]
    if len(left) < 2 or len(right) < 2:
        return 0.0
    e_l = np.sum(left ** 2)
    e_r = np.sum(right ** 2)
    if e_l < 1e-30 or e_r < 1e-30:
        return 0.0
    max_lag = min(int(0.001 * sample_rate), len(left) // 2)
    max_cc = 0.0
    norm = np.sqrt(e_l * e_r)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            cc = np.sum(left[lag:] * right[:len(left) - lag])
        else:
            cc = np.sum(left[:len(left) + lag] * right[-lag:])
        max_cc = max(max_cc, abs(cc) / norm)
    return float(np.clip(max_cc, 0.0, 1.0))
