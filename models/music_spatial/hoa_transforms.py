"""Higher-Order Ambisonics (HOA) transforms and spatial analysis.

Spherical harmonic encoding/decoding, soundfield rotation via
Wigner-d matrices, beamforming (plane-wave decomposition, max-rE),
and spatial analysis (diffuseness, direction-of-arrival).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import lpmv
from math import factorial


# --- ACN channel ordering, SN3D normalisation ---

def acn_index(l: int, m: int) -> int:
    """Ambisonic Channel Number for degree *l*, order *m*."""
    return l * l + l + m


def sn3d_norm(l: int, m: int) -> float:
    """SN3D (Schmidt semi-) normalisation factor."""
    return np.sqrt(
        (2.0 - (1.0 if m == 0 else 0.0))
        * factorial(l - abs(m))
        / factorial(l + abs(m))
    )


def n3d_norm(l: int, m: int) -> float:
    """N3D (full 3D) normalisation factor."""
    return sn3d_norm(l, m) * np.sqrt(2 * l + 1)


def _sph_harm_real(l: int, m: int, theta: NDArray, phi: NDArray) -> NDArray:
    """Real-valued spherical harmonic Y_l^m(θ,φ) via associated Legendre.

    Replaces scipy.special.sph_harm which was removed in scipy ≥ 1.17.
    θ = inclination (0=zenith), φ = azimuth. Uses SN3D normalisation.
    """
    abs_m = abs(m)
    # Associated Legendre polynomial P_l^|m|(cos θ)
    P = lpmv(abs_m, l, np.cos(theta))
    # SN3D normalisation
    norm = np.sqrt(
        (2.0 - (1.0 if m == 0 else 0.0))
        * factorial(l - abs_m)
        / factorial(l + abs_m)
    )
    if m > 0:
        return (norm * P * np.cos(m * phi)).astype(np.float64)
    elif m < 0:
        return (norm * P * np.sin(abs_m * phi)).astype(np.float64)
    else:
        return (norm * P).astype(np.float64)


def real_spherical_harmonic(l: int, m: int, theta: NDArray, phi: NDArray) -> NDArray:
    """Real-valued spherical harmonic Y_l^m(θ, φ) with SN3D normalisation.

    θ = inclination (0=zenith, π=nadir), φ = azimuth.
    """
    return _sph_harm_real(l, m, theta, phi)


# --- Encoding ---

def encode_hoa(
    mono: NDArray,
    azimuth: float,
    elevation: float,
    order: int,
    gain: float = 1.0,
) -> NDArray:
    """Encode a mono signal into HOA B-format (ACN/SN3D).

    Parameters
    ----------
    mono : (N,) source signal
    azimuth : source azimuth in radians
    elevation : source elevation in radians (-π/2 to π/2)
    order : ambisonics order (1=FOA, 2, 3, …)
    gain : source gain

    Returns
    -------
    bformat : (n_channels, N) ambisonics signals
    """
    n_ch = (order + 1) ** 2
    theta = np.pi / 2.0 - elevation  # inclination from zenith
    phi = azimuth

    coeffs = np.zeros(n_ch)
    for l_deg in range(order + 1):
        for m_ord in range(-l_deg, l_deg + 1):
            idx = acn_index(l_deg, m_ord)
            val = real_spherical_harmonic(l_deg, m_ord,
                                         np.array([theta]),
                                         np.array([phi]))
            coeffs[idx] = float(val[0])
    mono = np.asarray(mono, dtype=np.float64)
    return gain * np.outer(coeffs, mono)


def encode_hoa_multi(
    sources: List[Tuple[NDArray, float, float, float]],
    order: int,
) -> NDArray:
    """Encode multiple sources, each (signal, azimuth, elevation, gain)."""
    n_ch = (order + 1) ** 2
    length = max(len(s[0]) for s in sources)
    bformat = np.zeros((n_ch, length))
    for sig, az, el, g in sources:
        enc = encode_hoa(sig, az, el, order, g)
        bformat[:, :enc.shape[1]] += enc
    return bformat


# --- Decoding ---

def decode_hoa_speakers(
    bformat: NDArray,
    speaker_positions: NDArray,
    order: int,
    method: str = "projection",
) -> NDArray:
    """Decode HOA to loudspeaker feeds.

    Parameters
    ----------
    bformat : (n_channels, N)
    speaker_positions : (n_speakers, 2) — azimuth, elevation in radians
    order : ambisonics order
    method : 'projection' (mode matching) or 'maxre' (max-rE weighted)

    Returns
    -------
    feeds : (n_speakers, N)
    """
    n_ch = (order + 1) ** 2
    n_spk = speaker_positions.shape[0]

    # Build re-encoding matrix Y: (n_ch, n_spk)
    Y = np.zeros((n_ch, n_spk))
    for s in range(n_spk):
        az, el = speaker_positions[s]
        theta = np.pi / 2.0 - el
        phi = az
        for l_deg in range(order + 1):
            for m_ord in range(-l_deg, l_deg + 1):
                idx = acn_index(l_deg, m_ord)
                Y[idx, s] = float(real_spherical_harmonic(
                    l_deg, m_ord, np.array([theta]), np.array([phi])
                )[0])

    if method == "maxre":
        # Max-rE weighting: a_l = cos(π l / (2(order+1)))
        weights = np.zeros(n_ch)
        for l_deg in range(order + 1):
            w = np.cos(np.pi * l_deg / (2 * (order + 1)))
            for m_ord in range(-l_deg, l_deg + 1):
                weights[acn_index(l_deg, m_ord)] = w
        bformat = bformat * weights[:, np.newaxis]

    # Pseudo-inverse decoding (mode matching)
    D = np.linalg.pinv(Y)  # (n_spk, n_ch)
    return D @ bformat[:n_ch]


# --- Rotation ---

def _wigner_d_small(l: int, beta: float) -> NDArray:
    """Wigner (small) d-matrix of degree *l* for rotation angle *beta* about Y.

    Uses the recurrence relation. Returns (2l+1, 2l+1) matrix.
    """
    from scipy.special import comb as _comb
    size = 2 * l + 1
    d = np.zeros((size, size))
    for m1 in range(-l, l + 1):
        for m2 in range(-l, l + 1):
            s_min = max(0, m2 - m1)
            s_max = min(l + m2, l - m1)
            total = 0.0
            for s in range(s_min, s_max + 1):
                binom1 = _comb(l + m2, s, exact=True)
                binom2 = _comb(l - m2, l - m1 - s, exact=True)
                sign = (-1.0) ** (l - m1 - s)
                c_half = np.cos(beta / 2.0)
                s_half = np.sin(beta / 2.0)
                power_c = 2 * s + m1 - m2
                power_s = 2 * l - 2 * s - m1 + m2
                if power_c < 0 or power_s < 0:
                    continue
                total += sign * binom1 * binom2 * c_half ** power_c * s_half ** power_s
            pre = np.sqrt(
                float(factorial(l + m1))
                * float(factorial(l - m1))
                * float(factorial(l + m2))
                * float(factorial(l - m2))
            )
            d[m1 + l, m2 + l] = pre * total / (
                float(factorial(l + m2))
                * float(factorial(l - m1))
            ) if (l + m2) >= 0 and (l - m1) >= 0 else 0.0
    # Normalise using the correct formula
    for m1 in range(-l, l + 1):
        for m2 in range(-l, l + 1):
            s_min = max(0, m2 - m1)
            s_max = min(l + m2, l - m1)
            val = 0.0
            for s in range(s_min, s_max + 1):
                binom1 = _comb(l + m2, s, exact=True)
                binom2 = _comb(l - m2, l - m1 - s, exact=True)
                sign = (-1.0) ** (l - m1 - s)
                c_half = np.cos(beta / 2.0)
                s_half = np.sin(beta / 2.0)
                pc = 2 * s + m1 - m2
                ps = 2 * l - 2 * s - m1 + m2
                val += sign * binom1 * binom2 * (c_half ** pc) * (s_half ** ps)
            d[m1 + l, m2 + l] = val
    return d


def rotate_soundfield_euler(
    bformat: NDArray,
    yaw: float,
    pitch: float,
    roll: float,
    order: int,
) -> NDArray:
    """Rotate an HOA soundfield by Euler angles (ZYZ convention).

    Parameters
    ----------
    bformat : (n_channels, N) ACN/SN3D
    yaw, pitch, roll : rotation angles in radians (α, β, γ in ZYZ)
    order : ambisonics order

    Returns
    -------
    rotated : (n_channels, N)
    """
    n_ch = (order + 1) ** 2
    R = np.eye(n_ch)

    for l_deg in range(order + 1):
        size = 2 * l_deg + 1
        start = l_deg * l_deg

        d_beta = _wigner_d_small(l_deg, pitch)

        # Phase matrices for yaw and roll (diagonal in complex SH basis)
        # For real SH: R_z(α) applied as block rotation
        Rz_alpha = np.zeros((size, size))
        Rz_gamma = np.zeros((size, size))
        for m in range(-l_deg, l_deg + 1):
            i = m + l_deg
            if m == 0:
                Rz_alpha[i, i] = 1.0
                Rz_gamma[i, i] = 1.0
            elif m > 0:
                Rz_alpha[i, i] = np.cos(m * yaw)
                Rz_alpha[i, -m + l_deg] = -np.sin(m * yaw)
                Rz_alpha[-m + l_deg, i] = np.sin(m * yaw)
                Rz_alpha[-m + l_deg, -m + l_deg] = np.cos(m * yaw)
                Rz_gamma[i, i] = np.cos(m * roll)
                Rz_gamma[i, -m + l_deg] = -np.sin(m * roll)
                Rz_gamma[-m + l_deg, i] = np.sin(m * roll)
                Rz_gamma[-m + l_deg, -m + l_deg] = np.cos(m * roll)

        block = Rz_alpha @ d_beta @ Rz_gamma
        R[start:start + size, start:start + size] = block

    return R @ bformat[:n_ch]


# --- Spatial analysis ---

def sound_intensity_vector(bformat: NDArray) -> NDArray:
    """Active sound intensity vector from FOA (W, Y, Z, X) signals.

    Assumes ACN ordering: ch0=W, ch1=Y, ch2=Z, ch3=X.
    Returns (3, N) — [x, y, z] intensity components.
    """
    W = bformat[0]
    Y, Z, X = bformat[1], bformat[2], bformat[3]
    # Active intensity ∝ Re{p* · v} where p∝W, v∝(X,Y,Z)
    Ix = W * X
    Iy = W * Y
    Iz = W * Z
    return np.stack([Ix, Iy, Iz], axis=0)


def doa_from_intensity(bformat: NDArray, frame_size: int = 512) -> NDArray:
    """Direction-of-arrival from active intensity, frame-wise.

    Returns (n_frames, 2) — azimuth and elevation in radians.
    """
    IV = sound_intensity_vector(bformat)
    n_frames = IV.shape[1] // frame_size
    doa = np.zeros((n_frames, 2))
    for i in range(n_frames):
        start = i * frame_size
        end = start + frame_size
        Ix = np.mean(IV[0, start:end])
        Iy = np.mean(IV[1, start:end])
        Iz = np.mean(IV[2, start:end])
        azimuth = np.arctan2(Iy, Ix)
        r_horiz = np.sqrt(Ix ** 2 + Iy ** 2)
        elevation = np.arctan2(Iz, r_horiz)
        doa[i] = [azimuth, elevation]
    return doa


def diffuseness(bformat: NDArray, frame_size: int = 512) -> NDArray:
    """Instantaneous diffuseness estimate ψ ∈ [0, 1].

    ψ = 1 − ‖⟨I_a⟩‖ / ⟨E⟩ where I_a is active intensity, E is energy density.
    ψ = 0 for a single plane wave, 1 for a perfectly diffuse field.
    """
    W = bformat[0]
    Y, Z, X = bformat[1], bformat[2], bformat[3]

    n_frames = len(W) // frame_size
    psi = np.zeros(n_frames)
    for i in range(n_frames):
        s, e = i * frame_size, (i + 1) * frame_size
        w = W[s:e]
        x, y, z = X[s:e], Y[s:e], Z[s:e]
        Ia = np.array([np.mean(w * x), np.mean(w * y), np.mean(w * z)])
        E = np.mean(w ** 2) + np.mean(x ** 2 + y ** 2 + z ** 2)
        E = max(E, 1e-20)
        psi[i] = 1.0 - np.linalg.norm(Ia) / (E / 2.0 + 1e-20)
        psi[i] = np.clip(psi[i], 0.0, 1.0)
    return psi


# --- Beamforming ---

def plane_wave_decomposition(
    bformat: NDArray,
    look_directions: NDArray,
    order: int,
) -> NDArray:
    """Plane-wave decomposition: steer beams toward *look_directions*.

    Parameters
    ----------
    bformat : (n_channels, N) HOA signals
    look_directions : (n_beams, 2) — azimuth, elevation in radians
    order : ambisonics order

    Returns
    -------
    beams : (n_beams, N)
    """
    n_ch = (order + 1) ** 2
    n_beams = look_directions.shape[0]

    Y_mat = np.zeros((n_beams, n_ch))
    for b in range(n_beams):
        az, el = look_directions[b]
        theta = np.pi / 2.0 - el
        phi = az
        for l_deg in range(order + 1):
            for m_ord in range(-l_deg, l_deg + 1):
                idx = acn_index(l_deg, m_ord)
                Y_mat[b, idx] = float(real_spherical_harmonic(
                    l_deg, m_ord, np.array([theta]), np.array([phi])
                )[0])

    return Y_mat @ bformat[:n_ch]


def max_re_weights(order: int) -> NDArray:
    """Max-rE energy concentration weights per degree.

    Returns (n_channels,) array.
    """
    n_ch = (order + 1) ** 2
    w = np.zeros(n_ch)
    for l_deg in range(order + 1):
        wl = np.cos(np.pi * l_deg / (2 * (order + 1)))
        for m_ord in range(-l_deg, l_deg + 1):
            w[acn_index(l_deg, m_ord)] = wl
    return w


def in_phase_weights(order: int) -> NDArray:
    """In-phase decoding weights per degree (all-positive lobes)."""
    n_ch = (order + 1) ** 2
    w = np.zeros(n_ch)
    for l_deg in range(order + 1):
        wl = (
            factorial(order) ** 2
            / (factorial(order + l_deg) * factorial(order - l_deg))
        )
        for m_ord in range(-l_deg, l_deg + 1):
            w[acn_index(l_deg, m_ord)] = wl
    return w
