"""Bowed string instrument physics: friction, modal synthesis, spectral models.

Covers the core mechanics of bowed string instruments --- bow-string friction,
Helmholtz motion, modal analysis with stiffness correction, wolf notes, and
extended techniques (sul ponticello, col legno, vibrato).

Conceptual grounding: Fletcher & Rossing (1998) *The Physics of Musical
Instruments*, 2nd ed., Springer.  Friction curve follows Smith (1986).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StringParams:
    """Physical parameters of a bowed string.

    Attributes:
        length: vibrating length in metres (nut to bridge).
        tension: string tension in Newtons.
        linear_density: mass per unit length in kg/m.
        Q_factor: quality factor governing decay (higher = less damping).
        stiffness_B: inharmonicity coefficient B for stiffness correction
            (dimensionless; 0 for an ideal flexible string).
    """
    length: float = 0.328            # ~ violin A string
    tension: float = 73.0            # Newtons
    linear_density: float = 0.00057  # kg/m  (wound steel A)
    Q_factor: float = 1500.0
    stiffness_B: float = 2.0e-5     # small for wound strings


@dataclass
class BowedStringState:
    """Instantaneous state of the bow-string system.

    Attributes:
        position: transverse displacement of the string at the bow point (m).
        velocity: transverse velocity at the bow point (m/s).
        bow_force: normal force of the bow on the string (N).
        bow_velocity: tangential velocity of the bow (m/s).
        friction_state: 1.0 for stick, 0.0 for slip.
    """
    position: float = 0.0
    velocity: float = 0.0
    bow_force: float = 0.5
    bow_velocity: float = 0.2
    friction_state: float = 1.0


# ---------------------------------------------------------------------------
# Characteristic impedance
# ---------------------------------------------------------------------------

def string_impedance(tension: float, linear_density: float) -> float:
    """Characteristic impedance Z_0 = sqrt(T * mu).

    Parameters:
        tension: string tension (N).
        linear_density: mass per unit length (kg/m).

    Returns:
        Impedance in kg/s.
    """
    if tension <= 0.0 or linear_density <= 0.0:
        raise ValueError("tension and linear_density must be positive")
    return math.sqrt(tension * linear_density)


# ---------------------------------------------------------------------------
# Helmholtz motion
# ---------------------------------------------------------------------------

def helmholtz_motion(
    n_samples: int,
    bow_position_ratio: float = 0.1284,
    bow_velocity: float = 0.2,
    sr: float = 44100.0,
) -> np.ndarray:
    """Generate the ideal Helmholtz sawtooth displacement of a bowed string.

    The Helmholtz motion is a triangular displacement that travels as two
    straight-line segments around the string, producing the characteristic
    sawtooth velocity waveform at the bridge.

    Parameters:
        n_samples: number of output samples.
        bow_position_ratio: beta = bow distance from bridge / string length.
            Default 0.1284 ~ 1/7.8 (typical for violin).
        bow_velocity: tangential bow velocity in m/s.
        sr: sample rate in Hz.

    Returns:
        1-D array of normalised displacement values in [-1, 1].
    """
    if n_samples <= 0:
        return np.array([], dtype=np.float64)

    beta = np.clip(bow_position_ratio, 0.01, 0.99)
    # The fundamental period in samples is set so the duty cycle matches beta.
    # We treat the full period as 1.0 (normalised).
    t = np.arange(n_samples, dtype=np.float64) / sr
    # Normalise time to [0, period) with an arbitrary fundamental that fills
    # one period at the Helmholtz frequency  f0 = v_bow / (2 * beta * L).
    # Since we normalise amplitude anyway, set period = 1 / f0 ~ 1.0/440.
    # A more useful default: produce a 440 Hz tone.
    period = 1.0 / 440.0
    phase = np.mod(t, period) / period  # [0, 1)

    # Helmholtz displacement: piecewise linear, peak at phase = beta
    displacement = np.where(
        phase < beta,
        phase / beta,
        (1.0 - phase) / (1.0 - beta),
    )
    # Centre about zero and normalise to [-1, 1]
    displacement = 2.0 * displacement - 1.0
    # Scale by bow velocity sign
    displacement *= np.sign(bow_velocity) if bow_velocity != 0 else 1.0
    return displacement


# ---------------------------------------------------------------------------
# Bridge force from Helmholtz displacement
# ---------------------------------------------------------------------------

def bridge_force_waveform(
    helmholtz_displacement: np.ndarray,
    bow_position_ratio: float = 0.1284,
    n_samples: Optional[int] = None,
) -> np.ndarray:
    """Compute the force waveform at the bridge from Helmholtz displacement.

    The bridge force is proportional to the spatial derivative of the
    displacement at x = 0 (bridge end).  For ideal Helmholtz motion the
    result is a sawtooth whose amplitude scales as 1 / (1 - beta).

    Parameters:
        helmholtz_displacement: displacement waveform (from *helmholtz_motion*).
        bow_position_ratio: beta (bow distance / string length).
        n_samples: if given, truncate or zero-pad to this length.

    Returns:
        1-D array of bridge force values (arbitrary units, normalised peak = 1).
    """
    beta = np.clip(bow_position_ratio, 0.01, 0.99)
    # The gradient at the bridge is ~ displacement / (1 - beta)
    force = helmholtz_displacement / (1.0 - beta)
    # Normalise
    peak = np.max(np.abs(force))
    if peak > 0:
        force = force / peak
    if n_samples is not None:
        if n_samples < len(force):
            force = force[:n_samples]
        elif n_samples > len(force):
            force = np.pad(force, (0, n_samples - len(force)))
    return force


# ---------------------------------------------------------------------------
# Bow-string friction (Smith 1986 hyperbolic model)
# ---------------------------------------------------------------------------

def bow_string_friction(
    relative_velocity: float | np.ndarray,
    static_coeff: float = 0.8,
    dynamic_coeff: float = 0.3,
) -> float | np.ndarray:
    """Hyperbolic friction curve for bow-string interaction (Smith 1986).

    f(v_rel) = mu_d + (mu_s - mu_d) * exp(-|v_rel| / v0)

    where v0 is chosen so the curve drops to approximately mu_d at
    moderate relative velocities.

    Parameters:
        relative_velocity: v_bow - v_string  (m/s).  Scalar or array.
        static_coeff: static friction coefficient mu_s.
        dynamic_coeff: dynamic (sliding) friction coefficient mu_d.

    Returns:
        Friction force coefficient (same shape as *relative_velocity*).
        Sign follows the relative velocity (opposes slip).
    """
    v0 = 0.1  # velocity scale (m/s)
    v_rel = np.asarray(relative_velocity, dtype=np.float64)
    magnitude = dynamic_coeff + (static_coeff - dynamic_coeff) * np.exp(
        -np.abs(v_rel) / v0
    )
    # Friction opposes relative motion
    result = magnitude * np.sign(v_rel)
    if np.ndim(relative_velocity) == 0:
        return float(result)
    return result


# ---------------------------------------------------------------------------
# Modal frequencies with stiffness correction (Fletcher & Rossing)
# ---------------------------------------------------------------------------

def modal_frequencies(
    string_params: StringParams,
    n_modes: int = 16,
) -> np.ndarray:
    """Natural frequencies of a stiff string.

    f_n = n * f_1 * sqrt(1 + B * n^2)

    where B is the inharmonicity coefficient and f_1 is the ideal
    fundamental  f_1 = (1 / 2L) * sqrt(T / mu).

    Parameters:
        string_params: physical string parameters.
        n_modes: number of modes to compute.

    Returns:
        1-D array of shape (n_modes,) with mode frequencies in Hz.
    """
    sp = string_params
    if sp.length <= 0 or sp.tension <= 0 or sp.linear_density <= 0:
        raise ValueError("length, tension, and linear_density must be positive")
    f1_ideal = (1.0 / (2.0 * sp.length)) * math.sqrt(sp.tension / sp.linear_density)
    ns = np.arange(1, n_modes + 1, dtype=np.float64)
    freqs = ns * f1_ideal * np.sqrt(1.0 + sp.stiffness_B * ns ** 2)
    return freqs


# ---------------------------------------------------------------------------
# Mode amplitudes at bow position
# ---------------------------------------------------------------------------

def mode_amplitudes_at_position(
    bow_position: float,
    string_length: float,
    n_modes: int = 16,
) -> np.ndarray:
    """Spatial Fourier coefficients (mode shapes) evaluated at the bow point.

    The nth mode shape of a fixed-fixed string is  sin(n * pi * x / L).

    Parameters:
        bow_position: distance from bridge to bow contact (m).
        string_length: total vibrating length (m).
        n_modes: number of modes.

    Returns:
        1-D array of shape (n_modes,).
    """
    if string_length <= 0:
        raise ValueError("string_length must be positive")
    ns = np.arange(1, n_modes + 1, dtype=np.float64)
    return np.sin(ns * math.pi * bow_position / string_length)


# ---------------------------------------------------------------------------
# Simple modal synthesis with friction coupling
# ---------------------------------------------------------------------------

def simulate_bowed_string(
    string_params: StringParams,
    bow_force: float = 0.5,
    bow_velocity: float = 0.2,
    bow_position: float = 0.04,
    n_samples: int = 44100,
    sr: float = 44100.0,
) -> np.ndarray:
    """Synthesise bowed string audio via modal decomposition with friction.

    A lightweight modal model: each mode is an oscillator whose amplitude
    is excited through the friction force projected onto that mode shape,
    and decays according to the string Q-factor.  The friction force is
    evaluated at the bow point using the Smith (1986) hyperbolic curve.

    Parameters:
        string_params: physical string parameters.
        bow_force: normal bow force on string (N).
        bow_velocity: tangential bow velocity (m/s).
        bow_position: distance from bridge to bow contact (m).
        n_samples: output length in samples.
        sr: sample rate.

    Returns:
        1-D array of synthesised audio (normalised to [-1, 1]).
    """
    n_modes = 32
    freqs = modal_frequencies(string_params, n_modes)
    phis = mode_amplitudes_at_position(bow_position, string_params.length, n_modes)

    dt = 1.0 / sr
    Z0 = string_impedance(string_params.tension, string_params.linear_density)

    # Per-mode state: amplitude and phase velocity
    a = np.zeros(n_modes, dtype=np.float64)
    v_mode = np.zeros(n_modes, dtype=np.float64)

    omega = 2.0 * math.pi * freqs
    # Decay rate per mode from Q
    gamma = omega / (2.0 * string_params.Q_factor)

    output = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        # String velocity at bow point = sum of modal velocities * mode shapes
        v_string = float(np.dot(v_mode, phis))

        # Friction
        v_rel = bow_velocity - v_string
        f_friction = bow_force * bow_string_friction(v_rel)

        # Drive each mode
        for m in range(n_modes):
            # Force on mode = friction * mode shape at bow point
            f_m = f_friction * phis[m]
            # Update modal velocity (simple Euler)
            v_mode[m] += dt * (f_m / Z0 - gamma[m] * v_mode[m] - omega[m] ** 2 * a[m])
            a[m] += dt * v_mode[m]

        # Output = displacement at a listening point (bridge ~ x=0)
        # At x = 0 all mode shapes are 0, so use a point near bridge
        listen_ratio = 0.01
        listen_phis = np.sin(
            np.arange(1, n_modes + 1) * math.pi * listen_ratio
        )
        output[i] = float(np.dot(a, listen_phis))

    # Normalise
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak
    return output


# ---------------------------------------------------------------------------
# Wolf note prediction
# ---------------------------------------------------------------------------

def wolf_note_frequency(
    string_params: StringParams,
    body_resonance_freq: float = 280.0,
    body_mass: float = 0.45,
) -> float:
    """Predict the wolf note frequency from string-body coupling.

    The wolf note occurs when a string mode falls near the main body
    resonance, causing a periodic energy exchange.  The wolf beat
    frequency is approximately:

        f_wolf = f_body * sqrt(m_string / m_body)

    where m_string is the effective mass of the string mode nearest to
    the body resonance.

    Parameters:
        string_params: physical string parameters.
        body_resonance_freq: main body resonance frequency (Hz).
            Default 280 Hz ~ typical cello body mode.
        body_mass: effective body mass at the bridge (kg).

    Returns:
        Wolf note frequency in Hz.
    """
    m_string = string_params.linear_density * string_params.length
    if body_mass <= 0:
        raise ValueError("body_mass must be positive")
    f_wolf = body_resonance_freq * math.sqrt(m_string / body_mass)
    return f_wolf


# ---------------------------------------------------------------------------
# Sul ponticello spectrum (bowing near the bridge)
# ---------------------------------------------------------------------------

def sul_ponticello_spectrum(
    fundamental: float,
    n_harmonics: int = 32,
    bow_position_ratio: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spectral content when bowing near the bridge (sul ponticello).

    Bowing close to the bridge emphasises high harmonics because the
    modes that have a node near the bow point are *not* suppressed.
    The amplitude of harmonic n is approximately:

        A_n ~ 1 / (n * |sin(n * pi * beta)|)

    with a floor at modes where the sin term is near zero (i.e. the
    bow sits on a node, which paradoxically excites that mode less).

    Parameters:
        fundamental: fundamental frequency (Hz).
        n_harmonics: number of harmonics to compute.
        bow_position_ratio: beta (bow distance / string length).

    Returns:
        (frequencies, amplitudes) --- both 1-D arrays of length *n_harmonics*.
        Amplitudes are normalised so the maximum is 1.
    """
    ns = np.arange(1, n_harmonics + 1, dtype=np.float64)
    freqs = fundamental * ns
    sin_vals = np.abs(np.sin(ns * math.pi * bow_position_ratio))
    # Avoid division by zero at nodes
    sin_vals = np.maximum(sin_vals, 0.05)
    amps = 1.0 / (ns * sin_vals)
    # Normalise
    amps /= np.max(amps)
    return freqs, amps


# ---------------------------------------------------------------------------
# Col legno spectrum (striking with the wood of the bow)
# ---------------------------------------------------------------------------

def col_legno_spectrum(
    fundamental: float,
    n_harmonics: int = 32,
    decay_rate: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spectral content for col legno (wood-striking) technique.

    Col legno produces a percussive, somewhat pitch-ambiguous sound.
    The spectrum is modelled as a broadband excitation with a steeper
    harmonic roll-off than arco (bowed) playing, reflecting the
    impulsive excitation and the poor coupling of a rigid bow stick
    to the string.

    Parameters:
        fundamental: fundamental frequency (Hz).
        n_harmonics: number of harmonics.
        decay_rate: spectral roll-off exponent (higher = duller).

    Returns:
        (frequencies, amplitudes) as 1-D arrays.  Amplitudes normalised
        to peak = 1.
    """
    ns = np.arange(1, n_harmonics + 1, dtype=np.float64)
    freqs = fundamental * ns
    amps = 1.0 / ns ** decay_rate
    amps /= np.max(amps)
    return freqs, amps


# ---------------------------------------------------------------------------
# Vibrato (FM modulation)
# ---------------------------------------------------------------------------

def vibrato_modulation(
    fundamental: float,
    rate_hz: float = 5.5,
    depth_cents: float = 30.0,
    n_samples: int = 44100,
    sr: float = 44100.0,
) -> np.ndarray:
    """Synthesise vibrato via frequency modulation.

    Parameters:
        fundamental: carrier (nominal) frequency in Hz.
        rate_hz: vibrato rate (Hz).  Typical violin vibrato: 5 -- 7 Hz.
        depth_cents: peak deviation in cents.  Typical: 20 -- 50 cents.
        n_samples: output length.
        sr: sample rate.

    Returns:
        1-D array of the modulated sinusoidal waveform in [-1, 1].
    """
    t = np.arange(n_samples, dtype=np.float64) / sr
    # Convert cents depth to frequency ratio
    depth_ratio = 2.0 ** (depth_cents / 1200.0) - 1.0
    max_deviation = fundamental * depth_ratio  # Hz

    # Instantaneous frequency
    f_inst = fundamental + max_deviation * np.sin(2.0 * math.pi * rate_hz * t)

    # Phase is integral of instantaneous frequency
    phase = 2.0 * math.pi * np.cumsum(f_inst) / sr
    return np.sin(phase)


# ---------------------------------------------------------------------------
# Public API list
# ---------------------------------------------------------------------------

__all__ = [
    "BowedStringState",
    "StringParams",
    "string_impedance",
    "helmholtz_motion",
    "bridge_force_waveform",
    "bow_string_friction",
    "modal_frequencies",
    "mode_amplitudes_at_position",
    "simulate_bowed_string",
    "wolf_note_frequency",
    "sul_ponticello_spectrum",
    "col_legno_spectrum",
    "vibrato_modulation",
]
