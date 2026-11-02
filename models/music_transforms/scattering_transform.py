"""Scattering transform for music signals.

Implements the wavelet scattering transform up to second order, following
Mallat (2012) "Group Invariant Scattering" and Andén & Mallat (2014)
"Deep Scattering Spectrum" for music/audio applications.

The scattering transform provides a locally translation-invariant,
Lipschitz-continuous representation of signals by cascading wavelet
modulus operators and averaging with a low-pass filter.  For music:

  - Order 0: overall energy (like RMS)
  - Order 1: spectral envelope (similar to mel-frequency features)
  - Order 2: amplitude modulation / spectro-temporal patterns
              (vibrato, tremolo, rhythmic texture)

Pure numpy/scipy implementation --- no ML frameworks required.

References
----------
.. [1] S. Mallat, "Group Invariant Scattering", *Communications on Pure
       and Applied Mathematics*, 65(10):1331--1398, 2012.
.. [2] J. Andén and S. Mallat, "Deep Scattering Spectrum", *IEEE Trans.
       Signal Processing*, 62(16):4114--4128, 2014.
.. [3] J. Andén, V. Lostanlen, and S. Mallat, "Joint Time--Frequency
       Scattering for Audio Classification", *IEEE MLSP*, 2015.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FilterBankParams:
    """Parameters for a wavelet filter bank.

    Attributes
    ----------
    J : int
        Number of octaves (scales).  The scattering transform is
        invariant to translations up to ``2**J`` samples.
    Q : int
        Number of wavelets per octave (quality factor).  Higher *Q*
        gives finer frequency resolution at the cost of coarser time
        resolution.
    sample_rate : float
        Sampling rate of the input signal in Hz.
    """

    J: int
    Q: int
    sample_rate: float

    def __post_init__(self) -> None:
        if self.J < 1:
            raise ValueError(f"J must be >= 1, got {self.J}")
        if self.Q < 1:
            raise ValueError(f"Q must be >= 1, got {self.Q}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")


@dataclass
class ScatteringCoefficients:
    """Container for scattering coefficients at orders 0, 1, and 2.

    Attributes
    ----------
    order_0 : NDArray
        Zeroth-order coefficient --- shape ``(N,)`` (time-domain
        averaged signal energy).
    order_1 : NDArray
        First-order coefficients --- shape ``(n_filters, N)``.
    order_2 : NDArray
        Second-order coefficients --- shape ``(n_paths, N)`` where
        *n_paths* is the number of valid ``(j1, j2)`` pairs.
    meta : dict
        Auxiliary information: filter bank params, centre frequencies,
        path indices, etc.
    """

    order_0: NDArray[np.floating]
    order_1: NDArray[np.floating]
    order_2: NDArray[np.floating]
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wavelet construction
# ---------------------------------------------------------------------------

def morlet_wavelet(
    N: int,
    xi: float,
    sigma: float,
) -> NDArray[np.complexfloating]:
    r"""Analytic Morlet wavelet in the frequency domain.

    The Morlet wavelet is a complex sinusoid windowed by a Gaussian::

        \hat{\psi}(\omega) = \sigma\sqrt{2\pi} \;
            \exp\!\bigl(-\tfrac{\sigma^2}{2}(\omega - \xi)^2\bigr)

    evaluated on the DFT frequency grid
    ``[0, 2\pi/N, \ldots, 2\pi(N-1)/N]``.

    Parameters
    ----------
    N : int
        Length of the frequency-domain filter (= signal length).
    xi : float
        Centre frequency in radians ``(0, \pi]``.
    sigma : float
        Bandwidth parameter.  Larger *sigma* means narrower bandwidth.

    Returns
    -------
    NDArray[np.complexfloating]
        Frequency-domain wavelet of length *N*, analytic (one-sided).
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    omega = np.arange(N) * (2.0 * np.pi / N)

    # Gaussian envelope centred at xi
    gauss = np.exp(-0.5 * (sigma ** 2) * (omega - xi) ** 2)

    # Normalise so that the wavelet has unit L2 norm in frequency
    norm = sigma * np.sqrt(2.0 * np.pi)
    psi_hat = norm * gauss

    return psi_hat.astype(np.complex128)


def build_wavelet_filterbank(
    J: int,
    Q: int,
    N: int,
) -> tuple[list[NDArray[np.complexfloating]], NDArray[np.complexfloating]]:
    r"""Construct a wavelet filter bank for the scattering transform.

    Builds ``J * Q`` band-pass Morlet wavelets logarithmically spaced
    across *J* octaves with *Q* filters per octave, plus a low-pass
    averaging filter :math:`\varphi` whose cut-off is
    :math:`\sim 2^{-J}`.

    Parameters
    ----------
    J : int
        Number of octaves.
    Q : int
        Wavelets per octave.
    N : int
        Signal length (DFT size).

    Returns
    -------
    psi_filters : list[NDArray]
        Band-pass wavelets in frequency domain, ordered from high to
        low centre frequency.
    phi_filter : NDArray
        Low-pass averaging filter in frequency domain.
    """
    if J < 1 or Q < 1 or N < 1:
        raise ValueError("J, Q, N must all be >= 1")

    xi_max = np.pi * 0.85       # highest centre freq (< pi to avoid aliasing)
    sigma_base = 0.13 * Q       # bandwidth tied to Q

    psi_filters: list[NDArray[np.complexfloating]] = []
    for j in range(J):
        for q in range(Q):
            # Dyadic-geometric spacing: xi_{j,q} = xi_max * 2^{-j - q/Q}
            exponent = j + q / Q
            xi_jq = xi_max * (2.0 ** (-exponent))
            # Wider in time (= narrower in freq) for lower freqs
            sigma_jq = sigma_base * (2.0 ** exponent)
            psi_hat = morlet_wavelet(N, xi_jq, sigma_jq)
            psi_filters.append(psi_hat)

    # Low-pass (Gaussian centred at 0 with scale ~ 2^J)
    omega = np.arange(N) * (2.0 * np.pi / N)
    sigma_phi = sigma_base * (2.0 ** J)
    phi_hat = np.exp(-0.5 * (sigma_phi ** 2) * omega ** 2).astype(np.complex128)

    # Unit DC gain: phi(0) = 1
    if np.abs(phi_hat[0]) > 0:
        phi_hat /= np.abs(phi_hat[0])

    return psi_filters, phi_hat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fft_convolve(
    signal_hat: NDArray[np.complexfloating],
    filter_hat: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Multiply in frequency domain, IFFT, and take modulus."""
    product = signal_hat * filter_hat
    convolved = np.fft.ifft(product)
    return np.abs(convolved).astype(np.float64)


def _subsample(x: NDArray, factor: int) -> NDArray:
    """Subsample by an integer factor (anti-aliased by the low-pass)."""
    if factor <= 1:
        return x
    return x[::factor]


# ---------------------------------------------------------------------------
# Scattering operators
# ---------------------------------------------------------------------------

def scattering_order_0(
    signal: NDArray[np.floating],
    phi_filter: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    r"""Zeroth-order scattering: low-pass average of the signal.

    .. math::

        S_0 x = |x| * \varphi

    (Mallat 2012, Eq. 2.5).  For generality we apply ``|x|`` first.

    Parameters
    ----------
    signal : NDArray
        Input signal of length *N*.
    phi_filter : NDArray
        Low-pass averaging filter in frequency domain (length *N*).

    Returns
    -------
    NDArray
        Zeroth-order scattering coefficients of length *N*.
    """
    N = len(signal)
    x_hat = np.fft.fft(np.abs(signal).astype(np.float64), n=N)
    return _fft_convolve(x_hat, phi_filter)


def scattering_order_1(
    signal: NDArray[np.floating],
    psi_filters: list[NDArray[np.complexfloating]],
    phi_filter: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    r"""First-order scattering coefficients.

    .. math::

        S_1 x(t,\,j) = \bigl| |x * \psi_j| * \varphi \bigr|(t)

    This is the classic "wavelet modulus + average" that captures the
    spectral envelope --- analogous to mel-frequency features but with
    proven invariance / stability properties (Mallat 2012, section 2).

    Parameters
    ----------
    signal : NDArray
        Input signal of length *N*.
    psi_filters : list[NDArray]
        Band-pass wavelet filters in frequency domain.
    phi_filter : NDArray
        Low-pass averaging filter in frequency domain.

    Returns
    -------
    NDArray
        Shape ``(len(psi_filters), N)`` --- one row per wavelet scale.
    """
    N = len(signal)
    x_hat = np.fft.fft(signal.astype(np.float64), n=N)

    n_filters = len(psi_filters)
    S1 = np.zeros((n_filters, N), dtype=np.float64)

    for j, psi_j in enumerate(psi_filters):
        # |x * psi_j|
        u1_j = _fft_convolve(x_hat, psi_j)
        # |x * psi_j| * phi
        u1_hat = np.fft.fft(u1_j, n=N)
        S1[j, :] = _fft_convolve(u1_hat, phi_filter)

    return S1


def scattering_order_2(
    signal: NDArray[np.floating],
    psi_filters_1: list[NDArray[np.complexfloating]],
    psi_filters_2: list[NDArray[np.complexfloating]],
    phi_filter: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    r"""Second-order scattering coefficients.

    .. math::

        S_2 x(t,\,j_1,\,j_2) =
            \bigl| \bigl| |x * \psi_{j_1}| * \psi_{j_2}\bigr|
            * \varphi \bigr|(t)

    Captures amplitude modulations (AM) of the first-order envelopes
    --- musically relevant for vibrato, tremolo, and rhythmic texture.
    Following Andén & Mallat (2014), only paths with ``j2 > j1`` are
    computed (the modulation frequency must be lower than the carrier).

    Parameters
    ----------
    signal : NDArray
        Input signal of length *N*.
    psi_filters_1 : list[NDArray]
        First-layer wavelet filters (carrier frequencies).
    psi_filters_2 : list[NDArray]
        Second-layer wavelet filters (modulation frequencies).
    phi_filter : NDArray
        Low-pass averaging filter.

    Returns
    -------
    NDArray
        Shape ``(n_paths, N)`` where *n_paths* is the number of valid
        ``(j1, j2)`` pairs with ``j2 > j1``.
    """
    N = len(signal)
    x_hat = np.fft.fft(signal.astype(np.float64), n=N)

    rows: list[NDArray[np.floating]] = []

    for j1, psi_j1 in enumerate(psi_filters_1):
        # First wavelet modulus: |x * psi_{j1}|
        u1 = _fft_convolve(x_hat, psi_j1)
        u1_hat = np.fft.fft(u1, n=N)

        for j2, psi_j2 in enumerate(psi_filters_2):
            if j2 <= j1:
                continue  # modulation scale must be coarser
            # Second wavelet modulus: ||x * psi_{j1}| * psi_{j2}|
            u2 = _fft_convolve(u1_hat, psi_j2)
            # Average: ... * phi
            u2_hat = np.fft.fft(u2, n=N)
            s2 = _fft_convolve(u2_hat, phi_filter)
            rows.append(s2)

    if len(rows) == 0:
        return np.zeros((0, N), dtype=np.float64)

    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Full scattering transform
# ---------------------------------------------------------------------------

def compute_scattering(
    signal: NDArray[np.floating],
    J: int = 8,
    Q: int = 8,
    sr: float = 22050.0,
) -> ScatteringCoefficients:
    """Full scattering transform up to second order.

    Convenience wrapper that builds the filter bank, computes orders
    0--2, and returns them in a :class:`ScatteringCoefficients`
    container.

    Parameters
    ----------
    signal : NDArray
        Input waveform (mono, 1-D).
    J : int
        Number of octaves.
    Q : int
        Wavelets per octave.
    sr : float
        Sample rate in Hz.

    Returns
    -------
    ScatteringCoefficients
        Zeroth-, first-, and second-order coefficients with metadata.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    N = len(signal)

    psi_filters, phi_filter = build_wavelet_filterbank(J, Q, N)

    s0 = scattering_order_0(signal, phi_filter)
    s1 = scattering_order_1(signal, psi_filters, phi_filter)
    s2 = scattering_order_2(signal, psi_filters, psi_filters, phi_filter)

    # Centre frequencies for downstream use
    xi_max = np.pi * 0.85
    center_freqs = []
    for j in range(J):
        for q in range(Q):
            xi_jq = xi_max * (2.0 ** (-(j + q / Q)))
            freq_hz = xi_jq / (2.0 * np.pi) * sr
            center_freqs.append(freq_hz)

    meta = {
        "J": J,
        "Q": Q,
        "sample_rate": sr,
        "N": N,
        "n_filters": len(psi_filters),
        "n_order2_paths": s2.shape[0],
        "center_frequencies_hz": np.array(center_freqs),
    }

    return ScatteringCoefficients(
        order_0=s0,
        order_1=s1,
        order_2=s2,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Distance and invariance utilities
# ---------------------------------------------------------------------------

def scattering_distance(
    coeff_a: ScatteringCoefficients,
    coeff_b: ScatteringCoefficients,
) -> float:
    """L2 distance between two scattering representations.

    Concatenates all orders into a single feature vector and computes
    Euclidean distance.  This is a natural metric in scattering space
    because the transform is Lipschitz-continuous (Mallat 2012,
    Theorem 2.12).

    Parameters
    ----------
    coeff_a, coeff_b : ScatteringCoefficients
        Two scattering representations to compare.

    Returns
    -------
    float
        Euclidean distance.
    """

    def _flatten(c: ScatteringCoefficients) -> NDArray:
        parts = [c.order_0.ravel(), c.order_1.ravel()]
        if c.order_2.size > 0:
            parts.append(c.order_2.ravel())
        return np.concatenate(parts)

    va = _flatten(coeff_a)
    vb = _flatten(coeff_b)

    # Truncate to the shorter if lengths differ (e.g. different N)
    min_len = min(len(va), len(vb))
    return float(np.linalg.norm(va[:min_len] - vb[:min_len]))


def time_shift_invariance_test(
    signal: NDArray[np.floating],
    shift: int,
    J: int = 8,
    Q: int = 8,
    sr: float = 22050.0,
) -> dict:
    """Measure how invariant the scattering representation is to time shifts.

    The scattering transform is designed to be invariant to translations
    up to ``2**J`` samples (Mallat 2012, Proposition 2.5).  This
    function computes the representation of the original and shifted
    signals and reports the relative change.

    Parameters
    ----------
    signal : NDArray
        Input waveform (1-D).
    shift : int
        Number of samples to circularly shift.
    J, Q, sr
        Scattering parameters forwarded to :func:`compute_scattering`.

    Returns
    -------
    dict
        Keys: ``shift``, ``invariance_scale`` (= ``2**J``),
        ``distance``, ``relative_distance``, ``is_within_invariance_scale``.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    shifted = np.roll(signal, shift)

    sc_orig = compute_scattering(signal, J=J, Q=Q, sr=sr)
    sc_shift = compute_scattering(shifted, J=J, Q=Q, sr=sr)

    dist = scattering_distance(sc_orig, sc_shift)

    # Norm of the original representation (distance to the zero vector)
    zero_coeffs = ScatteringCoefficients(
        order_0=np.zeros_like(sc_orig.order_0),
        order_1=np.zeros_like(sc_orig.order_1),
        order_2=np.zeros_like(sc_orig.order_2),
    )
    orig_norm = scattering_distance(sc_orig, zero_coeffs)
    rel = dist / orig_norm if orig_norm > 0.0 else 0.0

    return {
        "shift": shift,
        "invariance_scale": 2 ** J,
        "distance": dist,
        "relative_distance": rel,
        "is_within_invariance_scale": abs(shift) < 2 ** J,
    }


# ---------------------------------------------------------------------------
# Modulation spectrum
# ---------------------------------------------------------------------------

def modulation_spectrum(
    first_order_coeffs: NDArray[np.floating],
    sr: float = 22050.0,
) -> dict:
    """Compute modulation spectrum from first-order scattering coefficients.

    The modulation spectrum reveals temporal modulation rates in each
    frequency band --- musically important for characterising vibrato
    (~5--8 Hz), tremolo (~3--6 Hz), and syllabic rhythm (~2--4 Hz).
    See Andén & Mallat (2014), section 3.2.

    Parameters
    ----------
    first_order_coeffs : NDArray
        Shape ``(n_bands, T)`` --- output of :func:`scattering_order_1`.
    sr : float
        Sample rate in Hz (used to label modulation frequencies).

    Returns
    -------
    dict
        ``modulation_rates`` --- frequency axis in Hz.
        ``modulation_energy`` --- shape ``(n_bands, n_freq_bins)``
            power spectrum per frequency band.
        ``mean_modulation_rate`` --- energy-weighted mean rate per band.
        ``dominant_modulation_rate`` --- peak rate per band.
    """
    first_order_coeffs = np.atleast_2d(first_order_coeffs)
    n_bands, T = first_order_coeffs.shape

    # Modulation spectrum = |FFT of each band's temporal envelope|^2
    mod_fft = np.fft.rfft(first_order_coeffs, axis=1)
    mod_energy = np.abs(mod_fft) ** 2

    mod_freqs = np.fft.rfftfreq(T, d=1.0 / sr)

    # Per-band summaries
    mean_rates = np.zeros(n_bands, dtype=np.float64)
    dominant_rates = np.zeros(n_bands, dtype=np.float64)
    for b in range(n_bands):
        total_e = mod_energy[b].sum()
        if total_e > 0:
            mean_rates[b] = float(np.sum(mod_freqs * mod_energy[b]) / total_e)
            dominant_rates[b] = float(mod_freqs[np.argmax(mod_energy[b])])

    return {
        "modulation_rates": mod_freqs,
        "modulation_energy": mod_energy,
        "mean_modulation_rate": mean_rates,
        "dominant_modulation_rate": dominant_rates,
    }


# ---------------------------------------------------------------------------
# Joint time-frequency scattering
# ---------------------------------------------------------------------------

def joint_time_frequency_scattering(
    signal: NDArray[np.floating],
    J_time: int = 8,
    J_freq: int = 4,
    Q: int = 8,
) -> dict:
    """Joint time--frequency scattering for AM/FM analysis.

    Extends the standard scattering by applying a second wavelet
    transform along the *frequency* axis of the first-order scalogram,
    capturing joint spectro-temporal modulations such as frequency
    glides (FM) and spectral flux patterns.

    This follows the "joint scattering" extension described in
    Andén, Lostanlen & Mallat (2015).

    Parameters
    ----------
    signal : NDArray
        Input waveform (1-D).
    J_time : int
        Octaves for the temporal wavelet bank.
    J_freq : int
        Octaves for the frequential wavelet bank (applied along the
        frequency axis of S1).
    Q : int
        Wavelets per octave (shared by both banks).

    Returns
    -------
    dict
        ``S1`` --- first-order coefficients, shape ``(n_bands, T)``.
        ``S_freq`` --- frequency-scattered coefficients,
            shape ``(n_freq_paths, T)``.
        ``S_joint`` --- joint (time + freq) scattered coefficients,
            shape ``(n_joint_paths, T)``.
        ``meta`` --- auxiliary info (J_time, J_freq, Q, counts).
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    N = len(signal)

    # --- Temporal scattering (order 1) ---
    psi_time, phi_time = build_wavelet_filterbank(J_time, Q, N)
    S1 = scattering_order_1(signal, psi_time, phi_time)
    n_bands = S1.shape[0]

    empty_result = {
        "S1": S1,
        "S_freq": np.zeros((0, N), dtype=np.float64),
        "S_joint": np.zeros((0, N), dtype=np.float64),
        "meta": {
            "J_time": J_time,
            "J_freq": J_freq,
            "Q": Q,
            "n_bands": n_bands,
            "n_freq_paths": 0,
            "n_joint_paths": 0,
        },
    }

    if n_bands < 2:
        return empty_result

    # --- Frequential scattering along the frequency axis ---
    J_freq_eff = min(J_freq, int(np.floor(np.log2(max(n_bands, 2)))))
    if J_freq_eff < 1:
        J_freq_eff = 1
    Q_freq = max(1, Q // 2)
    psi_freq, phi_freq = build_wavelet_filterbank(J_freq_eff, Q_freq, n_bands)

    # Apply frequential wavelets column-wise to each time frame
    S_freq_rows: list[NDArray[np.floating]] = []
    for psi_f in psi_freq:
        # For each time step, convolve S1[:, t] with psi_f along freq
        band_filtered = np.zeros_like(S1)
        for t_idx in range(N):
            col = S1[:, t_idx]
            col_hat = np.fft.fft(col, n=n_bands)
            band_filtered[:, t_idx] = _fft_convolve(col_hat, psi_f)

        # Average across frequency with phi_freq -> one scalar per time
        avg = np.zeros(N, dtype=np.float64)
        for t_idx in range(N):
            col = band_filtered[:, t_idx]
            col_hat = np.fft.fft(col, n=n_bands)
            avg[t_idx] = np.mean(_fft_convolve(col_hat, phi_freq))
        S_freq_rows.append(avg)

    if not S_freq_rows:
        return empty_result

    S_freq = np.stack(S_freq_rows, axis=0)

    # --- Joint scattering: temporal scatter of the freq-scattered rows ---
    # Apply a subset of temporal wavelets to each freq-scattered row
    n_temporal_wavelets = min(len(psi_time), J_time * 2)
    S_joint_rows: list[NDArray[np.floating]] = []
    for row in S_freq_rows:
        row_hat = np.fft.fft(row, n=N)
        for psi_t in psi_time[:n_temporal_wavelets]:
            u = _fft_convolve(row_hat, psi_t)
            u_hat = np.fft.fft(u, n=N)
            s = _fft_convolve(u_hat, phi_time)
            S_joint_rows.append(s)

    S_joint = (
        np.stack(S_joint_rows, axis=0)
        if S_joint_rows
        else np.zeros((0, N), dtype=np.float64)
    )

    return {
        "S1": S1,
        "S_freq": S_freq,
        "S_joint": S_joint,
        "meta": {
            "J_time": J_time,
            "J_freq": J_freq,
            "Q": Q,
            "n_bands": n_bands,
            "n_freq_paths": S_freq.shape[0],
            "n_joint_paths": S_joint.shape[0],
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ScatteringCoefficients",
    "FilterBankParams",
    "morlet_wavelet",
    "build_wavelet_filterbank",
    "scattering_order_0",
    "scattering_order_1",
    "scattering_order_2",
    "compute_scattering",
    "scattering_distance",
    "time_shift_invariance_test",
    "modulation_spectrum",
    "joint_time_frequency_scattering",
]
