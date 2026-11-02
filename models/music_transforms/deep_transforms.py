"""Deep music transforms — scattering, NMF, EMD, synchrosqueezing, cochleagram."""
from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# §1  Scattering transform (Mallat, 2012)
# ---------------------------------------------------------------------------

def _morlet_wavelet(n: int, scale: float, omega0: float = 5.0) -> np.ndarray:
    t = np.arange(n) - n / 2
    t_scaled = t / scale
    return (np.pi ** -0.25) * np.exp(1j * omega0 * t_scaled) * np.exp(-0.5 * t_scaled ** 2) / np.sqrt(scale)


def scattering_first_order(x: np.ndarray, scales: np.ndarray,
                           omega0: float = 5.0) -> np.ndarray:
    """First-order scattering coefficients |x * ψ_λ| * φ."""
    n = len(x)
    S1 = np.zeros((len(scales), n))
    for i, s in enumerate(scales):
        psi = _morlet_wavelet(n, s, omega0)
        conv = np.fft.ifft(np.fft.fft(x) * np.fft.fft(psi.real))
        S1[i] = np.abs(conv)
    return S1


def scattering_second_order(S1: np.ndarray, scales: np.ndarray,
                            omega0: float = 5.0) -> np.ndarray:
    """Second-order: ||x * ψ_λ1| * ψ_λ2| * φ."""
    n_scales1, n = S1.shape
    n_scales2 = len(scales)
    S2 = np.zeros((n_scales1, n_scales2, n))
    for i in range(n_scales1):
        for j, s in enumerate(scales):
            psi = _morlet_wavelet(n, s, omega0)
            conv = np.fft.ifft(np.fft.fft(S1[i]) * np.fft.fft(psi.real))
            S2[i, j] = np.abs(conv)
    return S2


def scattering_coefficients(x: np.ndarray, J: int = 6,
                            Q: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Compute first and second order scattering coefficients."""
    scales = 2.0 ** (np.arange(J * Q) / Q + 1)
    S1 = scattering_first_order(x, scales)
    S2 = scattering_second_order(S1[:J], scales[:J * Q // 2])
    return S1, S2

# ---------------------------------------------------------------------------
# §2  Modulation spectrogram (Chi et al., 2005)
# ---------------------------------------------------------------------------

def modulation_spectrogram(spectrogram: np.ndarray,
                           sr_mod: float = 1.0) -> np.ndarray:
    """Spectrotemporal modulation analysis: FFT along time axis per freq bin."""
    n_freq, n_time = spectrogram.shape
    mod_spec = np.zeros_like(spectrogram)
    for i in range(n_freq):
        mod_spec[i] = np.abs(np.fft.rfft(spectrogram[i], n=n_time))[:n_time]
    return mod_spec


def am_envelope(x: np.ndarray) -> np.ndarray:
    """Amplitude modulation envelope via analytic signal."""
    analytic = sp_signal.hilbert(x)
    return np.abs(analytic)


def fm_instantaneous_freq(x: np.ndarray, sr: float = 1.0) -> np.ndarray:
    """Instantaneous frequency from analytic signal phase."""
    analytic = sp_signal.hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    freq = np.diff(phase) / (2.0 * np.pi) * sr
    return np.append(freq, freq[-1] if len(freq) > 0 else 0)

# ---------------------------------------------------------------------------
# §3  NMF (Lee & Seung, 2001)
# ---------------------------------------------------------------------------

def nmf_multiplicative_update(V: np.ndarray, n_components: int,
                              n_iter: int = 200,
                              eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """NMF with multiplicative update rules: V ≈ W·H, W,H ≥ 0."""
    m, n = V.shape
    rng = np.random.default_rng(42)
    W = rng.random((m, n_components)) + eps
    H = rng.random((n_components, n)) + eps
    V_pos = np.maximum(V, 0)
    for _ in range(n_iter):
        # update H
        WtV = W.T @ V_pos
        WtWH = W.T @ W @ H + eps
        H *= WtV / WtWH
        # update W
        VHt = V_pos @ H.T
        WHHt = W @ (H @ H.T) + eps
        W *= VHt / WHHt
    return W, H


def nmf_reconstruct(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    return W @ H


def nmf_reconstruction_error(V: np.ndarray, W: np.ndarray,
                             H: np.ndarray) -> float:
    return float(np.linalg.norm(V - W @ H, 'fro'))

# ---------------------------------------------------------------------------
# §4  Synchrosqueezing (Daubechies et al., 2011)
# ---------------------------------------------------------------------------

def synchrosqueezed_cwt(x: np.ndarray, scales: np.ndarray,
                        omega0: float = 5.0,
                        threshold: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Synchrosqueezed CWT: reassign frequency via instantaneous frequency."""
    n = len(x)
    W = np.zeros((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        psi = _morlet_wavelet(n, s, omega0)
        W[i] = np.fft.ifft(np.fft.fft(x) * np.fft.fft(psi))
    # instantaneous frequency
    dW = np.diff(W, axis=1)
    dW = np.hstack([dW, dW[:, -1:]])
    omega = np.zeros_like(W, dtype=float)
    mask = np.abs(W) > threshold
    omega[mask] = np.imag(dW[mask] / (W[mask] + 1e-30)) / (2 * np.pi)
    # reassign
    freq_bins = omega0 / (2 * np.pi * scales)
    ssq = np.zeros_like(W, dtype=float)
    for i in range(len(scales)):
        for t in range(n):
            if mask[i, t]:
                target = np.argmin(np.abs(freq_bins - abs(omega[i, t])))
                ssq[target, t] += np.abs(W[i, t]) ** 2
    return ssq, omega

# ---------------------------------------------------------------------------
# §5  Empirical mode decomposition (Huang et al., 1998)
# ---------------------------------------------------------------------------

def _envelope(x: np.ndarray, kind: str = "upper") -> np.ndarray:
    """Cubic spline envelope through local extrema."""
    n = len(x)
    t = np.arange(n)
    if kind == "upper":
        idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    else:
        idx = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
    if len(idx) < 2:
        return np.full(n, x.mean())
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(idx, x[idx], bc_type='natural')
    return cs(t)


def sift(x: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Extract one IMF via sifting."""
    h = x.copy()
    for _ in range(max_iter):
        upper = _envelope(h, "upper")
        lower = _envelope(h, "lower")
        m = (upper + lower) / 2.0
        h_new = h - m
        sd = np.sum((h_new - h) ** 2) / (np.sum(h ** 2) + 1e-30)
        h = h_new
        if sd < tol:
            break
    return h


def emd(x: np.ndarray, max_imfs: int = 8) -> List[np.ndarray]:
    """Empirical mode decomposition into intrinsic mode functions."""
    imfs = []
    residual = x.copy()
    for _ in range(max_imfs):
        if np.std(residual) < 1e-10:
            break
        imf = sift(residual)
        imfs.append(imf)
        residual = residual - imf
    imfs.append(residual)
    return imfs


def hilbert_spectrum(imfs: List[np.ndarray],
                     sr: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Hilbert spectrum: instantaneous frequency and amplitude for each IMF."""
    freqs = []
    amps = []
    for imf in imfs:
        analytic = sp_signal.hilbert(imf)
        amps.append(np.abs(analytic))
        phase = np.unwrap(np.angle(analytic))
        f = np.diff(phase) / (2 * np.pi) * sr
        freqs.append(np.append(f, f[-1] if len(f) > 0 else 0))
    return np.array(freqs), np.array(amps)

# ---------------------------------------------------------------------------
# §6  MFCC pipeline
# ---------------------------------------------------------------------------

def mel_filterbank(n_filters: int, n_fft: int, sr: float,
                   fmin: float = 0.0, fmax: Optional[float] = None
                   ) -> np.ndarray:
    """Triangular mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_filters + 2)
    freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        for j in range(bins[i], bins[i + 1]):
            if j < fb.shape[1]:
                fb[i, j] = (j - bins[i]) / max(bins[i + 1] - bins[i], 1)
        for j in range(bins[i + 1], bins[i + 2]):
            if j < fb.shape[1]:
                fb[i, j] = (bins[i + 2] - j) / max(bins[i + 2] - bins[i + 1], 1)
    return fb


def mfcc(x: np.ndarray, sr: float = 22050.0, n_mfcc: int = 13,
         n_fft: int = 2048, hop: int = 512,
         n_mels: int = 40) -> np.ndarray:
    """Full MFCC pipeline: STFT → mel → log → DCT → lifter."""
    # STFT
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames <= 0:
        return np.zeros((n_mfcc, 1))
    frames = np.stack([x[i * hop:i * hop + n_fft] *
                       np.hanning(n_fft) for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    # mel
    fb = mel_filterbank(n_mels, n_fft, sr)
    mel_spec = spec @ fb.T
    log_mel = np.log(mel_spec + 1e-10)
    # DCT-II
    from scipy.fft import dct
    coeffs = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    # liftering
    lift = 1 + (n_mfcc / 2.0) * np.sin(np.pi * np.arange(n_mfcc) / n_mfcc)
    return (coeffs * lift).T


def mfcc_delta(mfcc_coeffs: np.ndarray, order: int = 1) -> np.ndarray:
    """Delta (velocity) or delta-delta (acceleration) of MFCCs."""
    result = mfcc_coeffs
    for _ in range(order):
        padded = np.pad(result, ((0, 0), (1, 1)), mode='edge')
        result = (padded[:, 2:] - padded[:, :-2]) / 2.0
    return result

# ---------------------------------------------------------------------------
# §7  Constant-Q transform (Brown, 1991)
# ---------------------------------------------------------------------------

def cqt_kernel(sr: float, fmin: float, n_bins: int,
               bins_per_octave: int = 12) -> List[np.ndarray]:
    """CQT analysis kernels (frequency-domain)."""
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)
    kernels = []
    for f in freqs:
        N_k = int(np.ceil(Q * sr / f))
        t = np.arange(N_k)
        win = np.hanning(N_k)
        kernel = win * np.exp(-2j * np.pi * f * t / sr) / N_k
        kernels.append(kernel)
    return kernels


def cqt(x: np.ndarray, sr: float = 22050.0, fmin: float = 32.7,
        n_bins: int = 84, bins_per_octave: int = 12,
        hop: int = 512) -> np.ndarray:
    """Constant-Q transform via direct convolution."""
    kernels = cqt_kernel(sr, fmin, n_bins, bins_per_octave)
    n_frames = max(1, 1 + (len(x) - max(len(k) for k in kernels)) // hop)
    C = np.zeros((n_bins, n_frames), dtype=complex)
    for i, kernel in enumerate(kernels):
        nk = len(kernel)
        for t in range(n_frames):
            start = t * hop
            end = start + nk
            if end <= len(x):
                C[i, t] = np.dot(x[start:end], kernel)
    return np.abs(C)

# ---------------------------------------------------------------------------
# §8  Cochleagram (Lyon/Slaney, 1988)
# ---------------------------------------------------------------------------

def cochleagram(x: np.ndarray, sr: float = 22050.0,
                n_channels: int = 64, hop: int = 256) -> np.ndarray:
    """Simplified cochleagram via gammatone-like bandpass cascade."""
    fmin, fmax = 50.0, sr / 2 - 100
    cf = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_channels))
    n_frames = max(1, len(x) // hop)
    coch = np.zeros((n_channels, n_frames))
    for i, f in enumerate(cf):
        bw = 1.019 * (24.7 * (4.37 * f / 1000.0 + 1.0))
        Q = f / max(bw, 1.0)
        b, a = sp_signal.iirpeak(f / (sr / 2), Q)
        filtered = sp_signal.lfilter(b, a, x)
        env = np.abs(sp_signal.hilbert(filtered))
        for t in range(n_frames):
            start = t * hop
            end = min(start + hop, len(env))
            if start < len(env):
                coch[i, t] = np.mean(env[start:end])
    return coch

# ---------------------------------------------------------------------------
# §9  Chromagram variants (HPCP, CRP, CENS)
# ---------------------------------------------------------------------------

def hpcp(spectrogram: np.ndarray, sr: float, n_fft: int,
         ref_freq: float = 440.0) -> np.ndarray:
    """Harmonic Pitch Class Profile (Gómez, 2006)."""
    n_bins, n_frames = spectrogram.shape
    freqs = np.arange(n_bins) * sr / n_fft
    chroma = np.zeros((12, n_frames))
    for i, f in enumerate(freqs):
        if f < 50 or f > 5000:
            continue
        pc = int(np.round(12 * np.log2(f / ref_freq + 1e-30))) % 12
        chroma[pc] += spectrogram[i]
    # normalise per frame
    norms = chroma.max(axis=0, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return chroma / norms


def cens(chroma: np.ndarray, win: int = 41) -> np.ndarray:
    """Chroma Energy Normalised Statistics (Müller & Ewert, 2011)."""
    # L1 normalise
    norms = chroma.sum(axis=0, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = chroma / norms
    # quantise
    thresholds = [0.05, 0.1, 0.2, 0.4, 1.0]
    quantised = np.zeros_like(normed)
    for i, th in enumerate(thresholds):
        quantised += (normed >= th).astype(float)
    quantised /= len(thresholds)
    # smooth
    kernel = np.hanning(win)
    kernel /= kernel.sum()
    smoothed = np.zeros_like(quantised)
    for i in range(12):
        smoothed[i] = np.convolve(quantised[i], kernel, mode='same')
    return smoothed

# ---------------------------------------------------------------------------
# §10  Pipeline
# ---------------------------------------------------------------------------

@dataclass
class DeepTransformAnalysis:
    mfcc_coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
    mfcc_delta1: np.ndarray = field(default_factory=lambda: np.array([]))
    cqt_mag: np.ndarray = field(default_factory=lambda: np.array([]))
    nmf_W: np.ndarray = field(default_factory=lambda: np.array([]))
    nmf_H: np.ndarray = field(default_factory=lambda: np.array([]))
    nmf_error: float = 0.0
    n_imfs: int = 0
    scattering_energy: float = 0.0
    cochleagram_energy: float = 0.0


def analyze_transforms_deep(x: np.ndarray, sr: float = 22050.0
                            ) -> DeepTransformAnalysis:
    """Full deep transform analysis pipeline."""
    mc = mfcc(x, sr)
    md1 = mfcc_delta(mc, order=1)
    cq = cqt(x, sr, n_bins=48, hop=512)

    # NMF on magnitude spectrogram
    n_fft = 2048
    hop = 512
    n_frames = max(1, 1 + (len(x) - n_fft) // hop)
    if n_frames > 1:
        frames = np.stack([x[i * hop:i * hop + n_fft] *
                           np.hanning(n_fft) for i in range(n_frames)])
        spec = np.abs(np.fft.rfft(frames, axis=1)).T
        W, H = nmf_multiplicative_update(spec, n_components=4, n_iter=50)
        err = nmf_reconstruction_error(spec, W, H)
    else:
        W = H = np.array([])
        err = 0.0

    # EMD
    if len(x) > 100:
        imfs = emd(x[:min(len(x), 4096)], max_imfs=5)
    else:
        imfs = [x]

    # scattering (on short segment for efficiency)
    seg = x[:min(len(x), 2048)]
    scales = 2.0 ** (np.arange(12) / 4 + 1)
    S1 = scattering_first_order(seg, scales)
    scat_energy = float(np.sum(S1 ** 2))

    coch = cochleagram(x, sr, n_channels=32, hop=512)
    coch_energy = float(np.sum(coch ** 2))

    return DeepTransformAnalysis(
        mfcc_coeffs=mc,
        mfcc_delta1=md1,
        cqt_mag=cq,
        nmf_W=W,
        nmf_H=H,
        nmf_error=err,
        n_imfs=len(imfs),
        scattering_energy=scat_energy,
        cochleagram_energy=coch_energy,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    sr = 22050.0
    t = np.arange(int(sr * 0.5)) / sr
    x = np.sin(2 * np.pi * 440 * t) + 0.3 * rng.standard_normal(len(t))
    result = analyze_transforms_deep(x, sr)
    assert result.mfcc_coeffs.shape[0] == 13
    assert result.n_imfs >= 2
    assert result.scattering_energy > 0
    print(f"MFCC shape: {result.mfcc_coeffs.shape}")
    print(f"CQT shape: {result.cqt_mag.shape}")
    print(f"NMF error: {result.nmf_error:.2f}")
    print(f"IMFs: {result.n_imfs}")
    print(f"Scattering energy: {result.scattering_energy:.2f}")
    print("deep_transforms OK")
