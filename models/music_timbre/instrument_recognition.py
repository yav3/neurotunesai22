"""Instrument recognition: feature extraction (MFCCs, spectral shape, attack),
template-based and Mahalanobis classification.  Pure numpy."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class InstrumentClass(str, Enum):
    PIANO = "piano"
    GUITAR = "guitar"
    VIOLIN = "violin"
    CELLO = "cello"
    FLUTE = "flute"
    CLARINET = "clarinet"
    TRUMPET = "trumpet"
    DRUM_KIT = "drum_kit"
    BASS = "bass"
    VOICE = "voice"
    ORGAN = "organ"
    SAXOPHONE = "saxophone"


@dataclass
class InstrumentFeatures:
    spectral_centroid: float = 0.0
    spectral_spread: float = 0.0
    spectral_flux: float = 0.0
    zero_crossing_rate: float = 0.0
    mfcc: np.ndarray = field(default_factory=lambda: np.zeros(13))
    attack_time_ms: float = 0.0
    harmonic_ratio: float = 0.0
    temporal_centroid: float = 0.5


def _spectral_centroid(S: np.ndarray, freqs: np.ndarray) -> float:
    total = S.sum()
    if total < 1e-12:
        return 0.0
    return float(np.dot(freqs, S) / total)


def _spectral_spread(S: np.ndarray, freqs: np.ndarray, centroid: float) -> float:
    total = S.sum()
    if total < 1e-12:
        return 0.0
    return float(np.sqrt(np.dot((freqs - centroid) ** 2, S) / total))


def _zero_crossing_rate(signal: np.ndarray) -> float:
    if len(signal) < 2:
        return 0.0
    signs = np.sign(signal)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / len(signal))


def _attack_time(signal: np.ndarray, sr: float) -> float:
    env = np.abs(signal)
    peak_idx = np.argmax(env)
    if peak_idx == 0:
        return 0.0
    threshold = 0.1 * env[peak_idx]
    onset_idx = 0
    for i in range(peak_idx):
        if env[i] >= threshold:
            onset_idx = i
            break
    return float((peak_idx - onset_idx) / sr * 1000.0)


def _harmonic_ratio(signal: np.ndarray, sr: float) -> float:
    n = len(signal)
    if n < 64:
        return 0.0
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[n - 1:]
    autocorr = autocorr / (autocorr[0] + 1e-12)
    min_lag = int(sr / 4000)
    max_lag = min(int(sr / 50), len(autocorr) - 1)
    if min_lag >= max_lag:
        return 0.0
    segment = autocorr[min_lag:max_lag]
    if len(segment) == 0:
        return 0.0
    return float(np.max(segment))


def _mfcc(signal: np.ndarray, sr: float, n_mfcc: int = 13,
          n_fft: int = 2048, n_mels: int = 40) -> np.ndarray:
    win = np.hanning(min(n_fft, len(signal)))
    padded = np.zeros(n_fft)
    padded[: len(win)] = signal[: len(win)] * win
    spectrum = np.abs(np.fft.rfft(padded))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    mel_lo = 2595.0 * np.log10(1 + 0 / 700.0)
    mel_hi = 2595.0 * np.log10(1 + sr / 2 / 700.0)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, len(spectrum)))
    for m in range(n_mels):
        lo, c, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        for k in range(lo, c):
            if c > lo:
                fb[m, k] = (k - lo) / (c - lo)
        for k in range(c, hi):
            if hi > c:
                fb[m, k] = (hi - k) / (hi - c)

    mel_energy = fb @ spectrum
    log_mel = np.log(mel_energy + 1e-10)
    from scipy.fft import dct
    ceps = dct(log_mel, type=2, norm="ortho")
    return ceps[:n_mfcc]


def _temporal_centroid(signal: np.ndarray) -> float:
    env = np.abs(signal)
    total = env.sum()
    if total < 1e-12:
        return 0.5
    t = np.arange(len(env)) / max(len(env) - 1, 1)
    return float(np.dot(t, env) / total)


def extract_features(signal: np.ndarray, sr: float) -> InstrumentFeatures:
    n_fft = min(2048, len(signal))
    win = np.hanning(n_fft)
    padded = np.zeros(n_fft)
    padded[: min(len(signal), n_fft)] = signal[: min(len(signal), n_fft)] * win[: min(len(signal), n_fft)]
    S = np.abs(np.fft.rfft(padded))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    centroid = _spectral_centroid(S, freqs)
    spread = _spectral_spread(S, freqs, centroid)
    zcr = _zero_crossing_rate(signal)
    attack = _attack_time(signal, sr)
    hr = _harmonic_ratio(signal, sr)
    mfcc_vals = _mfcc(signal, sr)
    tc = _temporal_centroid(signal)

    flux_val = 0.0
    if len(signal) > n_fft:
        S2 = np.abs(np.fft.rfft(signal[n_fft // 2 : n_fft // 2 + n_fft] * win))
        flux_val = float(np.sqrt(np.sum((S2 - S) ** 2)))

    return InstrumentFeatures(
        spectral_centroid=centroid, spectral_spread=spread,
        spectral_flux=flux_val, zero_crossing_rate=zcr,
        mfcc=mfcc_vals, attack_time_ms=attack,
        harmonic_ratio=hr, temporal_centroid=tc,
    )


def feature_vector(feats: InstrumentFeatures) -> np.ndarray:
    scalars = [feats.spectral_centroid, feats.spectral_spread,
               feats.spectral_flux, feats.zero_crossing_rate,
               feats.attack_time_ms, feats.harmonic_ratio, feats.temporal_centroid]
    return np.concatenate([scalars, feats.mfcc])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


INSTRUMENT_TEMPLATES: Dict[InstrumentClass, InstrumentFeatures] = {
    InstrumentClass.PIANO: InstrumentFeatures(2000, 1500, 0.3, 0.05, np.zeros(13), 10, 0.7, 0.3),
    InstrumentClass.GUITAR: InstrumentFeatures(1800, 1200, 0.2, 0.04, np.zeros(13), 15, 0.65, 0.35),
    InstrumentClass.VIOLIN: InstrumentFeatures(3000, 2000, 0.15, 0.06, np.zeros(13), 30, 0.85, 0.45),
    InstrumentClass.CELLO: InstrumentFeatures(1500, 1000, 0.12, 0.03, np.zeros(13), 40, 0.80, 0.45),
    InstrumentClass.FLUTE: InstrumentFeatures(4000, 1800, 0.10, 0.02, np.zeros(13), 25, 0.60, 0.50),
    InstrumentClass.CLARINET: InstrumentFeatures(2500, 1600, 0.08, 0.03, np.zeros(13), 35, 0.75, 0.48),
    InstrumentClass.TRUMPET: InstrumentFeatures(3500, 2200, 0.25, 0.04, np.zeros(13), 20, 0.70, 0.40),
    InstrumentClass.DRUM_KIT: InstrumentFeatures(5000, 4000, 0.80, 0.15, np.zeros(13), 5, 0.10, 0.20),
    InstrumentClass.BASS: InstrumentFeatures(800, 500, 0.15, 0.02, np.zeros(13), 20, 0.55, 0.40),
    InstrumentClass.VOICE: InstrumentFeatures(2200, 1400, 0.20, 0.08, np.zeros(13), 50, 0.90, 0.50),
    InstrumentClass.ORGAN: InstrumentFeatures(1800, 1500, 0.05, 0.01, np.zeros(13), 60, 0.80, 0.55),
    InstrumentClass.SAXOPHONE: InstrumentFeatures(2800, 1800, 0.18, 0.05, np.zeros(13), 30, 0.75, 0.42),
}


def classify_nearest(
    feats: InstrumentFeatures,
    templates: Optional[Dict[InstrumentClass, InstrumentFeatures]] = None,
) -> Tuple[InstrumentClass, float]:
    tpls = templates or INSTRUMENT_TEMPLATES
    fv = feature_vector(feats)
    best_cls = InstrumentClass.PIANO
    best_sim = -1.0
    for cls, tpl in tpls.items():
        tv = feature_vector(tpl)
        sim = cosine_similarity(fv, tv)
        if sim > best_sim:
            best_sim = sim
            best_cls = cls
    return best_cls, best_sim


def classify_mahalanobis(
    feats: InstrumentFeatures,
    means: Dict[InstrumentClass, np.ndarray],
    covs: Dict[InstrumentClass, np.ndarray],
) -> Tuple[InstrumentClass, float]:
    fv = feature_vector(feats)
    best_cls = InstrumentClass.PIANO
    best_dist = np.inf
    for cls in means:
        diff = fv - means[cls]
        try:
            inv_cov = np.linalg.inv(covs[cls])
        except np.linalg.LinAlgError:
            inv_cov = np.eye(len(diff))
        d = float(np.sqrt(diff @ inv_cov @ diff))
        if d < best_dist:
            best_dist = d
            best_cls = cls
    confidence = 1.0 / (1.0 + best_dist)
    return best_cls, confidence


def temporal_class_profile(
    signals: List[np.ndarray], sr: float,
) -> Dict[InstrumentClass, float]:
    if not signals:
        return {}
    counts: Dict[InstrumentClass, int] = {}
    for sig in signals:
        feats = extract_features(sig, sr)
        cls, _ = classify_nearest(feats)
        counts[cls] = counts.get(cls, 0) + 1
    total = len(signals)
    return {cls: cnt / total for cls, cnt in counts.items()}


def feature_importance(
    feats_list: List[InstrumentFeatures], labels: List[InstrumentClass],
) -> Dict[str, float]:
    if len(feats_list) < 2:
        return {}
    vectors = np.array([feature_vector(f) for f in feats_list])
    label_set = list(set(labels))
    label_idx = np.array([label_set.index(l) for l in labels])
    grand_mean = vectors.mean(axis=0)
    n_features = vectors.shape[1]
    importance: Dict[str, float] = {}
    names = ["centroid", "spread", "flux", "zcr", "attack", "harmonic_ratio", "temporal_centroid"] + \
            [f"mfcc_{i}" for i in range(13)]
    for d in range(n_features):
        between = 0.0
        within = 0.0
        for c in range(len(label_set)):
            mask = label_idx == c
            if mask.sum() == 0:
                continue
            class_mean = vectors[mask, d].mean()
            between += mask.sum() * (class_mean - grand_mean[d]) ** 2
            within += np.sum((vectors[mask, d] - class_mean) ** 2)
        ratio = between / (within + 1e-12)
        fname = names[d] if d < len(names) else f"dim_{d}"
        importance[fname] = float(ratio)
    return importance
