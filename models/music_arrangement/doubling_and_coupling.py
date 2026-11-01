"""Orchestral doubling theory and instrument coupling analysis.

Spectral-domain analysis of instrument doublings, masking risk,
blend quality, and coupling strength.  Grounded in the orchestration
principles of Rimsky-Korsakov (*Principles of Orchestration*) and
Samuel Adler (*The Study of Orchestration*):

* Effective doublings reinforce the harmonic series without introducing
  roughness from critical-band interference (Adler ch. 26).
* Octave couplings strengthen the fundamental while preserving timbral
  identity; unison doublings add power but risk beating when slightly
  detuned (Rimsky-Korsakov, "Combination of instruments in unison").
* Tutti density must be managed so that inner voices remain audible
  and the registral spread does not collapse into a mass of sound
  (Rimsky-Korsakov, "The full orchestra").
* Section balance follows the inverse-square relationship between
  perceived loudness and dynamic range (Adler, "Scoring for full
  orchestra").

Pure numpy/scipy — no ML frameworks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DoublingPair:
    """A pair of instruments sounding together.

    Attributes
    ----------
    instruments : tuple[str, str]
        Names of the two instruments in the doubling.
    interval : int
        Interval in semitones between the two written pitches
        (0 = unison, 12 = octave, etc.).
    blend_score : float
        Overall blend quality in [0, 1].
    """
    instruments: Tuple[str, str]
    interval: int
    blend_score: float


@dataclass
class CouplingProfile:
    """Full coupling analysis between two instruments.

    Attributes
    ----------
    instruments : tuple[str, str]
        Names of the two instruments.
    spectral_overlap : float
        Overlap coefficient in [0, 1].
    masking_risk : float
        Probability that one instrument masks the other, in [0, 1].
    blend_quality : float
        Inverse-roughness-weighted blend metric in [0, 1].
    """
    instruments: Tuple[str, str]
    spectral_overlap: float
    masking_risk: float
    blend_quality: float


# ---------------------------------------------------------------------------
# Psychoacoustic helpers
# ---------------------------------------------------------------------------

def _critical_bandwidth(freq: float) -> float:
    """Equivalent Rectangular Bandwidth (ERB) in Hz at *freq* Hz.

    Glasberg & Moore (1990) formula, the standard model for
    auditory critical-band width.
    """
    return 24.7 * (4.37e-3 * freq + 1.0)


def _bark(freq: float) -> float:
    """Convert Hz to Bark scale (Traunmuller 1990)."""
    return 13.0 * math.atan(0.00076 * freq) + 3.5 * math.atan((freq / 7500.0) ** 2)


def _roughness_pair(f1: float, a1: float, f2: float, a2: float) -> float:
    """Roughness contribution from a single partial pair.

    Sethares (1993) roughness model: roughness peaks when
    the frequency separation is ~25 % of the critical bandwidth
    and drops to near zero beyond one critical band.
    """
    s = 0.24 / (_critical_bandwidth(min(f1, f2)) * 0.25 + 1e-12)
    df = abs(f2 - f1)
    amp = a1 * a2
    return amp * (math.exp(-3.5 * s * df) - math.exp(-5.75 * s * df))


# ---------------------------------------------------------------------------
# Core spectral functions
# ---------------------------------------------------------------------------

def spectral_overlap(
    spec_a: NDArray[np.floating],
    spec_b: NDArray[np.floating],
    freqs: NDArray[np.floating],
) -> float:
    """Overlap coefficient between two instrument spectra.

    Uses the Bhattacharyya-style overlap:
        OVL = sum(sqrt(Pa * Pb)) / sqrt(sum(Pa) * sum(Pb))
    where Pa, Pb are the magnitude spectra treated as distributions.

    Parameters
    ----------
    spec_a, spec_b : array, shape (N,)
        Magnitude spectra (power or amplitude — both work, but
        should be the same type for both inputs).
    freqs : array, shape (N,)
        Corresponding frequency bins in Hz (used only for
        dimensional validation; the metric is bin-based).

    Returns
    -------
    float
        Overlap coefficient in [0, 1].  1 = identical spectral
        shapes, 0 = completely non-overlapping.
    """
    a = np.asarray(spec_a, dtype=np.float64).ravel()
    b = np.asarray(spec_b, dtype=np.float64).ravel()
    f = np.asarray(freqs, dtype=np.float64).ravel()

    if a.shape[0] != b.shape[0] or a.shape[0] != f.shape[0]:
        raise ValueError("spec_a, spec_b, and freqs must have the same length")

    # Ensure non-negative
    a = np.maximum(a, 0.0)
    b = np.maximum(b, 0.0)

    sa = a.sum()
    sb = b.sum()
    if sa < 1e-15 or sb < 1e-15:
        return 0.0

    return float(np.sum(np.sqrt(a * b)) / math.sqrt(sa * sb))


def masking_risk(
    spec_a: NDArray[np.floating],
    spec_b: NDArray[np.floating],
    freqs: NDArray[np.floating],
) -> float:
    """Probability that one instrument masks the other.

    Masking is assessed per critical band: if one spectrum exceeds
    the other by more than the *masking threshold* (roughly 6 dB,
    following Moore 2012) within a critical band, the weaker signal
    is deemed masked in that band.  The returned value is the
    energy-weighted fraction of masked bands.

    Parameters
    ----------
    spec_a, spec_b : array, shape (N,)
        Magnitude spectra.
    freqs : array, shape (N,)
        Frequency bins in Hz.

    Returns
    -------
    float
        Masking risk in [0, 1].  0 = no masking, 1 = complete masking.
    """
    a = np.asarray(spec_a, dtype=np.float64).ravel()
    b = np.asarray(spec_b, dtype=np.float64).ravel()
    f = np.asarray(freqs, dtype=np.float64).ravel()

    if a.shape[0] != b.shape[0] or a.shape[0] != f.shape[0]:
        raise ValueError("spec_a, spec_b, and freqs must have the same length")

    a = np.maximum(a, 0.0)
    b = np.maximum(b, 0.0)

    if a.sum() < 1e-15 or b.sum() < 1e-15:
        return 0.0

    masking_threshold_db = 6.0
    ratio_threshold = 10.0 ** (masking_threshold_db / 20.0)  # amplitude ratio

    # Walk through critical bands (Bark-scale partitioning)
    bark_lo = _bark(max(f[0], 20.0))
    bark_hi = _bark(min(f[-1], 20000.0))
    n_bands = max(int(bark_hi - bark_lo), 1)

    total_energy = 0.0
    masked_energy = 0.0

    for band_idx in range(n_bands):
        b_lo = bark_lo + band_idx * (bark_hi - bark_lo) / n_bands
        b_hi = bark_lo + (band_idx + 1) * (bark_hi - bark_lo) / n_bands
        bark_vals = np.array([_bark(max(fi, 20.0)) for fi in f])
        mask = (bark_vals >= b_lo) & (bark_vals < b_hi)
        if not np.any(mask):
            continue

        ea = np.sum(a[mask] ** 2)
        eb = np.sum(b[mask] ** 2)
        band_energy = ea + eb
        total_energy += band_energy

        # Check if one dominates the other beyond threshold
        if ea > 1e-15 and eb > 1e-15:
            amp_a = math.sqrt(ea)
            amp_b = math.sqrt(eb)
            ratio = max(amp_a, amp_b) / min(amp_a, amp_b)
            if ratio > ratio_threshold:
                masked_energy += min(ea, eb)

    if total_energy < 1e-15:
        return 0.0

    return float(np.clip(masked_energy / total_energy, 0.0, 1.0))


def blend_quality(
    spec_a: NDArray[np.floating],
    spec_b: NDArray[np.floating],
    freqs: NDArray[np.floating],
) -> float:
    """How well two instrument spectra blend together.

    Combines two perceptual factors (Adler, ch. 26):
    1. **Inverse roughness** — lower Sethares-model roughness means
       smoother blend.
    2. **Spectral centroid proximity** — instruments whose brightness
       (centroid) is close blend more naturally.

    Parameters
    ----------
    spec_a, spec_b : array, shape (N,)
        Magnitude spectra.
    freqs : array, shape (N,)
        Frequency bins in Hz.

    Returns
    -------
    float
        Blend quality in [0, 1].  Higher is better.
    """
    a = np.asarray(spec_a, dtype=np.float64).ravel()
    b = np.asarray(spec_b, dtype=np.float64).ravel()
    f = np.asarray(freqs, dtype=np.float64).ravel()

    if a.shape[0] != b.shape[0] or a.shape[0] != f.shape[0]:
        raise ValueError("spec_a, spec_b, and freqs must have the same length")

    a = np.maximum(a, 0.0)
    b = np.maximum(b, 0.0)

    sum_a = a.sum()
    sum_b = b.sum()
    if sum_a < 1e-15 or sum_b < 1e-15:
        return 0.0

    # --- Inverse roughness component ---
    # Use top-N partials as a proxy (full pairwise is O(N^2))
    n_partials = min(16, len(f))
    idx_a = np.argsort(a)[-n_partials:]
    idx_b = np.argsort(b)[-n_partials:]

    total_roughness = 0.0
    for ia in idx_a:
        for ib in idx_b:
            total_roughness += _roughness_pair(f[ia], a[ia], f[ib], b[ib])

    # Normalize by product of total amplitudes
    norm = sum_a * sum_b
    roughness_normalized = total_roughness / norm if norm > 1e-15 else 0.0
    # Map through a sigmoid so values near zero are "smooth"
    inverse_roughness = 1.0 / (1.0 + 10.0 * roughness_normalized)

    # --- Spectral centroid proximity ---
    centroid_a = float(np.dot(f, a) / sum_a)
    centroid_b = float(np.dot(f, b) / sum_b)
    # Normalize distance by the mean centroid (relative proximity)
    mean_centroid = (centroid_a + centroid_b) / 2.0
    if mean_centroid > 1e-6:
        centroid_dist = abs(centroid_a - centroid_b) / mean_centroid
    else:
        centroid_dist = 0.0
    centroid_proximity = math.exp(-2.0 * centroid_dist)

    # Weighted combination (roughness is the primary concern)
    return float(np.clip(0.6 * inverse_roughness + 0.4 * centroid_proximity, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Doubling and coupling metrics
# ---------------------------------------------------------------------------

def doubling_effectiveness(
    pitches: Sequence[float],
    instrument_a_formants: Sequence[float],
    instrument_b_formants: Sequence[float],
) -> float:
    """Score how effective a doubling is across a pitch range.

    Compares formant structures of two instruments at each pitch.
    A doubling is most effective when the instruments reinforce each
    other's formants without coinciding so closely that they create
    roughness (Rimsky-Korsakov: "Strengthening of tone-quality is
    only achieved when timbres are dissimilar enough to avoid beating
    but similar enough to fuse").

    Parameters
    ----------
    pitches : sequence of float
        MIDI pitch values to evaluate across.
    instrument_a_formants : sequence of float
        Formant center frequencies (Hz) of instrument A.
    instrument_b_formants : sequence of float
        Formant center frequencies (Hz) of instrument B.

    Returns
    -------
    float
        Effectiveness score in [0, 1].  Higher is better.
    """
    if not pitches or not instrument_a_formants or not instrument_b_formants:
        return 0.0

    formants_a = np.array(instrument_a_formants, dtype=np.float64)
    formants_b = np.array(instrument_b_formants, dtype=np.float64)

    scores: list[float] = []

    for midi_pitch in pitches:
        f0 = 440.0 * (2.0 ** ((float(midi_pitch) - 69.0) / 12.0))

        # Build harmonic series for this pitch
        n_harmonics = min(12, int(16000.0 / max(f0, 1.0)))
        harmonics = f0 * np.arange(1, n_harmonics + 1)

        # Score: reinforcement without roughness
        reinforcement = 0.0
        roughness_penalty = 0.0

        for fa in formants_a:
            for fb in formants_b:
                # Formants near the same harmonic reinforce each other
                for h in harmonics:
                    dist_a = abs(fa - h) / max(h, 1.0)
                    dist_b = abs(fb - h) / max(h, 1.0)
                    if dist_a < 0.15 and dist_b < 0.15:
                        reinforcement += 1.0

                # Formants too close create roughness
                cb = _critical_bandwidth((fa + fb) / 2.0)
                separation = abs(fa - fb)
                if 0 < separation < cb * 0.5:
                    roughness_penalty += 1.0 - separation / (cb * 0.5)

        n_pairs = len(formants_a) * len(formants_b)
        if n_pairs > 0:
            norm_reinforce = min(reinforcement / n_pairs, 1.0)
            norm_roughness = min(roughness_penalty / n_pairs, 1.0)
        else:
            norm_reinforce = 0.0
            norm_roughness = 0.0

        scores.append(max(0.0, norm_reinforce - 0.5 * norm_roughness))

    return float(np.clip(np.mean(scores), 0.0, 1.0))


def octave_coupling_strength(
    fundamental_a: float,
    fundamental_b: float,
) -> float:
    """Strength of octave coupling between two notes.

    Octave coupling occurs when two pitches are (approximately) an
    octave apart.  Perfect octaves fuse strongly (perceived as a
    single, richer tone); deviations weaken the coupling rapidly.
    Based on Terhardt's virtual-pitch model.

    Parameters
    ----------
    fundamental_a, fundamental_b : float
        Fundamental frequencies in Hz.

    Returns
    -------
    float
        Coupling strength in [0, 1].  1 = perfect octave, 0 = no
        octave relationship.
    """
    if fundamental_a <= 0 or fundamental_b <= 0:
        return 0.0

    # Ratio in terms of octave alignment
    ratio = max(fundamental_a, fundamental_b) / min(fundamental_a, fundamental_b)

    if ratio < 1.0 + 1e-12:
        # Unison — strong coupling but not octave coupling
        return 0.0

    # Find nearest octave ratio (2^n for integer n)
    log_ratio = math.log2(ratio)
    nearest_octave = round(log_ratio)
    if nearest_octave < 1:
        return 0.0

    # Deviation from perfect octave in cents
    deviation_cents = abs(log_ratio - nearest_octave) * 1200.0

    # Coupling drops as a Gaussian with sigma ~25 cents
    # (about a quarter-tone tolerance before fusion breaks)
    sigma_cents = 25.0
    coupling = math.exp(-0.5 * (deviation_cents / sigma_cents) ** 2)

    # Weight by octave distance: closer octaves couple more strongly
    # (Rimsky-Korsakov: doubling at the octave is strongest, two octaves
    # progressively weaker)
    distance_weight = 1.0 / nearest_octave

    return float(np.clip(coupling * distance_weight, 0.0, 1.0))


def unison_beating(
    freq_a: float,
    freq_b: float,
) -> Tuple[float, float]:
    """Beat frequency and perceptual roughness of near-unison doubling.

    When two instruments play nominally the same pitch, tuning
    discrepancies create audible beats.  Up to about 15 Hz the beats
    are perceived as fluctuations in loudness; beyond that they become
    roughness (Plomp & Levelt, 1965).

    Parameters
    ----------
    freq_a, freq_b : float
        Fundamental frequencies in Hz.

    Returns
    -------
    beat_freq : float
        Beat frequency in Hz (always non-negative).
    roughness : float
        Perceptual roughness in [0, 1].  0 = smooth, 1 = very rough.
    """
    if freq_a <= 0 or freq_b <= 0:
        return 0.0, 0.0

    beat_freq = abs(freq_a - freq_b)

    # Roughness model: peaks near 25-35% of critical bandwidth,
    # drops off beyond one critical band
    mean_freq = (freq_a + freq_b) / 2.0
    cb = _critical_bandwidth(mean_freq)

    if cb < 1e-12:
        return beat_freq, 0.0

    x = beat_freq / cb
    # Plomp-Levelt roughness curve (simplified): peaks at x ~ 0.25
    roughness = x * math.exp(1.0 - x / 0.25) / 0.25 if x > 0 else 0.0
    roughness = min(roughness, 1.0)

    return beat_freq, roughness


def section_balance(
    instrument_levels_db: Sequence[float],
    target_balance: Sequence[float],
) -> float:
    """How well balanced a section is relative to a target balance.

    Computes the RMS deviation between actual power ratios and the
    target power ratios across all instruments in the section.
    Rimsky-Korsakov's ideal: strings-to-winds should be roughly 3:1
    in a full orchestra; within a section, parts should follow the
    dynamic markings proportionally.

    Parameters
    ----------
    instrument_levels_db : sequence of float
        Sound pressure level of each instrument in dB.
    target_balance : sequence of float
        Desired relative level of each instrument (linear scale,
        will be normalized to sum to 1).

    Returns
    -------
    float
        Balance score in [0, 1].  1 = perfect balance, 0 = very
        unbalanced.
    """
    levels = np.asarray(instrument_levels_db, dtype=np.float64)
    target = np.asarray(target_balance, dtype=np.float64)

    if levels.shape[0] != target.shape[0]:
        raise ValueError("instrument_levels_db and target_balance must have the same length")

    if levels.shape[0] == 0:
        return 1.0

    # Convert dB to linear power
    linear = 10.0 ** (levels / 10.0)
    total = linear.sum()
    if total < 1e-15:
        return 0.0

    actual_ratios = linear / total

    # Normalize target
    t_sum = target.sum()
    if t_sum < 1e-15:
        return 0.0
    target_ratios = target / t_sum

    # RMS deviation
    rms_dev = float(np.sqrt(np.mean((actual_ratios - target_ratios) ** 2)))

    # Map to [0, 1]: perfect = 0 deviation → score 1.0
    # Maximum possible deviation when one instrument has all the energy
    # is bounded by ~1.0, so scale accordingly
    return float(np.clip(1.0 - rms_dev * math.sqrt(len(levels)), 0.0, 1.0))


def tutti_density(
    n_instruments: int,
    register_spread: float,
    dynamic_range: float,
) -> float:
    """Density metric for tutti passages.

    A tutti passage's density depends on how many instruments are
    playing, how wide the registral spread is (in octaves), and the
    dynamic range.  Rimsky-Korsakov recommends wide spacing in loud
    tutti to maintain clarity; dense scoring in a narrow register
    creates mud.

    Parameters
    ----------
    n_instruments : int
        Number of simultaneously sounding instruments.
    register_spread : float
        Registral spread in octaves (e.g. 4.0 for a four-octave span).
    dynamic_range : float
        Dynamic range in dB (typically 10-60 for orchestral writing).

    Returns
    -------
    float
        Density metric in [0, 1].  Higher = denser (not necessarily
        bad — a dense tutti can be effective if well-spaced).
    """
    if n_instruments <= 0:
        return 0.0

    # Instruments per octave: higher = denser
    effective_spread = max(register_spread, 0.5)  # avoid division by zero
    instruments_per_octave = n_instruments / effective_spread

    # Normalize: a typical full orchestra has ~20 independent parts
    # over ~5 octaves = 4 per octave.  Score of 1.0 at about 8 per octave.
    spatial_density = min(instruments_per_octave / 8.0, 1.0)

    # Dynamic compression: higher dynamic range = more headroom = lower density
    dynamic_factor = 1.0 - min(max(dynamic_range, 0.0) / 60.0, 1.0)

    # Combined density: geometric mean biased toward spatial density
    density = 0.7 * spatial_density + 0.3 * dynamic_factor

    return float(np.clip(density, 0.0, 1.0))


def recommend_doublings(
    melody_pitches: Sequence[float],
    available_instruments_with_ranges: Dict[str, Tuple[float, float]],
    top_k: int = 5,
) -> List[DoublingPair]:
    """Suggest optimal doublings for a melody.

    For each pair of available instruments, evaluates how effective
    doubling would be across the given melody pitches, considering
    range fitness, octave coupling potential, and complementary
    timbral character (approximated via registral placement).

    Follows Adler's doubling hierarchy:
    1. Same family, octave apart (strongest reinforcement)
    2. Different family, unison (colour blend)
    3. Same family, unison (power boost)

    Parameters
    ----------
    melody_pitches : sequence of float
        MIDI pitch values of the melody to be doubled.
    available_instruments_with_ranges : dict
        Mapping of instrument name to (low_midi, high_midi) range.
    top_k : int
        Number of top doublings to return.

    Returns
    -------
    list of DoublingPair
        Sorted by blend_score descending.
    """
    pitches = np.asarray(melody_pitches, dtype=np.float64)
    if len(pitches) == 0 or len(available_instruments_with_ranges) < 2:
        return []

    instruments = list(available_instruments_with_ranges.keys())
    ranges = available_instruments_with_ranges

    results: list[DoublingPair] = []

    for i in range(len(instruments)):
        for j in range(i + 1, len(instruments)):
            inst_a = instruments[i]
            inst_b = instruments[j]
            lo_a, hi_a = ranges[inst_a]
            lo_b, hi_b = ranges[inst_b]

            # --- Unison doubling ---
            # Check what fraction of melody fits both ranges
            fits_both = np.sum(
                (pitches >= lo_a) & (pitches <= hi_a) &
                (pitches >= lo_b) & (pitches <= hi_b)
            )
            unison_coverage = float(fits_both) / len(pitches) if len(pitches) > 0 else 0.0

            if unison_coverage > 0.5:
                # Estimate blend from register similarity
                mid_a = (lo_a + hi_a) / 2.0
                mid_b = (lo_b + hi_b) / 2.0
                # Instruments in similar registers blend well in unison
                register_sim = math.exp(-abs(mid_a - mid_b) / 24.0)
                # But different-register instruments create interesting colour
                colour_bonus = 0.1 * (1.0 - register_sim)
                blend = float(np.clip(
                    0.6 * unison_coverage + 0.3 * register_sim + 0.1 + colour_bonus,
                    0.0, 1.0,
                ))
                results.append(DoublingPair(
                    instruments=(inst_a, inst_b),
                    interval=0,
                    blend_score=blend,
                ))

            # --- Octave doubling ---
            fits_octave = np.sum(
                (pitches >= lo_a) & (pitches <= hi_a) &
                ((pitches + 12) >= lo_b) & ((pitches + 12) <= hi_b)
            )
            octave_coverage = float(fits_octave) / len(pitches) if len(pitches) > 0 else 0.0

            if octave_coverage > 0.3:
                # Octave doublings are generally effective (Adler hierarchy #1)
                f0_mid = 440.0 * (2.0 ** ((float(np.median(pitches)) - 69.0) / 12.0))
                coupling = octave_coupling_strength(f0_mid, f0_mid * 2.0)
                blend = float(np.clip(
                    0.5 * octave_coverage + 0.4 * coupling + 0.1,
                    0.0, 1.0,
                ))
                results.append(DoublingPair(
                    instruments=(inst_a, inst_b),
                    interval=12,
                    blend_score=blend,
                ))

            # Also check the reverse direction (inst_b an octave below)
            fits_octave_rev = np.sum(
                (pitches >= lo_b) & (pitches <= hi_b) &
                ((pitches + 12) >= lo_a) & ((pitches + 12) <= hi_a)
            )
            octave_coverage_rev = float(fits_octave_rev) / len(pitches) if len(pitches) > 0 else 0.0

            if octave_coverage_rev > 0.3:
                f0_mid = 440.0 * (2.0 ** ((float(np.median(pitches)) - 69.0) / 12.0))
                coupling = octave_coupling_strength(f0_mid, f0_mid * 2.0)
                blend = float(np.clip(
                    0.5 * octave_coverage_rev + 0.4 * coupling + 0.1,
                    0.0, 1.0,
                ))
                results.append(DoublingPair(
                    instruments=(inst_b, inst_a),
                    interval=12,
                    blend_score=blend,
                ))

    # Sort by blend_score descending, take top_k
    results.sort(key=lambda dp: dp.blend_score, reverse=True)
    return results[:top_k]
