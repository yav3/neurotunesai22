"""Musical form analysis: section detection, form classification, structural metrics.

Identifies large-scale formal structures (binary, ternary, rondo, sonata,
verse-chorus, strophic, through-composed) from energy/novelty curves and
feature sequences.  Computes structural metrics: golden-ratio climax
proximity (Howat 1983), arch-form fit (Cone 1968), palindrome symmetry,
and section similarity via DTW.

References
----------
Caplin, W. E. (1998). *Classical Form*. Oxford University Press.
Hepokoski, J. & Darcy, W. (2006). *Elements of Sonata Theory*. Oxford.
Lerdahl, F. & Jackendoff, R. (1983). *A Generative Theory of Tonal Music*.
Howat, R. (1983). *Debussy in Proportion*. Cambridge University Press.
Cone, E. T. (1968). *Musical Form and Musical Performance*. Norton.
Müller, M. (2015). *Fundamentals of Music Processing*. Springer.

Pure numpy / scipy — no external ML dependencies.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

PHI = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio ≈ 1.618


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FormSection:
    """One section of a musical form."""
    label: str
    start_beat: float
    end_beat: float
    energy: float
    key_center: Optional[int] = None  # pitch-class 0-11
    is_climax: bool = False

    @property
    def duration(self) -> float:
        return self.end_beat - self.start_beat


@dataclass
class FormAnalysis:
    """Result of a complete form analysis."""
    sections: List[FormSection]
    form_type: str
    golden_ratio_proximity: float
    arch_score: float
    symmetry: float
    similarity_matrix: Optional[np.ndarray] = None
    sonata_likelihood: float = 0.0
    verse_chorus_likelihood: float = 0.0


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def detect_sections(
    energy_curve: np.ndarray,
    novelty_curve: np.ndarray,
    threshold: float = 0.3,
    min_section_len: int = 8,
) -> List[Tuple[int, int]]:
    """Find section boundaries via peaks in the novelty curve.

    Parameters
    ----------
    energy_curve : 1-D array (one value per beat / frame).
    novelty_curve : 1-D array, same length — e.g. Foote checkerboard output.
    threshold : relative peak-height threshold (fraction of max novelty).
    min_section_len : minimum frames between boundaries.

    Returns
    -------
    List of (start, end) tuples — frame indices.
    """
    n = len(novelty_curve)
    if n < 3:
        return [(0, n)]
    abs_thresh = threshold * np.max(np.abs(novelty_curve))
    peaks: List[int] = []
    for i in range(1, n - 1):
        if (novelty_curve[i] > novelty_curve[i - 1]
                and novelty_curve[i] > novelty_curve[i + 1]
                and novelty_curve[i] >= abs_thresh):
            if not peaks or (i - peaks[-1]) >= min_section_len:
                peaks.append(i)
    boundaries = [0] + peaks + [n]
    sections: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s >= 1:
            sections.append((s, e))
    return sections


# ---------------------------------------------------------------------------
# Feature-based similarity
# ---------------------------------------------------------------------------

def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Simple DTW distance between two 1-D or 2-D feature sequences."""
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = float(np.sum((a[i - 1] - b[j - 1]) ** 2))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[na, nb]))


def section_similarity_matrix(feature_sequences: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise DTW similarity between sections.

    Parameters
    ----------
    feature_sequences : list of arrays, one per section (frames × features).

    Returns
    -------
    Symmetric similarity matrix ∈ [0, 1] (1 = identical).
    """
    k = len(feature_sequences)
    dist = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = _dtw_distance(feature_sequences[i], feature_sequences[j])
            dist[i, j] = dist[j, i] = d
    max_d = np.max(dist) if np.max(dist) > 0 else 1.0
    sim = 1.0 - dist / max_d
    np.fill_diagonal(sim, 1.0)
    return sim


def label_sections_by_similarity(
    sim_matrix: np.ndarray,
    threshold: float = 0.7,
) -> List[str]:
    """Assign alphabetical labels (A, B, C, …) by grouping similar sections.

    Two sections share a label iff their similarity ≥ *threshold*.  Labels
    are assigned greedily in presentation order.
    """
    k = sim_matrix.shape[0]
    labels: List[Optional[str]] = [None] * k
    next_label = 0
    for i in range(k):
        if labels[i] is not None:
            continue
        lbl = chr(ord("A") + min(next_label, 25))
        next_label += 1
        labels[i] = lbl
        for j in range(i + 1, k):
            if labels[j] is None and sim_matrix[i, j] >= threshold:
                labels[j] = lbl
    return [la or "?" for la in labels]


# ---------------------------------------------------------------------------
# Form classification
# ---------------------------------------------------------------------------

def _label_string(sections: List[FormSection]) -> str:
    return "".join(s.label for s in sections)


def classify_form(sections: List[FormSection]) -> str:
    """Classify overall form from labelled sections.

    Returns one of: ``strophic``, ``binary``, ``ternary``, ``rondo``,
    ``sonata``, ``verse-chorus``, ``through-composed``.
    """
    if not sections:
        return "through-composed"
    ls = _label_string(sections)
    unique = set(ls)

    # single label repeated → strophic
    if len(unique) == 1:
        return "strophic"

    # ABACA… → rondo (A returns at positions 0, 2, 4)
    if len(ls) >= 5 and ls[0] == ls[2] == ls[4]:
        return "rondo"

    # ABA → ternary (first = last, differs from middle)
    if len(ls) >= 3 and ls[0] == ls[-1] and ls[0] != ls[1]:
        return "ternary"

    # ABAB… → verse-chorus (alternating pair, ≥ 4 sections)
    if len(ls) >= 4 and len(unique) == 2:
        pair = ls[:2]
        if all(ls[i:i + 2] == pair for i in range(0, len(ls) - 1, 2)):
            return "verse-chorus"

    # two labels, no return → binary
    if len(unique) == 2 and ls[0] != ls[-1]:
        return "binary"

    return "through-composed"


# ---------------------------------------------------------------------------
# Sonata & verse-chorus likelihood
# ---------------------------------------------------------------------------

def sonata_form_likelihood(
    sections: List[FormSection],
    key_sequence: Optional[Sequence[int]] = None,
) -> float:
    """Heuristic sonata-form likelihood ∈ [0, 1].

    Checks for three macro-sections (exposition / development /
    recapitulation) and key-area relationships (I → V → instability → I).
    """
    if len(sections) < 3:
        return 0.0
    ls = _label_string(sections)
    score = 0.0

    if ls[0] == ls[-1]:
        score += 0.3
    mid_labels = set(ls[1:-1])
    if ls[0] not in mid_labels:
        score += 0.2

    if key_sequence is not None and len(key_sequence) == len(sections):
        ks = list(key_sequence)
        tonic = ks[0]
        dominant = (tonic + 7) % 12
        if len(ks) >= 2 and ks[1] == dominant:
            score += 0.2
        if ks[-1] == tonic:
            score += 0.2
        mid_keys = ks[1:-1]
        if mid_keys and len(set(mid_keys)) >= 2:
            score += 0.1
    return min(score, 1.0)


def verse_chorus_likelihood(sections: List[FormSection]) -> float:
    """Heuristic verse-chorus likelihood ∈ [0, 1].

    Based on alternating-pair repetition and energy contrast.
    """
    if len(sections) < 4:
        return 0.0
    ls = _label_string(sections)
    unique = set(ls)
    score = 0.0

    if len(unique) == 2:
        score += 0.3
    elif len(unique) == 3:
        score += 0.15

    pairs = [ls[i:i + 2] for i in range(0, len(ls) - 1, 2)]
    if pairs and all(p == pairs[0] for p in pairs):
        score += 0.3

    energies = [s.energy for s in sections]
    contrasts = [abs(energies[i + 1] - energies[i])
                 for i in range(len(energies) - 1)]
    if contrasts:
        mean_contrast = float(np.mean(contrasts))
        score += min(mean_contrast / 2.0, 0.2)

    if len(unique) >= 2:
        label_energies: Dict[str, List[float]] = {}
        for s in sections:
            label_energies.setdefault(s.label, []).append(s.energy)
        stds = [np.std(v) for v in label_energies.values() if len(v) > 1]
        std_within = float(np.mean(stds)) if stds else 1.0
        if std_within < 0.15:
            score += 0.2

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Golden ratio & arch metrics
# ---------------------------------------------------------------------------

def golden_ratio_position(total_length: float) -> float:
    """Return the golden-ratio position φ/(1+φ) ≈ 0.618 of *total_length*.

    References: Howat (1983) on Debussy's proportional structures.
    """
    return total_length / PHI


def climax_proximity_to_golden_ratio(
    sections: List[FormSection],
    total_length: float,
) -> float:
    """How close the climax is to the golden-ratio point ∈ [0, 1].

    1.0 = climax peak exactly at the φ point; 0.0 = as far as possible.
    """
    if not sections or total_length <= 0:
        return 0.0
    climax_sections = [s for s in sections if s.is_climax]
    if not climax_sections:
        climax_sections = [max(sections, key=lambda s: s.energy)]
    peak = climax_sections[0]
    peak_pos = (peak.start_beat + peak.end_beat) / 2.0
    phi_pos = golden_ratio_position(total_length)
    max_dev = max(phi_pos, total_length - phi_pos)
    if max_dev == 0:
        return 1.0
    return float(1.0 - abs(peak_pos - phi_pos) / max_dev)


def arch_form_score(energy_values: np.ndarray) -> float:
    """How well *energy_values* follows a Gaussian arch (peak in middle).

    Returns Pearson r with a symmetric Gaussian centred at the midpoint.
    Result ∈ [-1, 1]; higher is more arch-like.
    """
    n = len(energy_values)
    if n < 3:
        return 0.0
    x = np.linspace(-3, 3, n)
    template = np.exp(-0.5 * x ** 2)
    ev = np.asarray(energy_values, dtype=float)
    if np.std(ev) < 1e-12 or np.std(template) < 1e-12:
        return 0.0
    return float(np.corrcoef(ev, template)[0, 1])


def palindrome_score(section_labels: Sequence[str]) -> float:
    """Mirror-symmetry of the label sequence ∈ [0, 1].

    1.0 for a perfect palindrome (ABCBA); 0.0 for no symmetry.
    """
    n = len(section_labels)
    if n <= 1:
        return 1.0
    half = n // 2
    matches = sum(
        1 for i in range(half)
        if section_labels[i] == section_labels[n - 1 - i]
    )
    return matches / half


# ---------------------------------------------------------------------------
# Energy trajectory
# ---------------------------------------------------------------------------

def energy_trajectory(sections: List[FormSection]) -> np.ndarray:
    """Normalised energy arc over sections ∈ [0, 1]."""
    if not sections:
        return np.array([])
    e = np.array([s.energy for s in sections], dtype=float)
    mn, mx = float(e.min()), float(e.max())
    if mx - mn < 1e-12:
        return np.full(len(e), 0.5)
    return (e - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_form(
    energy: np.ndarray,
    novelty: np.ndarray,
    features: Optional[List[np.ndarray]] = None,
    key_sequence: Optional[Sequence[int]] = None,
    section_threshold: float = 0.3,
    similarity_threshold: float = 0.7,
    min_section_len: int = 8,
) -> FormAnalysis:
    """Full-pipeline form analysis.

    Parameters
    ----------
    energy : 1-D energy curve (one value per beat / frame).
    novelty : 1-D novelty curve, same length.
    features : optional list of per-section feature arrays (for DTW labelling).
    key_sequence : optional pitch-class per detected section (for sonata scoring).
    section_threshold : novelty peak threshold.
    similarity_threshold : DTW label grouping threshold.
    min_section_len : minimum section length in frames.

    Returns
    -------
    FormAnalysis with classified form, structural metrics, and section list.
    """
    bounds = detect_sections(energy, novelty, section_threshold, min_section_len)
    n = len(energy)

    raw_sections: List[FormSection] = []
    for idx, (s, e) in enumerate(bounds):
        sec_energy = float(np.mean(energy[s:e])) if e > s else 0.0
        kc = None
        if key_sequence is not None and idx < len(key_sequence):
            kc = int(key_sequence[idx])
        raw_sections.append(FormSection(
            label="?",
            start_beat=float(s),
            end_beat=float(e),
            energy=sec_energy,
            key_center=kc,
        ))

    # mark climax
    if raw_sections:
        ci = int(np.argmax([s.energy for s in raw_sections]))
        raw_sections[ci].is_climax = True

    # label by similarity
    sim_mat: Optional[np.ndarray] = None
    if features is not None and len(features) == len(raw_sections):
        sim_mat = section_similarity_matrix(features)
        labels = label_sections_by_similarity(sim_mat, similarity_threshold)
    else:
        if raw_sections:
            e_arr = np.array([s.energy for s in raw_sections])
            med = float(np.median(e_arr))
            labels = ["A" if en <= med else "B" for en in e_arr]
        else:
            labels = []

    for sec, lbl in zip(raw_sections, labels):
        sec.label = lbl

    form_type = classify_form(raw_sections)
    total = float(n)
    gr_prox = climax_proximity_to_golden_ratio(raw_sections, total)
    e_traj = energy_trajectory(raw_sections) if raw_sections else energy
    arch = arch_form_score(e_traj)
    sym = palindrome_score([s.label for s in raw_sections])
    son_lik = sonata_form_likelihood(raw_sections, key_sequence)
    vc_lik = verse_chorus_likelihood(raw_sections)

    return FormAnalysis(
        sections=raw_sections,
        form_type=form_type,
        golden_ratio_proximity=gr_prox,
        arch_score=arch,
        symmetry=sym,
        similarity_matrix=sim_mat,
        sonata_likelihood=son_lik,
        verse_chorus_likelihood=vc_lik,
    )
