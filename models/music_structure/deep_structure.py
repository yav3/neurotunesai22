"""Deep music structure analysis — GTTM, form detection, prolongation, repetition."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# §1  GTTM preference rules (Lerdahl & Jackendoff, 1983)
# ---------------------------------------------------------------------------

def gpr_proximity(iois: np.ndarray) -> np.ndarray:
    """GPR 2a: group boundary strength ∝ IOI gap relative to neighbours."""
    if len(iois) < 3:
        return np.zeros(max(len(iois) - 1, 0))
    strength = np.zeros(len(iois) - 1)
    for i in range(1, len(iois) - 1):
        local_mean = (iois[i - 1] + iois[i] + iois[i + 1]) / 3.0
        if local_mean > 0:
            strength[i] = iois[i] / local_mean
    return strength


def gpr_change(features: np.ndarray) -> np.ndarray:
    """GPR 3: boundary where successive features change maximally."""
    if features.ndim == 1:
        features = features[:, None]
    diffs = np.linalg.norm(np.diff(features, axis=0), axis=1)
    if diffs.max() > 0:
        diffs /= diffs.max()
    return diffs


def mpr_regularity(onsets: np.ndarray, level: int = 1) -> float:
    """MPR 10: metrical regularity — ratio of on-beat to off-beat events."""
    if len(onsets) < 2:
        return 0.0
    grid = 2 ** level
    phases = np.round(onsets * grid) % grid
    on_beat = np.sum(phases == 0)
    return float(on_beat / len(onsets))


def gttm_grouping_score(iois: np.ndarray, features: np.ndarray,
                        alpha: float = 0.6) -> np.ndarray:
    """Combined grouping boundary score: α·proximity + (1-α)·change."""
    n = min(len(iois), features.shape[0])
    prox = gpr_proximity(iois[:n])
    chg = gpr_change(features[:n])
    m = min(len(prox), len(chg))
    return alpha * prox[:m] + (1.0 - alpha) * chg[:m]

# ---------------------------------------------------------------------------
# §2  Hierarchical phrase segmentation via SSM diagonal analysis
# ---------------------------------------------------------------------------

def ssm_from_features(features: np.ndarray) -> np.ndarray:
    """Self-similarity matrix from L2-normalised features."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = features / norms
    return normed @ normed.T


def checkerboard_kernel(size: int) -> np.ndarray:
    """Foote (2000) checkerboard kernel for novelty detection."""
    k = np.ones((size, size))
    k[:size // 2, size // 2:] = -1
    k[size // 2:, :size // 2] = -1
    return k


def ssm_novelty(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """Novelty curve from SSM via checkerboard kernel convolution."""
    n = ssm.shape[0]
    kern = checkerboard_kernel(kernel_size)
    half = kernel_size // 2
    novelty = np.zeros(n)
    for i in range(half, n - half):
        patch = ssm[i - half:i + half, i - half:i + half]
        if patch.shape == kern.shape:
            novelty[i] = np.abs(np.sum(patch * kern))
    return novelty


def hierarchical_segment(ssm: np.ndarray, levels: int = 3,
                         kernel_sizes: Tuple[int, ...] = (64, 32, 16)
                         ) -> List[List[int]]:
    """Multi-level segmentation: novelty peaks at different kernel scales."""
    boundaries = []
    for lev in range(min(levels, len(kernel_sizes))):
        nov = ssm_novelty(ssm, kernel_sizes[lev])
        thresh = np.mean(nov) + np.std(nov)
        peaks = []
        for i in range(1, len(nov) - 1):
            if nov[i] > nov[i - 1] and nov[i] > nov[i + 1] and nov[i] > thresh:
                peaks.append(i)
        boundaries.append(peaks)
    return boundaries

# ---------------------------------------------------------------------------
# §3  Large-scale form detection
# ---------------------------------------------------------------------------

FORM_TEMPLATES = {
    "binary": "AB",
    "ternary": "ABA",
    "rondo": "ABACA",
    "sonata": "ABA'",
    "strophic": "AAA",
    "through-composed": "ABCD",
    "verse-chorus": "ABAB",
    "twelve-bar": "AAB",
}


def section_similarity(features_list: List[np.ndarray]) -> np.ndarray:
    """Pairwise cosine similarity between section-mean feature vectors."""
    means = []
    for f in features_list:
        m = f.mean(axis=0) if f.ndim == 2 else f
        norm = np.linalg.norm(m)
        means.append(m / norm if norm > 0 else m)
    n = len(means)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = float(np.dot(means[i], means[j]))
    return sim


def label_sections(sim: np.ndarray, threshold: float = 0.85) -> str:
    """Assign letter labels: sections above threshold are same label."""
    n = sim.shape[0]
    labels = [''] * n
    current_label = 0
    for i in range(n):
        if labels[i] == '':
            labels[i] = chr(ord('A') + current_label)
            for j in range(i + 1, n):
                if labels[j] == '' and sim[i, j] >= threshold:
                    labels[j] = labels[i]
            current_label += 1
    return ''.join(labels)


def classify_form(form_string: str) -> str:
    """Match a section-label string against known form templates."""
    best_match = "unknown"
    best_score = 0.0
    for name, template in FORM_TEMPLATES.items():
        if len(form_string) == 0:
            continue
        # simple edit-distance-like match
        m = min(len(form_string), len(template))
        matches = sum(1 for a, b in zip(form_string[:m], template[:m]) if a == b)
        score = matches / max(len(form_string), len(template))
        if score > best_score:
            best_score = score
            best_match = name
    return best_match

# ---------------------------------------------------------------------------
# §4  Structural complexity measures
# ---------------------------------------------------------------------------

def lempel_ziv_complexity(seq: np.ndarray) -> float:
    """Lempel-Ziv (1976) complexity: number of distinct substrings."""
    s = ''.join(str(int(x)) for x in seq.flatten())
    n = len(s)
    if n == 0:
        return 0.0
    i, k, l_val = 0, 1, 1
    c = 1
    while k < n:
        if s[k] == s[i]:
            i += 1
            k += 1
        else:
            if i >= l_val:
                c += 1
                l_val = k
                i = 0
            else:
                i = 0
                if s[k] != s[i]:
                    c += 1
                    l_val = k
            k += 1
    if k != l_val:
        c += 1
    return c / (n / np.log2(max(n, 2)))


def information_rate(seq: np.ndarray, order: int = 1) -> float:
    """Entropy rate of a sequence via order-k Markov estimate."""
    symbols = np.unique(seq)
    n_sym = len(symbols)
    if n_sym <= 1 or len(seq) <= order:
        return 0.0
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    counts = {}
    for i in range(len(seq) - order):
        ctx = tuple(seq[i:i + order])
        nxt = seq[i + order]
        if ctx not in counts:
            counts[ctx] = np.zeros(n_sym)
        counts[ctx][sym_to_idx[nxt]] += 1
    h = 0.0
    total = 0.0
    for ctx, c in counts.items():
        s = c.sum()
        if s > 0:
            p = c / s
            p = p[p > 0]
            h -= s * np.sum(p * np.log2(p))
            total += s
    return float(h / total) if total > 0 else 0.0


def compression_ratio(seq: np.ndarray) -> float:
    """Ratio of unique n-grams to total n-grams (n=2)."""
    if len(seq) < 2:
        return 1.0
    bigrams = set()
    for i in range(len(seq) - 1):
        bigrams.add((int(seq[i]), int(seq[i + 1])))
    return len(bigrams) / (len(seq) - 1)

# ---------------------------------------------------------------------------
# §5  Prolongational reduction (simplified tree-based)
# ---------------------------------------------------------------------------

@dataclass
class ProlongNode:
    start: int
    end: int
    tension: float = 0.0
    children: List['ProlongNode'] = field(default_factory=list)
    label: str = ""


def tension_at_boundary(features_before: np.ndarray,
                        features_after: np.ndarray) -> float:
    """Tension = cosine distance between adjacent section means."""
    m1 = features_before.mean(axis=0) if features_before.ndim == 2 else features_before
    m2 = features_after.mean(axis=0) if features_after.ndim == 2 else features_after
    n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return 1.0 - float(np.dot(m1, m2) / (n1 * n2))


def build_prolongation_tree(features: np.ndarray,
                            boundaries: List[int]) -> ProlongNode:
    """Recursive bisection prolongation tree."""
    root = ProlongNode(start=0, end=features.shape[0], label="root")
    if not boundaries:
        return root
    bounds = sorted(set([0] + boundaries + [features.shape[0]]))
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        t = 0.0
        if i > 0:
            prev = features[bounds[i - 1]:s]
            curr = features[s:e]
            if len(prev) > 0 and len(curr) > 0:
                t = tension_at_boundary(prev, curr)
        root.children.append(ProlongNode(start=s, end=e, tension=t,
                                         label=f"section_{i}"))
    return root


def tree_depth(node: ProlongNode) -> int:
    if not node.children:
        return 0
    return 1 + max(tree_depth(c) for c in node.children)


def tree_branching(node: ProlongNode) -> float:
    """Mean branching factor."""
    if not node.children:
        return 0.0
    total_children = len(node.children)
    sub = sum(tree_branching(c) for c in node.children if c.children)
    n_internal = sum(1 for c in node.children if c.children)
    if n_internal > 0:
        return (total_children + sub) / (1 + n_internal)
    return float(total_children)

# ---------------------------------------------------------------------------
# §6  Section transition analysis
# ---------------------------------------------------------------------------

def transition_energy_change(rms: np.ndarray, boundary: int,
                             window: int = 10) -> float:
    """Energy ratio across a section boundary."""
    before = rms[max(0, boundary - window):boundary]
    after = rms[boundary:boundary + window]
    if len(before) == 0 or len(after) == 0:
        return 0.0
    e_before = np.mean(before ** 2)
    e_after = np.mean(after ** 2)
    if e_before == 0:
        return float('inf') if e_after > 0 else 0.0
    return float(e_after / e_before)


def transition_smoothness(features: np.ndarray, boundary: int,
                          window: int = 5) -> float:
    """Smoothness = 1 − normalised cosine distance across boundary."""
    before = features[max(0, boundary - window):boundary].mean(axis=0)
    after = features[boundary:boundary + window].mean(axis=0)
    n1, n2 = np.linalg.norm(before), np.linalg.norm(after)
    if n1 == 0 or n2 == 0:
        return 1.0
    cos_sim = float(np.dot(before, after) / (n1 * n2))
    return max(0.0, cos_sim)


def transition_quality(rms: np.ndarray, features: np.ndarray,
                       boundary: int) -> Dict[str, float]:
    """Combined transition quality metrics."""
    return {
        "energy_ratio": transition_energy_change(rms, boundary),
        "smoothness": transition_smoothness(features, boundary),
        "contrast": 1.0 - transition_smoothness(features, boundary),
    }

# ---------------------------------------------------------------------------
# §7  Repetition structure — Smith-Waterman local alignment
# ---------------------------------------------------------------------------

def smith_waterman(seq_a: np.ndarray, seq_b: np.ndarray,
                   match: float = 2.0, mismatch: float = -1.0,
                   gap: float = -0.5) -> Tuple[float, List[Tuple[int, int]]]:
    """Smith-Waterman local alignment score and traceback."""
    m, n = len(seq_a), len(seq_b)
    H = np.zeros((m + 1, n + 1))
    traceback = np.zeros((m + 1, n + 1), dtype=int)  # 0=stop,1=diag,2=up,3=left
    max_score, max_pos = 0.0, (0, 0)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if seq_a[i - 1] == seq_b[j - 1] else mismatch
            scores = [0, H[i - 1, j - 1] + s, H[i - 1, j] + gap, H[i, j - 1] + gap]
            best = int(np.argmax(scores))
            H[i, j] = scores[best]
            traceback[i, j] = best
            if H[i, j] > max_score:
                max_score = H[i, j]
                max_pos = (i, j)
    # traceback
    aligned = []
    i, j = max_pos
    while traceback[i, j] != 0:
        if traceback[i, j] == 1:
            aligned.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif traceback[i, j] == 2:
            i -= 1
        else:
            j -= 1
    aligned.reverse()
    return float(max_score), aligned


def repetition_matrix(section_labels: str) -> np.ndarray:
    """Binary repetition matrix: R[i,j] = 1 iff label_i == label_j."""
    n = len(section_labels)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if section_labels[i] == section_labels[j]:
                R[i, j] = 1.0
    return R


def repetition_fraction(section_labels: str) -> float:
    """Fraction of section pairs that are repetitions (excluding diagonal)."""
    n = len(section_labels)
    if n <= 1:
        return 0.0
    R = repetition_matrix(section_labels)
    off_diag = R.sum() - n
    return float(off_diag / (n * (n - 1)))

# ---------------------------------------------------------------------------
# §8  Structural expectation — surprise at boundaries
# ---------------------------------------------------------------------------

def section_transition_probs(labels: str) -> Dict[str, Dict[str, float]]:
    """Empirical transition probabilities between section labels."""
    trans: Dict[str, Dict[str, float]] = {}
    for i in range(len(labels) - 1):
        src, dst = labels[i], labels[i + 1]
        if src not in trans:
            trans[src] = {}
        trans[src][dst] = trans[src].get(dst, 0) + 1
    for src in trans:
        total = sum(trans[src].values())
        for dst in trans[src]:
            trans[src][dst] /= total
    return trans


def boundary_surprise(labels: str) -> np.ndarray:
    """Information-theoretic surprise at each section transition."""
    probs = section_transition_probs(labels)
    surprises = np.zeros(max(len(labels) - 1, 0))
    for i in range(len(labels) - 1):
        src, dst = labels[i], labels[i + 1]
        p = probs.get(src, {}).get(dst, 1e-10)
        surprises[i] = -np.log2(p)
    return surprises


def entropy_rate_sections(labels: str) -> float:
    """Shannon entropy rate of the section label sequence."""
    if len(labels) <= 1:
        return 0.0
    probs = section_transition_probs(labels)
    h = 0.0
    n = len(labels) - 1
    for i in range(n):
        src, dst = labels[i], labels[i + 1]
        p = probs.get(src, {}).get(dst, 1e-10)
        h -= np.log2(p)
    return float(h / n)

# ---------------------------------------------------------------------------
# §9  Form-function mapping — section role classification
# ---------------------------------------------------------------------------

ROLE_PROFILES = {
    "intro":    {"energy": 0.3, "novelty": 0.7, "repetition": 0.1},
    "verse":    {"energy": 0.5, "novelty": 0.4, "repetition": 0.7},
    "chorus":   {"energy": 0.8, "novelty": 0.3, "repetition": 0.9},
    "bridge":   {"energy": 0.6, "novelty": 0.8, "repetition": 0.2},
    "coda":     {"energy": 0.2, "novelty": 0.5, "repetition": 0.3},
    "development": {"energy": 0.7, "novelty": 0.9, "repetition": 0.4},
    "recapitulation": {"energy": 0.6, "novelty": 0.2, "repetition": 0.95},
}


def classify_section_role(energy: float, novelty: float,
                          repetition: float) -> str:
    """Nearest-profile classification of a section's functional role."""
    features = np.array([energy, novelty, repetition])
    best_role = "unknown"
    best_dist = float('inf')
    for role, profile in ROLE_PROFILES.items():
        prof = np.array([profile["energy"], profile["novelty"],
                         profile["repetition"]])
        d = float(np.linalg.norm(features - prof))
        if d < best_dist:
            best_dist = d
            best_role = role
    return best_role


def assign_roles(section_energies: np.ndarray,
                 section_novelties: np.ndarray,
                 section_labels: str) -> List[str]:
    """Assign functional roles to each section."""
    n = len(section_labels)
    rep_scores = np.zeros(n)
    for i in range(n):
        rep_scores[i] = sum(1 for j in range(n) if j != i and
                            section_labels[j] == section_labels[i]) / max(n - 1, 1)
    roles = []
    for i in range(n):
        e = float(section_energies[i]) if i < len(section_energies) else 0.5
        nv = float(section_novelties[i]) if i < len(section_novelties) else 0.5
        roles.append(classify_section_role(e, nv, rep_scores[i]))
    return roles

# ---------------------------------------------------------------------------
# §10  Pipeline
# ---------------------------------------------------------------------------

@dataclass
class DeepStructureAnalysis:
    form_string: str = ""
    form_type: str = "unknown"
    n_sections: int = 0
    section_roles: List[str] = field(default_factory=list)
    lz_complexity: float = 0.0
    info_rate: float = 0.0
    compression: float = 0.0
    repetition_frac: float = 0.0
    entropy_rate: float = 0.0
    boundary_surprises: np.ndarray = field(default_factory=lambda: np.array([]))
    tree_depth: int = 0
    transition_qualities: List[Dict[str, float]] = field(default_factory=list)
    hierarchical_boundaries: List[List[int]] = field(default_factory=list)


def analyze_structure_deep(features: np.ndarray,
                           rms: Optional[np.ndarray] = None,
                           section_boundaries: Optional[List[int]] = None
                           ) -> DeepStructureAnalysis:
    """Full deep structure analysis pipeline."""
    if features.ndim == 1:
        features = features[:, None]
    n = features.shape[0]
    if rms is None:
        rms = np.linalg.norm(features, axis=1)

    ssm = ssm_from_features(features)
    hier_bounds = hierarchical_segment(ssm)

    if section_boundaries is None:
        section_boundaries = hier_bounds[0] if hier_bounds and hier_bounds[0] else []

    bounds = sorted(set([0] + list(section_boundaries) + [n]))
    sections = [features[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]

    if len(sections) >= 2:
        sim = section_similarity(sections)
        form_str = label_sections(sim)
    else:
        form_str = "A"

    form_type = classify_form(form_str)

    # quantise for complexity
    quantised = np.digitize(rms, np.linspace(rms.min(), rms.max() + 1e-10, 8))
    lz = lempel_ziv_complexity(quantised)
    ir = information_rate(quantised, order=2)
    cr = compression_ratio(quantised)
    rep = repetition_fraction(form_str)
    ent_rate = entropy_rate_sections(form_str)
    surprises = boundary_surprise(form_str)

    # section-level metrics for role classification
    sec_energies = np.array([np.mean(rms[bounds[i]:bounds[i + 1]] ** 2)
                             for i in range(len(bounds) - 1)])
    if sec_energies.max() > 0:
        sec_energies /= sec_energies.max()
    nov = ssm_novelty(ssm, min(16, n // 4 + 1))
    sec_novelties = np.array([np.mean(nov[bounds[i]:bounds[i + 1]])
                              for i in range(len(bounds) - 1)])
    if sec_novelties.max() > 0:
        sec_novelties /= sec_novelties.max()
    roles = assign_roles(sec_energies, sec_novelties, form_str)

    # transition qualities
    trans_q = []
    for b in section_boundaries:
        if 0 < b < n:
            trans_q.append(transition_quality(rms, features, b))

    tree = build_prolongation_tree(features, list(section_boundaries))

    return DeepStructureAnalysis(
        form_string=form_str,
        form_type=form_type,
        n_sections=len(sections),
        section_roles=roles,
        lz_complexity=lz,
        info_rate=ir,
        compression=cr,
        repetition_frac=rep,
        entropy_rate=ent_rate,
        boundary_surprises=surprises,
        tree_depth=tree_depth(tree),
        transition_qualities=trans_q,
        hierarchical_boundaries=hier_bounds,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # build a simple ABA structure
    feat_a = rng.standard_normal((50, 12))
    feat_b = rng.standard_normal((50, 12)) + 3.0
    features = np.vstack([feat_a, feat_b, feat_a + rng.standard_normal((50, 12)) * 0.1])
    result = analyze_structure_deep(features, section_boundaries=[50, 100])
    assert result.n_sections == 3
    assert len(result.form_string) == 3
    assert result.lz_complexity > 0
    print(f"Form: {result.form_string} ({result.form_type})")
    print(f"Roles: {result.section_roles}")
    print(f"LZ={result.lz_complexity:.3f}, IR={result.info_rate:.3f}, "
          f"CR={result.compression:.3f}")
    print("deep_structure OK")
