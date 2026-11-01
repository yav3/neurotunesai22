"""Automatic role assignment for ensemble collaboration — skill-based matching and workload balancing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CollaboratorRole(Enum):
    MELODIST = "melodist"
    HARMONIST = "harmonist"
    BASSIST = "bassist"
    RHYTHMIST = "rhythmist"
    ARRANGER = "arranger"
    CONDUCTOR = "conductor"


@dataclass
class SkillProfile:
    melodic_strength: float
    harmonic_strength: float
    rhythmic_strength: float
    range_comfort: tuple[int, int] = (48, 84)
    arranging_skill: float = 0.5
    leadership_skill: float = 0.5


ROLE_RANGES: dict[CollaboratorRole, tuple[int, int]] = {
    CollaboratorRole.MELODIST: (60, 96),
    CollaboratorRole.HARMONIST: (48, 84),
    CollaboratorRole.BASSIST: (28, 60),
    CollaboratorRole.RHYTHMIST: (35, 72),
    CollaboratorRole.ARRANGER: (28, 96),
    CollaboratorRole.CONDUCTOR: (28, 96),
}


def role_compatibility(role: CollaboratorRole, profile: SkillProfile) -> float:
    if role == CollaboratorRole.MELODIST:
        base = profile.melodic_strength
    elif role == CollaboratorRole.HARMONIST:
        base = 0.6 * profile.harmonic_strength + 0.4 * profile.melodic_strength
    elif role == CollaboratorRole.BASSIST:
        base = 0.5 * profile.harmonic_strength + 0.5 * profile.rhythmic_strength
    elif role == CollaboratorRole.RHYTHMIST:
        base = profile.rhythmic_strength
    elif role == CollaboratorRole.ARRANGER:
        base = profile.arranging_skill
    elif role == CollaboratorRole.CONDUCTOR:
        base = profile.leadership_skill
    else:
        base = 0.5
    role_lo, role_hi = ROLE_RANGES[role]
    pf_lo, pf_hi = profile.range_comfort
    overlap_lo = max(role_lo, pf_lo)
    overlap_hi = min(role_hi, pf_hi)
    role_span = role_hi - role_lo
    overlap = max(0, overlap_hi - overlap_lo)
    range_fit = overlap / role_span if role_span > 0 else 0.0
    return 0.7 * base + 0.3 * range_fit


def _build_cost_matrix(profiles: list[SkillProfile], roles: list[CollaboratorRole]) -> list[list[float]]:
    return [[1.0 - role_compatibility(r, p) for r in roles] for p in profiles]


def assign_roles(
    profiles: list[SkillProfile],
    ensemble_size: int | None = None,
) -> list[tuple[int, CollaboratorRole]]:
    if not profiles:
        return []
    n = len(profiles)
    all_roles = list(CollaboratorRole)
    if ensemble_size is None:
        ensemble_size = min(n, len(all_roles))
    roles = all_roles[:ensemble_size]
    if n < len(roles):
        roles = roles[:n]
    cost = _build_cost_matrix(profiles, roles)
    n_p = len(profiles)
    n_r = len(roles)
    used_p: set[int] = set()
    used_r: set[int] = set()
    assignments: list[tuple[int, CollaboratorRole]] = []
    for _ in range(min(n_p, n_r)):
        best_val = float("inf")
        best_i, best_j = 0, 0
        for i in range(n_p):
            if i in used_p:
                continue
            for j in range(n_r):
                if j in used_r:
                    continue
                if cost[i][j] < best_val:
                    best_val = cost[i][j]
                    best_i, best_j = i, j
        used_p.add(best_i)
        used_r.add(best_j)
        assignments.append((best_i, roles[best_j]))
    return assignments


def suggest_voice_ranges(roles: list[CollaboratorRole]) -> dict[CollaboratorRole, tuple[int, int]]:
    return {r: ROLE_RANGES[r] for r in roles}


def workload_balance(complexities: list[float]) -> float:
    if not complexities or len(complexities) < 2:
        return 0.0
    n = len(complexities)
    total = sum(complexities)
    if total <= 0:
        return 0.0
    s = sorted(complexities)
    numerator = sum((2 * (i + 1) - n - 1) * s[i] for i in range(n))
    return numerator / (n * total)


def ensemble_diversity(profiles: list[SkillProfile]) -> float:
    if len(profiles) < 2:
        return 0.0
    vectors = [
        (p.melodic_strength, p.harmonic_strength, p.rhythmic_strength,
         p.arranging_skill, p.leadership_skill)
        for p in profiles
    ]
    n = len(vectors)
    total_dist = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j])) ** 0.5
            total_dist += d
            pairs += 1
    return total_dist / pairs if pairs > 0 else 0.0
