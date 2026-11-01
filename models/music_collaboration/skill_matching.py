"""Musician skill and role matching for ensemble formation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MusicianProfile:
    name: str
    instruments: list[str] = field(default_factory=list)
    skill_levels: dict[str, float] = field(default_factory=dict)  # instrument -> 0-1
    genres: list[str] = field(default_factory=list)
    availability: float = 1.0  # 0-1

    def skill_for(self, instrument: str) -> float:
        return self.skill_levels.get(instrument, 0.0)


@dataclass
class EnsembleRequirement:
    roles: dict[str, str]  # role_name -> instrument
    min_skill: dict[str, float] = field(default_factory=dict)  # role -> min skill
    preferred_genres: list[str] = field(default_factory=list)


def genre_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def role_fitness(musician: MusicianProfile, instrument: str,
                 min_skill: float = 0.0, preferred_genres: list[str] | None = None) -> float:
    """Score [0,1] for how well a musician fits a specific role."""
    skill = musician.skill_for(instrument)
    if skill < min_skill:
        return 0.0
    genre_score = genre_overlap(musician.genres, preferred_genres or [])
    return 0.6 * skill + 0.2 * genre_score + 0.2 * musician.availability


def greedy_assign(musicians: list[MusicianProfile],
                  requirement: EnsembleRequirement) -> dict[str, str | None]:
    """Greedy assignment: for each role, pick the best-fit available musician."""
    assigned: dict[str, str | None] = {}
    used: set[str] = set()
    sorted_roles = sorted(requirement.roles.keys())

    for role in sorted_roles:
        instrument = requirement.roles[role]
        min_sk = requirement.min_skill.get(role, 0.0)
        best_score = -1.0
        best_name: str | None = None
        for m in musicians:
            if m.name in used:
                continue
            score = role_fitness(m, instrument, min_sk, requirement.preferred_genres)
            if score > best_score:
                best_score = score
                best_name = m.name
        if best_name is not None and best_score > 0:
            assigned[role] = best_name
            used.add(best_name)
        else:
            assigned[role] = None
    return assigned


def ensemble_compatibility(musicians: list[MusicianProfile]) -> float:
    """Average pairwise genre overlap across all musician pairs."""
    n = len(musicians)
    if n < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += genre_overlap(musicians[i].genres, musicians[j].genres)
            pairs += 1
    return total / pairs


def skill_gap_analysis(musicians: list[MusicianProfile],
                       requirement: EnsembleRequirement) -> dict[str, float]:
    """For each role, compute the gap between required and best available skill."""
    assignment = greedy_assign(musicians, requirement)
    gaps: dict[str, float] = {}
    for role, instrument in requirement.roles.items():
        min_sk = requirement.min_skill.get(role, 0.0)
        name = assignment.get(role)
        if name is None:
            gaps[role] = min_sk
        else:
            m = next(p for p in musicians if p.name == name)
            sk = m.skill_for(instrument)
            gaps[role] = max(0.0, min_sk - sk)
    return gaps


def recommend_practice(gaps: dict[str, float], hours_per_week: float = 5.0) -> dict[str, float]:
    """Estimate weeks of practice needed to close each skill gap (0.01 skill/hour heuristic)."""
    rate = 0.01
    result: dict[str, float] = {}
    for role, gap in gaps.items():
        if gap <= 0:
            result[role] = 0.0
        else:
            result[role] = math.ceil(gap / (rate * hours_per_week))
    return result
