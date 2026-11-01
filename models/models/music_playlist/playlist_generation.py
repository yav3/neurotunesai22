"""Rule-based playlist generation from a track pool."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackCandidate:
    track_id: int
    tempo: float
    energy: float
    key: int = 0
    valence: float = 0.5
    genre: str = ""
    duration_sec: float = 210.0


@dataclass
class GenerationRules:
    target_duration_sec: float = 3600.0
    max_tracks: int = 30
    tempo_range: tuple[float, float] = (80.0, 160.0)
    energy_range: tuple[float, float] = (0.2, 0.9)
    max_tempo_jump: float = 30.0
    max_energy_jump: float = 0.3
    genre_weights: dict[str, float] = field(default_factory=dict)
    avoid_artist_repeat_within: int = 3
    seed_track_id: int | None = None


def _tempo_penalty(prev_tempo: float, cand_tempo: float, max_jump: float) -> float:
    diff = abs(prev_tempo - cand_tempo)
    if diff <= max_jump:
        return 0.0
    return (diff - max_jump) / max_jump


def _energy_penalty(prev_energy: float, cand_energy: float, max_jump: float) -> float:
    diff = abs(prev_energy - cand_energy)
    if diff <= max_jump:
        return 0.0
    return (diff - max_jump) / max_jump


def generate_playlist(
    pool: list[TrackCandidate],
    rules: GenerationRules,
) -> list[TrackCandidate]:
    """Greedy rule-based playlist generation.

    Selects tracks from pool one at a time, scoring each candidate
    by tempo/energy continuity and genre preference.
    """
    if not pool:
        return []

    available = {t.track_id: t for t in pool}
    result: list[TrackCandidate] = []
    total_dur = 0.0

    if rules.seed_track_id is not None and rules.seed_track_id in available:
        seed = available.pop(rules.seed_track_id)
        result.append(seed)
        total_dur += seed.duration_sec

    while len(result) < rules.max_tracks and total_dur < rules.target_duration_sec:
        if not available:
            break

        if not result:
            best_id = min(available, key=lambda tid: abs(available[tid].tempo - 120.0))
            chosen = available.pop(best_id)
            result.append(chosen)
            total_dur += chosen.duration_sec
            continue

        prev = result[-1]
        best_score = -math.inf
        best_id = None

        for tid, cand in available.items():
            if cand.tempo < rules.tempo_range[0] or cand.tempo > rules.tempo_range[1]:
                continue
            if cand.energy < rules.energy_range[0] or cand.energy > rules.energy_range[1]:
                continue

            score = 1.0
            score -= _tempo_penalty(prev.tempo, cand.tempo, rules.max_tempo_jump)
            score -= _energy_penalty(prev.energy, cand.energy, rules.max_energy_jump)

            if rules.genre_weights and cand.genre in rules.genre_weights:
                score += rules.genre_weights[cand.genre]

            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is None:
            break

        chosen = available.pop(best_id)
        result.append(chosen)
        total_dur += chosen.duration_sec

    return result


def fill_gap(
    current: list[TrackCandidate],
    pool: list[TrackCandidate],
    position: int,
    n_candidates: int = 5,
) -> list[TrackCandidate]:
    """Suggest tracks to insert at a given position for smoother flow.

    Returns up to n_candidates sorted by transition quality.
    """
    if position < 0 or position > len(current):
        return []

    used_ids = {t.track_id for t in current}
    candidates = [t for t in pool if t.track_id not in used_ids]

    if not candidates:
        return []

    prev_tempo = current[position - 1].tempo if position > 0 else 120.0
    prev_energy = current[position - 1].energy if position > 0 else 0.5
    next_tempo = current[position].tempo if position < len(current) else prev_tempo
    next_energy = current[position].energy if position < len(current) else prev_energy

    target_tempo = (prev_tempo + next_tempo) / 2
    target_energy = (prev_energy + next_energy) / 2

    def score(t: TrackCandidate) -> float:
        td = abs(t.tempo - target_tempo) / max(target_tempo, 1.0)
        ed = abs(t.energy - target_energy)
        return -(td + ed)

    candidates.sort(key=score, reverse=True)
    return candidates[:n_candidates]


def split_playlist(
    playlist: list[TrackCandidate],
    max_duration_sec: float = 1800.0,
) -> list[list[TrackCandidate]]:
    """Split a long playlist into segments respecting duration limits."""
    segments: list[list[TrackCandidate]] = []
    current: list[TrackCandidate] = []
    current_dur = 0.0

    for track in playlist:
        if current_dur + track.duration_sec > max_duration_sec and current:
            segments.append(current)
            current = []
            current_dur = 0.0
        current.append(track)
        current_dur += track.duration_sec

    if current:
        segments.append(current)
    return segments
