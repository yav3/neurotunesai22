"""Constraint-based improvisation: generate melodic lines satisfying simultaneous musical constraints.

Treats note selection as a Constraint Satisfaction Problem (CSP):
- Domain: available pitches at each beat position
- Constraints: chord-tone alignment, interval limits, voice-leading smoothness,
  contour shape, range bounds, rhythmic density, phrase structure
- Solver: backtracking with forward-checking and domain reduction
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Set
from enum import Enum


class ConstraintType(Enum):
    CHORD_TONE = "chord_tone"
    INTERVAL_LIMIT = "interval_limit"
    RANGE = "range"
    CONTOUR = "contour"
    AVOID_REPETITION = "avoid_repetition"
    STEP_PREFERENCE = "step_preference"
    RESOLUTION = "resolution"
    DENSITY = "density"


@dataclass
class HarmonicContext:
    root: int
    chord_tones: List[int]
    scale_tones: List[int]
    avoid_notes: List[int] = field(default_factory=list)
    beat_position: float = 0.0
    is_strong_beat: bool = False


@dataclass
class MelodicConstraint:
    constraint_type: ConstraintType
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class ImprovConfig:
    min_pitch: int = 48
    max_pitch: int = 84
    max_interval: int = 12
    prefer_stepwise_pct: float = 0.6
    chord_tone_on_strong: bool = True
    avoid_repetition_window: int = 3
    contour_target: Optional[np.ndarray] = None
    resolution_targets: Optional[List[int]] = None


def _pitch_classes_in_range(pcs: List[int], lo: int, hi: int) -> List[int]:
    result = []
    for pc in pcs:
        p = lo + ((pc - lo) % 12)
        while p <= hi:
            result.append(p)
            p += 12
    return sorted(set(result))


def build_domains(contexts: List[HarmonicContext],
                  config: ImprovConfig) -> List[List[int]]:
    domains = []
    for ctx in contexts:
        if ctx.is_strong_beat and config.chord_tone_on_strong:
            pcs = ctx.chord_tones
        else:
            pcs = ctx.scale_tones
        available = _pitch_classes_in_range(pcs, config.min_pitch, config.max_pitch)
        for avoid_pc in ctx.avoid_notes:
            available = [p for p in available if p % 12 != avoid_pc % 12]
        domains.append(available)
    return domains


def _interval_ok(prev: int, curr: int, max_interval: int) -> bool:
    return abs(curr - prev) <= max_interval


def _is_stepwise(prev: int, curr: int) -> bool:
    return abs(curr - prev) <= 2


def _repetition_ok(history: List[int], candidate: int, window: int) -> bool:
    recent = history[-window:] if len(history) >= window else history
    return candidate not in recent


def _contour_score(position: int, pitch: int, target_contour: np.ndarray,
                   min_pitch: int, max_pitch: int) -> float:
    if position >= len(target_contour):
        return 1.0
    target_norm = target_contour[position]
    actual_norm = (pitch - min_pitch) / max(1, max_pitch - min_pitch)
    return 1.0 - abs(target_norm - actual_norm)


def forward_check(domains: List[List[int]], assigned: List[int],
                  config: ImprovConfig) -> List[List[int]]:
    if not assigned:
        return domains
    last = assigned[-1]
    idx = len(assigned)
    if idx >= len(domains):
        return domains
    pruned = list(domains)
    new_domain = [p for p in domains[idx]
                  if _interval_ok(last, p, config.max_interval)]
    if config.avoid_repetition_window > 0:
        new_domain = [p for p in new_domain
                      if _repetition_ok(assigned, p, config.avoid_repetition_window)]
    pruned[idx] = new_domain if new_domain else domains[idx]
    return pruned


def score_candidate(candidate: int, position: int, assigned: List[int],
                    context: HarmonicContext, config: ImprovConfig) -> float:
    score = 0.0
    if candidate % 12 in [ct % 12 for ct in context.chord_tones]:
        score += 2.0
    elif candidate % 12 in [st % 12 for st in context.scale_tones]:
        score += 1.0
    if assigned:
        if _is_stepwise(assigned[-1], candidate):
            score += config.prefer_stepwise_pct * 2.0
        interval = abs(candidate - assigned[-1])
        score -= interval * 0.05
    if config.contour_target is not None:
        score += _contour_score(position, candidate, config.contour_target,
                                config.min_pitch, config.max_pitch) * 1.5
    if config.resolution_targets and position == len(config.resolution_targets) - 1:
        for target in config.resolution_targets:
            if abs(candidate - target) <= 2:
                score += 3.0
                break
    return score


def solve_melody(contexts: List[HarmonicContext],
                 config: ImprovConfig,
                 max_backtracks: int = 1000) -> Optional[np.ndarray]:
    domains = build_domains(contexts, config)
    n = len(contexts)
    assigned: List[int] = []
    backtracks = 0

    def _solve(idx: int) -> bool:
        nonlocal backtracks
        if idx == n:
            return True
        pruned = forward_check(domains, assigned, config)
        candidates = pruned[idx]
        scored = [(score_candidate(c, idx, assigned, contexts[idx], config), c)
                  for c in candidates]
        scored.sort(key=lambda x: -x[0])
        for _, candidate in scored:
            assigned.append(candidate)
            if _solve(idx + 1):
                return True
            assigned.pop()
            backtracks += 1
            if backtracks >= max_backtracks:
                return False
        return False

    if _solve(0):
        return np.array(assigned)
    if assigned:
        while len(assigned) < n:
            assigned.append(assigned[-1] if assigned else config.min_pitch)
        return np.array(assigned)
    return None


def greedy_melody(contexts: List[HarmonicContext],
                  config: ImprovConfig) -> np.ndarray:
    domains = build_domains(contexts, config)
    assigned: List[int] = []
    for i, ctx in enumerate(contexts):
        candidates = domains[i]
        if assigned:
            candidates = [c for c in candidates
                          if _interval_ok(assigned[-1], c, config.max_interval)]
            if not candidates:
                candidates = domains[i]
        scored = [(score_candidate(c, i, assigned, ctx, config), c)
                  for c in candidates]
        scored.sort(key=lambda x: -x[0])
        if scored:
            assigned.append(scored[0][1])
        else:
            mid = (config.min_pitch + config.max_pitch) // 2
            assigned.append(mid)
    return np.array(assigned)


def phrase_structure_penalty(melody: np.ndarray, phrase_length: int = 8) -> float:
    n = len(melody)
    if n < phrase_length:
        return 0.0
    penalties = 0.0
    n_phrases = n // phrase_length
    for i in range(n_phrases):
        start = i * phrase_length
        phrase = melody[start:start + phrase_length]
        intervals = np.diff(phrase)
        if len(intervals) < 2:
            continue
        if abs(intervals[-1]) > 4:
            penalties += 1.0
        if phrase[-1] == phrase[-2]:
            penalties += 0.5
    return penalties / max(1, n_phrases)


def melodic_tension_curve(melody: np.ndarray, contexts: List[HarmonicContext]) -> np.ndarray:
    n = min(len(melody), len(contexts))
    tension = np.zeros(n)
    for i in range(n):
        pc = int(melody[i]) % 12
        ctx = contexts[i]
        chord_pcs = set(ct % 12 for ct in ctx.chord_tones)
        scale_pcs = set(st % 12 for st in ctx.scale_tones)
        if pc in chord_pcs:
            tension[i] = 0.0
        elif pc in scale_pcs:
            tension[i] = 0.3
        else:
            tension[i] = 0.8
        if i > 0:
            interval = abs(int(melody[i]) - int(melody[i - 1]))
            tension[i] += min(0.5, interval * 0.04)
    return tension


def evaluate_melody(melody: np.ndarray, contexts: List[HarmonicContext],
                    config: ImprovConfig) -> Dict:
    n = min(len(melody), len(contexts))
    intervals = np.abs(np.diff(melody[:n]))
    stepwise_count = int(np.sum(intervals <= 2))
    stepwise_ratio = stepwise_count / max(1, len(intervals))
    chord_tone_hits = 0
    scale_tone_hits = 0
    for i in range(n):
        pc = int(melody[i]) % 12
        chord_pcs = set(ct % 12 for ct in contexts[i].chord_tones)
        scale_pcs = set(st % 12 for st in contexts[i].scale_tones)
        if pc in chord_pcs:
            chord_tone_hits += 1
        elif pc in scale_pcs:
            scale_tone_hits += 1
    tension = melodic_tension_curve(melody[:n], contexts[:n])
    unique_pcs = len(set(int(p) % 12 for p in melody[:n]))
    return {
        "length": n,
        "pitch_range": int(np.max(melody[:n]) - np.min(melody[:n])),
        "mean_interval": float(np.mean(intervals)) if len(intervals) > 0 else 0.0,
        "max_interval": int(np.max(intervals)) if len(intervals) > 0 else 0,
        "stepwise_ratio": stepwise_ratio,
        "chord_tone_ratio": chord_tone_hits / max(1, n),
        "scale_tone_ratio": (chord_tone_hits + scale_tone_hits) / max(1, n),
        "unique_pitch_classes": unique_pcs,
        "mean_tension": float(np.mean(tension)),
        "tension_variance": float(np.var(tension)),
        "phrase_penalty": phrase_structure_penalty(melody[:n]),
    }


from typing import Dict
