from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.music_playlist.transition import TransitionScorer, TransitionScore
from models.music_playlist.energy_curve import EnergyCurve, EnergyCurveShape, EnergyCurveDesigner


@dataclass
class TrackSlot:
    index: int
    track: dict[str, Any]
    target_energy: float
    actual_energy: float
    transition_from_prev: TransitionScore | None = None


@dataclass
class PlaylistConstraints:
    n_tracks: int = 10
    energy_shape: EnergyCurveShape = EnergyCurveShape.PEAK
    min_energy: float = 0.2
    max_energy: float = 1.0
    min_transition_score: float = 0.3
    max_bpm_jump: float = 30.0
    target_duration_min: float | None = None
    avoid_same_artist: bool = True
    key_lock: str | None = None


class PlaylistOptimizer:
    def __init__(self, scorer: TransitionScorer | None = None):
        self.scorer = scorer or TransitionScorer()
        self.curve_designer = EnergyCurveDesigner()

    def optimize(
        self,
        candidates: list[dict[str, Any]],
        constraints: PlaylistConstraints,
        seed: int | None = None,
    ) -> list[TrackSlot]:
        if len(candidates) == 0:
            return []

        rng = np.random.default_rng(seed)
        n = min(constraints.n_tracks, len(candidates))

        curve = self.curve_designer.design(
            constraints.energy_shape, n,
            constraints.min_energy, constraints.max_energy,
        )

        selected = self._greedy_select(candidates, curve, constraints, rng)
        return selected

    def _greedy_select(
        self,
        candidates: list[dict[str, Any]],
        curve: EnergyCurve,
        constraints: PlaylistConstraints,
        rng: np.random.Generator,
    ) -> list[TrackSlot]:
        n = curve.n_slots
        used_indices: set[int] = set()
        slots: list[TrackSlot] = []

        for slot_idx in range(n):
            target_e = curve.at(slot_idx)
            lo, hi = curve.tolerance_at(slot_idx, 0.2)
            best_idx = -1
            best_score = -1.0

            for ci, cand in enumerate(candidates):
                if ci in used_indices:
                    continue

                cand_energy = float(cand.get("energy", 0.5))
                if cand_energy < lo - 0.1 or cand_energy > hi + 0.1:
                    continue

                if constraints.avoid_same_artist and slots:
                    last_artist = slots[-1].track.get("artist", "")
                    if last_artist and cand.get("artist", "") == last_artist:
                        continue

                if constraints.key_lock and slot_idx == 0:
                    if cand.get("key", "") != constraints.key_lock:
                        continue

                energy_fit = 1.0 - abs(cand_energy - target_e)

                if slots:
                    bpm_prev = float(slots[-1].track.get("bpm", 120))
                    bpm_cand = float(cand.get("bpm", 120))
                    if abs(bpm_prev - bpm_cand) > constraints.max_bpm_jump:
                        if abs(bpm_prev - bpm_cand * 2) > constraints.max_bpm_jump and \
                           abs(bpm_prev * 2 - bpm_cand) > constraints.max_bpm_jump:
                            continue

                    ts = self.scorer.score(slots[-1].track, cand)
                    if ts.overall < constraints.min_transition_score:
                        continue
                    combined = 0.6 * ts.overall + 0.4 * energy_fit
                else:
                    ts = None
                    combined = energy_fit

                if combined > best_score:
                    best_score = combined
                    best_idx = ci

            if best_idx < 0:
                for ci, cand in enumerate(candidates):
                    if ci not in used_indices:
                        best_idx = ci
                        break
                if best_idx < 0:
                    break

            chosen = candidates[best_idx]
            used_indices.add(best_idx)

            transition = None
            if slots:
                transition = self.scorer.score(slots[-1].track, chosen)

            slot = TrackSlot(
                index=slot_idx,
                track=chosen,
                target_energy=target_e,
                actual_energy=float(chosen.get("energy", 0.5)),
                transition_from_prev=transition,
            )
            slots.append(slot)

        return slots

    def evaluate(self, slots: list[TrackSlot]) -> dict[str, float]:
        if not slots:
            return {"overall": 0.0, "energy_fit": 0.0, "transition_avg": 0.0, "flow": 0.0}

        energy_diffs = [abs(s.actual_energy - s.target_energy) for s in slots]
        energy_fit = 1.0 - np.mean(energy_diffs)

        transitions = [s.transition_from_prev.overall for s in slots if s.transition_from_prev]
        transition_avg = float(np.mean(transitions)) if transitions else 0.5

        if len(slots) > 2:
            actual_energies = [s.actual_energy for s in slots]
            target_energies = [s.target_energy for s in slots]
            flow = float(np.corrcoef(actual_energies, target_energies)[0, 1])
            if not np.isfinite(flow):
                flow = 0.0
            flow = max(0.0, flow)
        else:
            flow = energy_fit

        overall = 0.4 * energy_fit + 0.35 * transition_avg + 0.25 * flow
        return {
            "overall": float(np.clip(overall, 0, 1)),
            "energy_fit": float(np.clip(energy_fit, 0, 1)),
            "transition_avg": float(np.clip(transition_avg, 0, 1)),
            "flow": float(np.clip(flow, 0, 1)),
        }
