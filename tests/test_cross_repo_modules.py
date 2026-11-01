"""Tests for cross-repo integrated modules: audio, therapeutic, EIF.

Validates both Layer 1 (numpy exact solvers) and Layer 2 (PyTorch
differentiable modules), plus the interface routing through UnifiedSolver
and PipelineComposer.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

passed = 0
failed = 0


def check(condition: bool, name: str) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


# ══════════════════════════════════════════════════════════════════
# Layer 1: AudioSolver (math_core)
# ══════════════════════════════════════════════════════════════════
print("\n=== Layer 1: AudioSolver ===\n")

from models.math_core.audio_solver import AudioSolver, AudioFeatures, CAMELOT_WHEEL

# Spectral centroid
freqs = np.linspace(0, 8000, 100)
mags = np.exp(-(freqs - 2000)**2 / (2 * 500**2))  # peaked at 2000 Hz
centroid = AudioSolver.spectral_centroid(mags, freqs)
check(1500 < centroid < 2500, "Spectral centroid near 2000 Hz")

# Spectral rolloff
rolloff = AudioSolver.spectral_rolloff(mags, freqs, 0.85)
check(rolloff > centroid, "Rolloff above centroid")

# Spectral entropy
H = AudioSolver.spectral_entropy(mags)
check(H > 0, "Spectral entropy positive")

# Band energies
bands = AudioSolver.band_energies(mags, freqs)
check("bass" in bands and "mid" in bands and "treble" in bands, "Band energy keys")
check(bands["mid"] > bands["bass"], "Mid > bass for 2kHz peak")

# Brightness
b = AudioSolver.brightness(2000.0)
check(0.4 < b < 0.6, "Brightness at 2000/4000 = 0.5")

# VAD estimation
v = AudioSolver.estimate_valence(0.5, 0.8)
check(0.0 <= v <= 1.0, "Valence in [0,1]")
check(v == float(np.clip(0.5 * 0.8 * 2, 0, 1)), "Valence formula exact")

a = AudioSolver.estimate_arousal(0.7, 0.6)
check(0.0 <= a <= 1.0, "Arousal in [0,1]")

d = AudioSolver.estimate_dominance(0.6, 0.5)
check(abs(d - 0.3) < 1e-6, "Dominance = energy * brightness")

# Mood scores
moods = AudioSolver.mood_scores(0.5, 0.5, 0.5)
check(set(moods.keys()) == {"happy", "energetic", "calm", "sad", "tense"}, "Mood score keys")
check(abs(moods["calm"] + moods["tense"] - 1.0) < 1e-6, "calm + tense = 1")

# Camelot
check(AudioSolver.key_to_camelot("C", "major") == "8B", "C major → 8B")
check(AudioSolver.key_to_camelot("A", "minor") == "8A", "A minor → 8A")

# Camelot compatibility
check(AudioSolver.camelot_compatibility("8B", "8B") == 1.0, "Same key = 1.0")
check(AudioSolver.camelot_compatibility("8B", "9B") == 0.8, "Adjacent = 0.8")
check(AudioSolver.camelot_compatibility("8B", "8A") == 0.6, "A↔B same number = 0.6")
check(AudioSolver.camelot_compatibility("8B", "3A") == 0.0, "Distant = 0.0")
check(AudioSolver.camelot_compatibility("1B", "12B") == 0.8, "Wrap-around adjacent")

# Pitch class
pc = AudioSolver.frequency_to_pitch_class(440.0)
check(pc == 0, "A4=440 Hz → pitch class 0")

print(f"\nLayer 1 Audio: {passed} passed, {failed} failed")

# ══════════════════════════════════════════════════════════════════
# Layer 1: TherapeuticSolver (math_core)
# ══════════════════════════════════════════════════════════════════
print("\n=== Layer 1: TherapeuticSolver ===\n")

from models.math_core.therapeutic_solver import (
    TherapeuticSolver, GOAL_PROFILES, ENERGY_PROGRESSIONS, VAD_PROGRESSIONS,
    GoalProfile, PhaseSpec, SessionPlan,
)

# Goal profiles
check(len(GOAL_PROFILES) == 6, "6 goal profiles")
focus = GOAL_PROFILES["focus-enhancement"]
check(focus.bpm_min == 78 and focus.bpm_max == 100, "Focus BPM range")
check(focus.valence_target == 0.6, "Focus valence target")

# VAD distance
t_vad = np.array([0.6, 0.7, 0.6])
g_vad = np.array([0.6, 0.7, 0.6])
check(TherapeuticSolver.vad_distance(t_vad, g_vad) < 1e-6, "Zero VAD distance for same point")

t_vad2 = np.array([0.0, 0.0, 0.0])
d2 = TherapeuticSolver.vad_distance(t_vad, t_vad2)
check(d2 > 0.5, "Nonzero VAD distance for different points")

# Track-goal score
score = TherapeuticSolver.track_goal_score(np.array([0.6, 0.7, 0.6]), 85.0, focus)
check(0.9 < score <= 1.0, "Perfect match → high score")

score_off_bpm = TherapeuticSolver.track_goal_score(np.array([0.6, 0.7, 0.6]), 200.0, focus)
check(score_off_bpm < score, "Off-BPM → lower score")

# EIF
eif = TherapeuticSolver.emotional_impact_factor(0.6, 0.7, 0.5, patient_valence=0.3)
check("eif_total" in eif and "eif_patient" in eif, "EIF keys present")
check(eif["eif_patient"] > eif["eif_total"], "Patient adjustment increases EIF when v ≠ v_patient")
check(0.0 <= eif["eif_total"] <= 2.0, "EIF total in reasonable range")

eif_same = TherapeuticSolver.emotional_impact_factor(0.5, 0.5, 0.5, patient_valence=0.5)
check(abs(eif_same["eif_patient"] - eif_same["eif_total"]) < 1e-6, "No adjustment when v = v_patient")

# Phase VAD interpolation
entry = np.array([0.2, 0.8, 0.3])
peak = np.array([0.5, 0.2, 0.6])
resolution = np.array([0.6, 0.3, 0.7])
traj = TherapeuticSolver.phase_vad_interpolation(entry, peak, resolution, 5)
check(traj.shape == (5, 3), "VAD trajectory shape (5,3)")
check(np.allclose(traj[0], entry), "Trajectory starts at entry")
check(np.allclose(traj[-1], resolution), "Trajectory ends at resolution")

# Energy to arousal
check(TherapeuticSolver.energy_to_arousal(1) == 0.0, "Energy 1 → arousal 0")
check(TherapeuticSolver.energy_to_arousal(5) == 1.0, "Energy 5 → arousal 1")
check(TherapeuticSolver.energy_to_arousal(3) == 0.5, "Energy 3 → arousal 0.5")

# Plan session
plan = TherapeuticSolver.plan_session("anxiety-support", 30.0, 5)
check(isinstance(plan, SessionPlan), "Plan is SessionPlan")
check(len(plan.phases) == 5, "5 phases")
check(abs(plan.total_duration_minutes - 30.0) < 1e-6, "Total duration correct")
check(plan.vad_trajectory.shape == (5, 3), "Plan VAD trajectory shape")
total_phase_dur = sum(p.duration_minutes for p in plan.phases)
check(abs(total_phase_dur - 30.0) < 0.1, "Phase durations sum to total")

# Score track for phase
phase = plan.phases[0]
score_p = TherapeuticSolver.score_track_for_phase(
    np.array([0.5, 0.5, 0.5]), 60.0, phase,
)
check(0.0 < score_p <= 1.0, "Phase score in (0,1]")

print(f"\nLayer 1 Therapeutic: {passed} passed, {failed} failed (cumulative)")

# ══════════════════════════════════════════════════════════════════
# Layer 2: PyTorch Audio Module
# ══════════════════════════════════════════════════════════════════
print("\n=== Layer 2: AudioVADEstimator + CamelotEmbedding ===\n")

from models.math.audio_features import AudioVADEstimator, CamelotEmbedding

vad_est = AudioVADEstimator(n_spectral_features=13)
x = torch.randn(4, 13)
out = vad_est(x)
check("valence" in out and "arousal" in out and "dominance" in out, "VAD keys present")
check(out["valence"].shape == (4,), "Valence shape (batch,)")
check((out["valence"] >= 0).all() and (out["valence"] <= 1).all(), "Valence in [0,1]")
check("mood_scores" in out, "Mood scores dict present")
check("happy" in out["mood_scores"], "Happy mood present")

camelot = CamelotEmbedding(embed_dim=16)
keys_a = torch.tensor([0, 1, 2, 3])
keys_b = torch.tensor([0, 1, 5, 10])
emb = camelot(keys_a)
check(emb.shape == (4, 16), "Camelot embedding shape")
compat = camelot.compatibility(keys_a, keys_b)
check(compat.shape == (4,), "Compatibility shape")

# Gradient flow
loss = out["valence"].sum() + compat.sum()
loss.backward()
check(any(p.grad is not None and p.grad.abs().sum() > 0 for p in vad_est.parameters()), "Gradients through VAD estimator")
check(any(p.grad is not None and p.grad.abs().sum() > 0 for p in camelot.parameters()), "Gradients through Camelot")

print(f"\nLayer 2 Audio: {passed} passed, {failed} failed (cumulative)")

# ══════════════════════════════════════════════════════════════════
# Layer 2: PyTorch Therapeutic Module
# ══════════════════════════════════════════════════════════════════
print("\n=== Layer 2: TherapeuticScorer + EIF + SessionPlanner ===\n")

from models.math.therapeutic_session import (
    TherapeuticScorer, EmotionalImpactFactor, SessionPlanner,
)

scorer = TherapeuticScorer(vad_dim=3, n_goals=6)
track_vad = torch.randn(4, 3)
track_bpm = torch.full((4,), 85.0)
goal_idx = torch.tensor([0, 1, 2, 3])
scores = scorer(track_vad, track_bpm, goal_idx)
check(scores.shape == (4,), "Scorer output shape")

eif_mod = EmotionalImpactFactor()
v = torch.tensor([0.6, 0.3, 0.8, 0.1])
ar = torch.tensor([0.7, 0.2, 0.5, 0.9])
dom = torch.tensor([0.5, 0.4, 0.6, 0.3])
pv = torch.tensor([0.5, 0.5, 0.5, 0.5])
eif_out = eif_mod(v, ar, dom, pv)
check("eif_total" in eif_out and "eif_patient" in eif_out, "EIF module keys")
check(eif_out["eif_total"].shape == (4,), "EIF total shape")

planner = SessionPlanner(vad_dim=3, n_goals=6)
init_vad = torch.randn(2, 3)
g_idx = torch.tensor([0, 3])
plan_out = planner(init_vad, g_idx, n_phases=5)
check("vad_trajectory" in plan_out and "phase_scores" in plan_out, "Planner output keys")
check(plan_out["vad_trajectory"].shape == (2, 5, 3), "Planner trajectory shape")
check(plan_out["phase_scores"].shape == (2, 5), "Planner scores shape")

# Gradient flow through all
loss2 = scores.sum() + eif_out["eif_total"].sum() + plan_out["phase_scores"].sum()
loss2.backward()
check(any(p.grad is not None and p.grad.abs().sum() > 0 for p in scorer.parameters()), "Gradients through scorer")
check(any(p.grad is not None and p.grad.abs().sum() > 0 for p in eif_mod.parameters()), "Gradients through EIF")
check(any(p.grad is not None and p.grad.abs().sum() > 0 for p in planner.parameters()), "Gradients through planner")

print(f"\nLayer 2 Therapeutic: {passed} passed, {failed} failed (cumulative)")

# ══════════════════════════════════════════════════════════════════
# Interface: UnifiedSolver routing
# ══════════════════════════════════════════════════════════════════
print("\n=== Interface: UnifiedSolver (new problem types) ===\n")

from models.interface.problem import Problem, ProblemType, Answer
from models.interface.solver import UnifiedSolver

solver = UnifiedSolver()

# Audio VAD
ans = solver.solve(Problem(
    type=ProblemType.AUDIO_VAD_ESTIMATE,
    params={"brightness": 0.5, "rms_energy": 0.8, "rhythm_strength": 0.6},
))
check(isinstance(ans, Answer), "Audio VAD returns Answer")
check("valence" in ans.values, "Audio VAD has valence")
check(0 <= ans.values["valence"] <= 1, "Audio VAD valence in range")

# Audio mood
ans_mood = solver.solve(Problem(
    type=ProblemType.AUDIO_MOOD_SCORES,
    params={"brightness": 0.6, "rms_energy": 0.5, "rhythm_strength": 0.7},
))
check("happy" in ans_mood.values, "Mood scores has 'happy'")

# Camelot compatibility
ans_cam = solver.solve(Problem(
    type=ProblemType.AUDIO_CAMELOT_COMPAT,
    params={"key1": "8B", "key2": "9B"},
))
check(ans_cam.values["compatibility"] == 0.8, "Camelot 8B↔9B = 0.8")

# Therapeutic track score
ans_ts = solver.solve(Problem(
    type=ProblemType.THERAPEUTIC_TRACK_SCORE,
    params={"track_vad": [0.6, 0.7, 0.6], "track_bpm": 85.0},
    query={"goal_id": "focus-enhancement"},
))
check(ans_ts.values["score"] > 0.9, "Perfect focus track → high score")

# Therapeutic EIF
ans_eif = solver.solve(Problem(
    type=ProblemType.THERAPEUTIC_EIF,
    params={"valence": 0.6, "arousal": 0.7, "dominance": 0.5},
))
check("eif_total" in ans_eif.values, "EIF routed correctly")

# Therapeutic session plan
ans_plan = solver.solve(Problem(
    type=ProblemType.THERAPEUTIC_SESSION_PLAN,
    params={"goal_id": "sleep-preparation", "duration_minutes": 60.0},
    query={"n_phases": 5},
))
check(ans_plan.values["n_phases"] == 5, "Plan has 5 phases")
check(ans_plan.values["total_duration"] == 60.0, "Plan duration correct")
check(isinstance(ans_plan.values["vad_trajectory"], np.ndarray), "Plan has VAD trajectory")

# Therapeutic phase score
ans_ph = solver.solve(Problem(
    type=ProblemType.THERAPEUTIC_PHASE_SCORE,
    params={
        "track_vad": [0.5, 0.3, 0.5],
        "track_bpm": 65.0,
        "phase_vad": [0.5, 0.2, 0.6],
        "phase_bpm_range": [50, 70],
    },
))
check(0 < ans_ph.values["score"] <= 1.0, "Phase score in range")

print(f"\nInterface: {passed} passed, {failed} failed (cumulative)")

# ══════════════════════════════════════════════════════════════════
# Pipeline: therapeutic_session composer
# ══════════════════════════════════════════════════════════════════
print("\n=== Pipeline: therapeutic_session ===\n")

from models.interface.composer import PipelineComposer

composer = PipelineComposer()
result = composer.therapeutic_session(
    patient_vad=np.array([0.3, 0.7, 0.4]),
    goal_id="anxiety-support",
    duration_minutes=30.0,
)
check("plan" in result and "eif" in result and "binaural_targets" in result, "Pipeline output keys")
check(isinstance(result["plan"], Answer), "Plan is Answer")
check(isinstance(result["eif"], Answer), "EIF is Answer")
check(len(result["binaural_targets"]) == 5, "5 binaural targets (one per phase)")
for bt in result["binaural_targets"]:
    check("f_beat_hz" in bt.values, "Binaural target has f_beat")

print(f"\nPipeline: {passed} passed, {failed} failed (cumulative)")

# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"TOTAL: {passed} passed, {failed} failed")
print(f"{'='*50}")

if failed > 0:
    sys.exit(1)
