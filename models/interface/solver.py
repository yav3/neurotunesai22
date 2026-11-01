"""Unified solver — routes Problems to the appropriate math_core solver
and returns structured Answers.

This is the primary interface: pose a problem, get an answer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.interface.problem import Problem, ProblemType, Answer
from models.math_core.pk_solver import (
    PKSolver, PKParams,
    StructuralPKPDParams, DoseEvent, simulate_structural_pkpd, toxicity_hazard,
)
from models.math_core.pd_solver import PDSolver, PDParams
from models.math_core.progression_solver import (
    ProgressionSolver, LinearProgressionParams, EpisodicParams, EventParams,
)
from models.math_core.vade_geometry import VADEGeometry
from models.math_core.spectral import SpectralSolver
from models.math_core.audio_solver import AudioSolver
from models.math_core.therapeutic_solver import (
    TherapeuticSolver, GOAL_PROFILES, PhaseSpec,
)
from models.math_core.eif_solver import eif_from_parameters, eif_spreadsheet, recommend
from models.math_core.entrainment_solver import (
    phase_locking_value, inter_trial_coherence,
    frequency_following_response, beat_alignment,
)
from models.math_core.cns_solver import (
    gpcr_rtms_modulation, cns_synchronization_distance, affect_purity,
)
from models.math_core.rtp_solver import (
    dominance_factor, evolution_of_internal_component,
    simulate_relationships, eliminate_drivers, test_drivers, identify_target,
)
from models.math_core.drift_solver import DriftEstimator
from models.math_core.gpcr_solver import compute_bias
from models.math_core.grn_solver import (
    GRNSolver, GRNNode, GRNEdge,
    propagate_canonical, compute_dominance_scores,
    sensitivity_analysis, network_motifs,
)


SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic_data" / "schemas"


class UnifiedSolver:
    """Route any Problem to the correct exact solver."""

    def __init__(self):
        self._drug_db = self._load_drugs()

    def _load_drugs(self) -> dict:
        path = SCHEMA_DIR / "drug_parameters.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data.get("drugs", {})
        return {}

    def solve(self, problem: Problem) -> Answer:
        dispatch = {
            ProblemType.PK_CONCENTRATION: self._pk_concentration,
            ProblemType.PK_STEADY_STATE: self._pk_steady_state,
            ProblemType.PK_MULTI_DOSE: self._pk_multi_dose,
            ProblemType.PD_EFFECT: self._pd_effect,
            ProblemType.PD_BIOMARKER: self._pd_biomarker,
            ProblemType.PD_INVERSE: self._pd_inverse,
            ProblemType.PKPD_TRAJECTORY: self._pkpd_trajectory,
            ProblemType.PROGRESSION_LINEAR: self._progression_linear,
            ProblemType.PROGRESSION_EPISODIC: self._progression_episodic,
            ProblemType.VADE_PROJECT: self._vade_project,
            ProblemType.VADE_GEODESIC: self._vade_geodesic,
            ProblemType.VADE_TRAJECTORY: self._vade_trajectory,
            ProblemType.VADE_SCHEDULE: self._vade_schedule,
            ProblemType.SPECTRAL_BAND_POWER: self._spectral_band_power,
            ProblemType.SPECTRAL_ENTROPY: self._spectral_entropy,
            ProblemType.SPECTRAL_THERAPEUTIC: self._spectral_therapeutic,
            ProblemType.BINAURAL_BEAT: self._binaural_beat,
            ProblemType.AUDIO_VAD_ESTIMATE: self._audio_vad,
            ProblemType.AUDIO_MOOD_SCORES: self._audio_mood,
            ProblemType.AUDIO_CAMELOT_COMPAT: self._audio_camelot,
            ProblemType.THERAPEUTIC_TRACK_SCORE: self._therapeutic_track_score,
            ProblemType.THERAPEUTIC_EIF: self._therapeutic_eif,
            ProblemType.THERAPEUTIC_SESSION_PLAN: self._therapeutic_session_plan,
            ProblemType.THERAPEUTIC_PHASE_SCORE: self._therapeutic_phase_score,
            # EIF
            ProblemType.EIF_FROM_PARAMETERS: self._eif_from_parameters,
            ProblemType.EIF_SPREADSHEET: self._eif_spreadsheet,
            ProblemType.EIF_RECOMMEND: self._eif_recommend,
            # Neural entrainment
            ProblemType.ENTRAINMENT_PLV: self._entrainment_plv,
            ProblemType.ENTRAINMENT_ITC: self._entrainment_itc,
            ProblemType.ENTRAINMENT_FFR: self._entrainment_ffr,
            ProblemType.ENTRAINMENT_BEAT: self._entrainment_beat,
            # CNS pharmacodynamics
            ProblemType.CNS_RTMS: self._cns_rtms,
            ProblemType.CNS_SYNC_DISTANCE: self._cns_sync_distance,
            ProblemType.CNS_AFFECT_PURITY: self._cns_affect_purity,
            # RTP kernel (patent claims)
            ProblemType.RTP_DOMINANCE: self._rtp_dominance,
            ProblemType.RTP_EVOLUTION: self._rtp_evolution,
            ProblemType.RTP_SIMULATE: self._rtp_simulate,
            ProblemType.RTP_IDENTIFY: self._rtp_identify,
            # VADE drift
            ProblemType.DRIFT_MONITOR: self._drift_monitor,
            # GPCR bias
            ProblemType.GPCR_BIAS: self._gpcr_bias,
            # GRN canonical
            ProblemType.GRN_PROPAGATE: self._grn_propagate,
            ProblemType.GRN_DOMINANCE: self._grn_dominance,
            ProblemType.GRN_SENSITIVITY: self._grn_sensitivity,
            ProblemType.GRN_MOTIFS: self._grn_motifs,
            # Structural PK/PD
            ProblemType.PKPD_STRUCTURAL: self._pkpd_structural,
            ProblemType.PKPD_TOXICITY: self._pkpd_toxicity,
            # GRN D/C/E dynamics
            ProblemType.GRN_SIMULATE: self._grn_simulate,
            ProblemType.GRN_IDENTIFY: self._grn_identify,
            # Disease progression (remaining)
            ProblemType.PROGRESSION_EVENT: self._progression_event,
            ProblemType.MCI_CONVERSION: self._mci_conversion,
            # Composite
            ProblemType.TREATMENT_PLAN: self._treatment_plan,
            ProblemType.DISEASE_FORECAST: self._disease_forecast,
        }
        fn = dispatch.get(problem.type)
        if fn is None:
            raise ValueError(f"No solver for {problem.type}")
        return fn(problem)

    # ── PK ────────────────────────────────────────────────────────────

    def _build_pk_params(self, p: Problem) -> PKParams:
        drug_name = p.params.get("drug", "")
        drug = self._drug_db.get(drug_name, {})
        pk = drug.get("pk_parameters", {})
        weight = p.params.get("weight_kg", pk.get("typical_body_weight_kg", 70))
        dose = p.params.get("dose_mg", drug.get("dosing", {}).get("amount_mg", 10))
        route = drug.get("route", p.params.get("route", "oral"))
        freq = drug.get("dosing", {}).get("doses_per_day", 1)
        tau = 24.0 / freq if freq > 0 else 24.0

        return PKParams(
            ka_per_hr=pk.get("ka_per_hr", 0.5),
            ke_per_hr=pk.get("ke_per_hr", 0.01),
            vd_l=pk.get("vd_l_per_kg", 10.0) * weight,
            bioavailability=pk.get("bioavailability", 1.0),
            dose_mg=dose,
            route=route,
            tau_hr=tau,
        )

    def _pk_concentration(self, p: Problem) -> Answer:
        pk = self._build_pk_params(p)
        t = np.array([p.query.get("time_hr", 24.0)])
        C = PKSolver.concentration_curve(pk, t)
        return Answer(
            values={"concentration_ng_ml": float(C[0])},
            problem_type=p.type,
        )

    def _pk_steady_state(self, p: Problem) -> Answer:
        pk = self._build_pk_params(p)
        css = PKSolver.steady_state_trough(pk)
        return Answer(
            values={"trough_ng_ml": css},
            problem_type=p.type,
        )

    def _pk_multi_dose(self, p: Problem) -> Answer:
        pk = self._build_pk_params(p)
        t_max = p.query.get("duration_hr", 168.0)
        n_doses = p.query.get("n_doses", int(t_max / pk.tau_hr) + 1)
        t = np.linspace(0, t_max, 500)
        C = PKSolver.multi_dose_oral(pk, t, n_doses)
        return Answer(
            values={
                "time_hr": t, "concentration_ng_ml": C,
                "peak_ng_ml": float(np.max(C)),
                "trough_ng_ml": float(C[-1]),
            },
            problem_type=p.type,
        )

    # ── PD ────────────────────────────────────────────────────────────

    def _build_pd_params(self, p: Problem) -> PDParams:
        drug = self._drug_db.get(p.params.get("drug", ""), {})
        pd = drug.get("pd_parameters", {})
        return PDParams(
            emax=pd.get("emax", 1.0),
            ec50_ng_ml=pd.get("ec50_ng_ml", 10.0),
            hill=pd.get("hill_coefficient", 1.0),
            direction=drug.get("pd_direction", "increase"),
            baseline=p.params.get("baseline", 0.0),
        )

    def _pd_effect(self, p: Problem) -> Answer:
        pd = self._build_pd_params(p)
        C = np.array([p.query.get("concentration_ng_ml", 10.0)])
        E = PDSolver.effect(C, pd)
        return Answer(values={"effect": float(E[0])}, problem_type=p.type)

    def _pd_biomarker(self, p: Problem) -> Answer:
        pd = self._build_pd_params(p)
        C = np.array([p.query.get("concentration_ng_ml", 10.0)])
        B = PDSolver.biomarker_response(C, pd)
        return Answer(values={"biomarker": float(B[0])}, problem_type=p.type)

    def _pd_inverse(self, p: Problem) -> Answer:
        pd = self._build_pd_params(p)
        target = p.query.get("target_effect", 1.0)
        C = PDSolver.inverse_effect(target, pd)
        return Answer(values={"required_concentration_ng_ml": C}, problem_type=p.type)

    def _pkpd_trajectory(self, p: Problem) -> Answer:
        pk = self._build_pk_params(p)
        pd = self._build_pd_params(p)
        t_max = p.query.get("duration_hr", 168.0)
        n_doses = p.query.get("n_doses", 7)
        t = np.linspace(0, t_max, 500)
        result = PDSolver.pkpd_trajectory(pk, pd, t, n_doses)
        return Answer(values=result, problem_type=p.type)

    # ── Progression ───────────────────────────────────────────────────

    def _progression_linear(self, p: Problem) -> Answer:
        params = LinearProgressionParams(
            baseline=p.params.get("baseline", 29.0),
            annual_rate=p.params.get("annual_rate", -2.5),
            rate_std=p.params.get("rate_std", 1.0),
            noise_std=p.params.get("noise_std", 0.5),
        )
        months = np.arange(0, p.query.get("months", 60) + 1, p.query.get("interval", 6))
        trajectory = ProgressionSolver.linear_trajectory(params, months)
        return Answer(
            values={"time_months": months, "values": trajectory},
            problem_type=p.type,
        )

    def _progression_episodic(self, p: Problem) -> Answer:
        params = EpisodicParams(
            relapse_prob_per_year=p.params.get("relapse_prob", 0.5),
            mean_episode_months=p.params.get("episode_months", 6),
            std_episode_months=p.params.get("episode_std", 3),
            remission_target=p.params.get("remission_target", 6.0),
            episode_target=p.params.get("episode_target", 22.0),
        )
        months = np.arange(0, p.query.get("months", 48) + 1)
        trajectory = ProgressionSolver.episodic_trajectory(params, months)
        return Answer(
            values={"time_months": months, "values": trajectory},
            problem_type=p.type,
        )

    # ── VADE ──────────────────────────────────────────────────────────

    def _vade_project(self, p: Problem) -> Answer:
        X = np.array(p.params.get("features", [[0.5, 0.3, 0.7, 0.4, 0.6]]))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        d = X.shape[1]
        W = np.array(p.params.get("W", np.eye(4, d)))
        b = np.array(p.params.get("b", np.zeros(4)))
        projected = VADEGeometry.project_to_vade(X, W, b)
        valid = VADEGeometry.check_valence_constraint(projected)
        return Answer(
            values={"vade_point": projected, "constraint_satisfied": valid},
            problem_type=p.type,
        )

    def _vade_geodesic(self, p: Problem) -> Answer:
        p_pt = np.array(p.params["point"])
        q_pt = np.array(p.query["target"])
        d = VADEGeometry.geodesic_distance(p_pt, q_pt)
        return Answer(values={"geodesic_distance": float(d)}, problem_type=p.type)

    def _vade_trajectory(self, p: Problem) -> Answer:
        start = np.array(p.params["start"])
        target = np.array(p.params["target"])
        n = p.query.get("n_phases", 5)
        traj = VADEGeometry.session_trajectory(start, target, n)
        return Answer(values={"trajectory": traj}, problem_type=p.type)

    def _vade_schedule(self, p: Problem) -> Answer:
        base = p.params.get("base_minutes", 5.0)
        max_dur = p.params.get("max_duration", 90.0)
        sched = VADEGeometry.fibonacci_schedule(base, max_dur)
        return Answer(
            values={"phases_minutes": np.array(sched), "total_minutes": sum(sched)},
            problem_type=p.type,
        )

    # ── Spectral ──────────────────────────────────────────────────────

    def _spectral_band_power(self, p: Problem) -> Answer:
        psd = np.array(p.params["psd"])
        freqs = np.array(p.params["freqs"])
        powers = SpectralSolver.all_band_powers(psd, freqs)
        powers["theta_alpha_ratio"] = SpectralSolver.theta_alpha_ratio(powers)
        return Answer(values=powers, problem_type=p.type)

    def _spectral_entropy(self, p: Problem) -> Answer:
        psd = np.array(p.params["psd"])
        H = SpectralSolver.spectral_entropy(psd)
        return Answer(values={"entropy": H}, problem_type=p.type)

    def _spectral_therapeutic(self, p: Problem) -> Answer:
        powers = p.params.get("powers", {})
        goal = p.query.get("goal", "relaxation")
        score = SpectralSolver.therapeutic_score(powers, goal)
        return Answer(values={"score": score, "goal": goal}, problem_type=p.type)

    def _binaural_beat(self, p: Problem) -> Answer:
        band = p.query.get("target_band", "alpha")
        f_left, f_right, f_beat = SpectralSolver.binaural_beat_frequency(band)
        return Answer(
            values={"f_left_hz": f_left, "f_right_hz": f_right, "f_beat_hz": f_beat},
            problem_type=p.type,
        )

    # ── Audio ─────────────────────────────────────────────────────────

    def _audio_vad(self, p: Problem) -> Answer:
        brightness = p.params.get("brightness", 0.5)
        rms_energy = p.params.get("rms_energy", 0.5)
        rhythm_strength = p.params.get("rhythm_strength", 0.5)
        v = AudioSolver.estimate_valence(brightness, rms_energy)
        a = AudioSolver.estimate_arousal(rhythm_strength, rms_energy)
        d = AudioSolver.estimate_dominance(rms_energy, brightness)
        return Answer(values={"valence": v, "arousal": a, "dominance": d}, problem_type=p.type)

    def _audio_mood(self, p: Problem) -> Answer:
        brightness = p.params.get("brightness", 0.5)
        rms_energy = p.params.get("rms_energy", 0.5)
        rhythm_strength = p.params.get("rhythm_strength", 0.5)
        scores = AudioSolver.mood_scores(brightness, rms_energy, rhythm_strength)
        return Answer(values=scores, problem_type=p.type)

    def _audio_camelot(self, p: Problem) -> Answer:
        key1 = p.params.get("key1", "8B")
        key2 = p.params.get("key2", "8B")
        score = AudioSolver.camelot_compatibility(key1, key2)
        return Answer(values={"compatibility": score}, problem_type=p.type)

    # ── Therapeutic ───────────────────────────────────────────────────

    def _therapeutic_track_score(self, p: Problem) -> Answer:
        track_vad = np.array(p.params.get("track_vad", [0.5, 0.5, 0.5]))
        track_bpm = p.params.get("track_bpm", 100.0)
        goal_id = p.query.get("goal_id", "focus-enhancement")
        goal = GOAL_PROFILES.get(goal_id)
        if goal is None:
            return Answer(values={"score": 0.0}, problem_type=p.type)
        score = TherapeuticSolver.track_goal_score(track_vad, track_bpm, goal)
        return Answer(values={"score": score}, problem_type=p.type)

    def _therapeutic_eif(self, p: Problem) -> Answer:
        v = p.params.get("valence", 0.5)
        a = p.params.get("arousal", 0.5)
        d = p.params.get("dominance", 0.5)
        pv = p.params.get("patient_valence", 0.5)
        eif = TherapeuticSolver.emotional_impact_factor(v, a, d, pv)
        return Answer(values=eif, problem_type=p.type)

    def _therapeutic_session_plan(self, p: Problem) -> Answer:
        goal_id = p.params.get("goal_id", "focus-enhancement")
        duration = p.params.get("duration_minutes", 30.0)
        n_phases = p.query.get("n_phases", 5)
        plan = TherapeuticSolver.plan_session(goal_id, duration, n_phases)
        return Answer(
            values={
                "vad_trajectory": plan.vad_trajectory,
                "total_duration": plan.total_duration_minutes,
                "n_phases": len(plan.phases),
                "phase_names": [ph.name for ph in plan.phases],
            },
            problem_type=p.type,
        )

    def _therapeutic_phase_score(self, p: Problem) -> Answer:
        track_vad = np.array(p.params.get("track_vad", [0.5, 0.5, 0.5]))
        track_bpm = p.params.get("track_bpm", 100.0)
        phase_vad = np.array(p.params.get("phase_vad", [0.5, 0.5, 0.5]))
        phase_bpm = tuple(p.params.get("phase_bpm_range", [60, 100]))
        phase = PhaseSpec(name="query", duration_minutes=5.0, vad_target=phase_vad, bpm_range=phase_bpm)
        score = TherapeuticSolver.score_track_for_phase(track_vad, track_bpm, phase)
        return Answer(values={"score": score}, problem_type=p.type)

    # ── EIF ──────────────────────────────────────────────────────────

    def _eif_from_parameters(self, p: Problem) -> Answer:
        params = p.params.get("parameters", {})
        audience = p.params.get("audience", "general")
        result = eif_from_parameters(params, audience)
        return Answer(values=result, problem_type=p.type)

    def _eif_spreadsheet(self, p: Problem) -> Answer:
        result = eif_spreadsheet(
            energy=p.params.get("energy", 50),
            valence=p.params.get("valence", 50),
            tempo=p.params.get("tempo", 120.0),
            loudness_db=p.params.get("loudness_db", -10.0),
            acousticness=p.params.get("acousticness", 0.5),
        )
        return Answer(values=result, problem_type=p.type)

    def _eif_recommend(self, p: Problem) -> Answer:
        goal = p.query.get("goal", "relaxation")
        energy = p.params.get("energy", 50)
        valence = p.params.get("valence", 50)
        result = recommend(goal, energy, valence)
        return Answer(values=result, problem_type=p.type)

    # ── Entrainment ──────────────────────────────────────────────────

    def _entrainment_plv(self, p: Problem) -> Answer:
        s1 = np.array(p.params["signal1"])
        s2 = np.array(p.params["signal2"])
        result = phase_locking_value(s1, s2)
        return Answer(values=result, problem_type=p.type)

    def _entrainment_itc(self, p: Problem) -> Answer:
        trials = np.array(p.params["trials"])
        fs = p.params.get("fs", 256.0)
        result = inter_trial_coherence(trials, fs)
        return Answer(values=result, problem_type=p.type)

    def _entrainment_ffr(self, p: Problem) -> Answer:
        eeg = np.array(p.params["eeg"])
        stim_freq = p.params["stimulus_freq"]
        fs = p.params.get("fs", 256.0)
        bw = p.params.get("bandwidth", 2.0)
        result = frequency_following_response(eeg, stim_freq, fs, bw)
        return Answer(values=result, problem_type=p.type)

    def _entrainment_beat(self, p: Problem) -> Answer:
        signal = np.array(p.params["neural_signal"])
        beat_times = np.array(p.params["beat_times"])
        fs = p.params.get("fs", 256.0)
        result = beat_alignment(signal, beat_times, fs)
        return Answer(values=result, problem_type=p.type)

    # ── CNS pharmacodynamics ─────────────────────────────────────────

    def _cns_rtms(self, p: Problem) -> Answer:
        result = gpcr_rtms_modulation(
            g0=p.params.get("g0", 1.0),
            field_strength=p.params.get("field_strength", 1.5),
            f_rtms=p.params.get("f_rtms", 10.0),
            t_stim=p.params.get("t_stim", 20.0),
        )
        return Answer(values=result, problem_type=p.type)

    def _cns_sync_distance(self, p: Problem) -> Answer:
        result = cns_synchronization_distance(
            velocity=p.params.get("velocity", 4.0),
            frequency=p.params.get("frequency", 10.0),
            delta_phi=p.params.get("delta_phi", 0.0),
            delta_t=p.params.get("delta_t", 0.01),
            coherence=p.params.get("coherence", 0.8),
        )
        return Answer(values=result, problem_type=p.type)

    def _cns_affect_purity(self, p: Problem) -> Answer:
        result = affect_purity(
            valence=p.params.get("valence", 0.5),
            arousal=p.params.get("arousal", 0.5),
            dominance=p.params.get("dominance", 0.5),
        )
        return Answer(values=result, problem_type=p.type)

    # ── RTP kernel (patent claims) ───────────────────────────────────

    def _rtp_dominance(self, p: Problem) -> Answer:
        series = np.array(p.params["series"])
        result = dominance_factor(series)
        return Answer(values=result, problem_type=p.type)

    def _rtp_evolution(self, p: Problem) -> Answer:
        series = np.array(p.params["series"])
        window = p.params.get("window", 10)
        half_life = p.params.get("half_life", 5.0)
        result = evolution_of_internal_component(series, window, half_life)
        return Answer(values=result, problem_type=p.type)

    def _rtp_simulate(self, p: Problem) -> Answer:
        es = p.params["entity_series"]
        names = list(es.keys())
        matrix = np.column_stack([np.array(es[n]) for n in names])
        max_lag = p.params.get("max_lag", 4)
        score_matrix = simulate_relationships(matrix, max_lag)
        return Answer(
            values={"score_matrix": score_matrix, "entity_names": names},
            problem_type=p.type,
        )

    def _rtp_identify(self, p: Problem) -> Answer:
        es = p.params["entity_series"]
        names = list(es.keys())
        target_name = p.query["target"]
        target_idx = names.index(target_name)
        matrix = np.column_stack([np.array(es[n]) for n in names])
        max_lag = p.params.get("max_lag", 4)
        score_matrix = simulate_relationships(matrix, max_lag)
        kept = eliminate_drivers(score_matrix, target_idx)
        passing = test_drivers(matrix, target_idx, kept)
        best = identify_target(passing, score_matrix, target_idx)
        return Answer(
            values={
                "identified_driver": names[best] if best is not None else None,
                "identified_index": best,
                "passing_drivers": [(names[i], g) for i, g in passing],
            },
            problem_type=p.type,
        )

    # ── VADE drift ───────────────────────────────────────────────────

    def _drift_monitor(self, p: Problem) -> Answer:
        X_fit = np.array(p.params["fit_data"])
        stream = np.array(p.params["stream"])
        nominal = p.params.get("nominal_theta", None)
        alarm = p.params.get("alarm_factor", 3.0)
        est = DriftEstimator()
        est.fit_whitening(X_fit)
        result = est.monitor(stream, nominal, alarm)
        return Answer(values=result, problem_type=p.type)

    # ── GPCR bias ────────────────────────────────────────────────────

    def _gpcr_bias(self, p: Problem) -> Answer:
        x_base = np.array(p.params["x_base"])
        x_complex = np.array(p.params["x_complex"])
        result = compute_bias(x_base, x_complex)
        return Answer(
            values={
                "bias_factor": result.bias_factor,
                "fiedler_gap": result.fiedler_gap,
                "confidence": result.confidence,
                "functional_selectivity_ratio": result.functional_selectivity_ratio,
                "delta_phi_primary": result.delta_phi_primary,
                "delta_phi_arrestin": result.delta_phi_arrestin,
            },
            problem_type=p.type,
        )

    # ── GRN canonical ────────────────────────────────────────────────

    def _grn_propagate(self, p: Problem) -> Answer:
        adj = np.array(p.params["adjacency"])
        pert = np.array(p.params["perturbations"])
        damping = p.params.get("damping", 0.85)
        result = propagate_canonical(adj, pert, damping)
        return Answer(values=result, problem_type=p.type)

    def _grn_dominance(self, p: Problem) -> Answer:
        adj = np.array(p.params["adjacency"])
        state = np.array(p.params["state"])
        alpha = p.params.get("alpha", 0.5)
        beta = p.params.get("beta", 0.3)
        scores = compute_dominance_scores(adj, state, alpha, beta)
        return Answer(values={"dominance_scores": scores}, problem_type=p.type)

    def _grn_sensitivity(self, p: Problem) -> Answer:
        adj = np.array(p.params["adjacency"])
        mag = p.params.get("perturbation_magnitude", 0.1)
        jac = sensitivity_analysis(adj, mag)
        return Answer(values={"jacobian": jac}, problem_type=p.type)

    def _grn_motifs(self, p: Problem) -> Answer:
        adj = np.array(p.params["adjacency"])
        threshold = p.params.get("threshold", 1e-6)
        result = network_motifs(adj, threshold)
        return Answer(values=result, problem_type=p.type)

    # ── Structural PK/PD ─────────────────────────────────────────────

    def _pkpd_structural(self, p: Problem) -> Answer:
        params = StructuralPKPDParams(**{
            k: v for k, v in p.params.items()
            if k in StructuralPKPDParams.__dataclass_fields__
        })
        doses = [DoseEvent(**d) for d in p.params.get("doses", [{"time": 0, "amount": 100}])]
        t_end = p.query.get("t_end", 72.0)
        result = simulate_structural_pkpd(params, doses, t_end)
        return Answer(values=result, problem_type=p.type)

    def _pkpd_toxicity(self, p: Problem) -> Answer:
        conc = np.array(p.params["concentration"])
        time = np.array(p.params["time"])
        h0 = p.params.get("h0", 2e-4)
        beta_tox = p.params.get("beta_tox", 2.2)
        c50_tox = p.params.get("c50_tox", 1.6)
        result = toxicity_hazard(conc, time, h0, beta_tox, c50_tox)
        return Answer(values=result, problem_type=p.type)

    # ── GRN D/C/E dynamics ───────────────────────────────────────────

    def _grn_simulate(self, p: Problem) -> Answer:
        nodes = [GRNNode(n["name"], n.get("baseline", 0.0))
                 for n in p.params.get("nodes", [])]
        edges = [GRNEdge(**e) for e in p.params.get("edges", [])]
        solver = GRNSolver(nodes, edges)
        intensities = p.params.get("intensities_seq", [])
        frequencies = p.params.get("frequencies_seq", [])
        history = solver.simulate(intensities, frequencies)
        return Answer(
            values={k: v for k, v in history.items()},
            problem_type=p.type,
        )

    def _grn_identify(self, p: Problem) -> Answer:
        nodes = [GRNNode(n["name"], n.get("baseline", 0.0))
                 for n in p.params.get("nodes", [])]
        edges = [GRNEdge(**e) for e in p.params.get("edges", [])]
        solver = GRNSolver(nodes, edges)
        intensities = p.params.get("intensities_seq", [])
        frequencies = p.params.get("frequencies_seq", [])
        target = p.query["target_node"]
        impacts = solver.identify_critical_edges(intensities, frequencies, target)
        return Answer(
            values={
                "critical_edges": [
                    {"source": s, "target": t, "impact": d}
                    for s, t, d in impacts
                ],
            },
            problem_type=p.type,
        )

    # ── Disease progression (remaining) ──────────────────────────────

    def _progression_event(self, p: Problem) -> Answer:
        params = EventParams(
            event_rate_per_year=p.params.get("event_rate", 2.0),
            event_magnitude=p.params.get("event_magnitude", 5.0),
            event_std=p.params.get("event_std", 2.0),
            recovery_hours=p.params.get("recovery_hours", 48.0),
        )
        months = np.arange(0, p.query.get("months", 48) + 1)
        trajectory = ProgressionSolver.event_trajectory(params, months)
        return Answer(
            values={"time_months": months, "values": trajectory},
            problem_type=p.type,
        )

    def _mci_conversion(self, p: Problem) -> Answer:
        params = LinearProgressionParams(
            baseline=p.params.get("baseline_mmse", 26.0),
            annual_rate=p.params.get("annual_rate", -1.5),
            rate_std=p.params.get("rate_std", 0.8),
        )
        months = np.arange(0, p.query.get("months", 60) + 1, 6)
        trajectory = ProgressionSolver.linear_trajectory(params, months)
        threshold = p.params.get("conversion_threshold", 24.0)
        below = np.where(trajectory <= threshold)[0]
        conversion_month = float(months[below[0]]) if len(below) > 0 else float("inf")
        return Answer(
            values={
                "time_months": months,
                "values": trajectory,
                "conversion_month": conversion_month,
                "threshold": threshold,
            },
            problem_type=p.type,
        )

    # ── Composite ────────────────────────────────────────────────────

    def _treatment_plan(self, p: Problem) -> Answer:
        goal_id = p.params.get("goal_id", "focus-enhancement")
        duration = p.params.get("duration_minutes", 30.0)
        n_phases = p.query.get("n_phases", 5)
        plan = TherapeuticSolver.plan_session(goal_id, duration, n_phases)
        pk = self._build_pk_params(p) if "drug" in p.params else None
        pk_info = {}
        if pk is not None:
            css = PKSolver.steady_state_trough(pk)
            pk_info = {"drug_trough_ng_ml": css, "drug": p.params["drug"]}
        return Answer(
            values={
                "vad_trajectory": plan.vad_trajectory,
                "total_duration": plan.total_duration_minutes,
                "n_phases": len(plan.phases),
                "phase_names": [ph.name for ph in plan.phases],
                **pk_info,
            },
            problem_type=p.type,
        )

    def _disease_forecast(self, p: Problem) -> Answer:
        disease = p.params.get("disease_type", "linear")
        months = np.arange(0, p.query.get("months", 60) + 1, p.query.get("interval", 6))
        if disease in ("episodic", "mdd", "bd"):
            params = EpisodicParams(
                relapse_prob_per_year=p.params.get("relapse_prob", 0.5),
                mean_episode_months=p.params.get("episode_months", 6),
                std_episode_months=p.params.get("episode_std", 3),
                remission_target=p.params.get("remission_target", 6.0),
                episode_target=p.params.get("episode_target", 22.0),
            )
            trajectory = ProgressionSolver.episodic_trajectory(params, months)
        else:
            params = LinearProgressionParams(
                baseline=p.params.get("baseline", 29.0),
                annual_rate=p.params.get("annual_rate", -2.5),
                rate_std=p.params.get("rate_std", 1.0),
            )
            trajectory = ProgressionSolver.linear_trajectory(params, months)
        return Answer(
            values={"time_months": months, "values": trajectory, "disease_type": disease},
            problem_type=p.type,
        )
