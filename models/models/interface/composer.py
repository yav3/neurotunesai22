"""Pipeline composer — chain solvers for multi-step clinical problems.

Example: Treatment Plan = PK → PD → Progression forecast
    1. Compute drug concentration over time (PK)
    2. Map concentration → pharmacological effect (PD)
    3. Combine with disease progression to forecast outcome

Example: Session Design = VADE target → Fibonacci schedule → Spectral goals
    1. Map patient state to VADE target on S³
    2. Generate Fibonacci-constrained phase durations
    3. Assign spectral (EEG band) goals per phase via geodesic interpolation
"""
from __future__ import annotations

import numpy as np

from models.interface.problem import Problem, ProblemType, Answer
from models.interface.solver import UnifiedSolver


class PipelineComposer:
    """Compose multi-step mathematical pipelines."""

    def __init__(self, solver: UnifiedSolver | None = None):
        self.solver = solver or UnifiedSolver()

    def treatment_plan(
        self,
        drug: str,
        weight_kg: float,
        baseline_biomarker: float,
        disease: str,
        annual_rate: float,
        duration_months: int = 12,
    ) -> dict[str, Answer]:
        """Full treatment plan: PK → PD → progression + drug effect overlay."""

        # Step 1: PK multi-dose over treatment period
        pk_answer = self.solver.solve(Problem(
            type=ProblemType.PK_MULTI_DOSE,
            params={"drug": drug, "weight_kg": weight_kg},
            query={"duration_hr": duration_months * 30 * 24, "n_doses": duration_months * 30},
        ))

        # Step 2: PD effect at steady state
        pk_trough = pk_answer.values.get("trough_ng_ml", 0)
        pd_answer = self.solver.solve(Problem(
            type=ProblemType.PD_BIOMARKER,
            params={"drug": drug, "baseline": baseline_biomarker},
            query={"concentration_ng_ml": pk_trough},
        ))

        # Step 3: Disease progression (natural history)
        prog_answer = self.solver.solve(Problem(
            type=ProblemType.PROGRESSION_LINEAR,
            params={
                "baseline": baseline_biomarker,
                "annual_rate": annual_rate,
                "rate_std": abs(annual_rate) * 0.3,
            },
            query={"months": duration_months, "interval": 1},
        ))

        # Step 4: Combine — treated trajectory = natural + drug effect
        natural = prog_answer.values["values"]
        drug_effect = pd_answer.values.get("biomarker", baseline_biomarker) - baseline_biomarker
        treated = natural + drug_effect  # additive drug effect assumption

        return {
            "pk": pk_answer,
            "pd": pd_answer,
            "natural_progression": prog_answer,
            "treated_trajectory": Answer(
                values={
                    "time_months": prog_answer.values["time_months"],
                    "natural": natural,
                    "treated": treated,
                    "drug_effect": drug_effect,
                },
                problem_type=ProblemType.TREATMENT_PLAN,
            ),
        }

    def session_design(
        self,
        current_vade: np.ndarray,
        target_vade: np.ndarray,
        max_duration_minutes: float = 60.0,
        base_phase_minutes: float = 5.0,
    ) -> dict[str, Answer]:
        """Design a therapeutic session: VADE trajectory + Fibonacci schedule + spectral goals."""

        # Step 1: Fibonacci schedule
        sched = self.solver.solve(Problem(
            type=ProblemType.VADE_SCHEDULE,
            params={"base_minutes": base_phase_minutes, "max_duration": max_duration_minutes},
        ))
        n_phases = len(sched.values.get("phases_minutes", []))

        # Step 2: VADE geodesic trajectory
        traj = self.solver.solve(Problem(
            type=ProblemType.VADE_TRAJECTORY,
            params={"start": current_vade.tolist(), "target": target_vade.tolist()},
            query={"n_phases": n_phases},
        ))

        # Step 3: Map each waypoint's arousal to an EEG target band
        trajectory = traj.values.get("trajectory", np.array([]))
        phase_goals = []
        for point in trajectory:
            arousal = point[1] if len(point) > 1 else 0.5
            if arousal < 0.2:
                phase_goals.append("sleep")
            elif arousal < 0.4:
                phase_goals.append("meditation")
            elif arousal < 0.6:
                phase_goals.append("relaxation")
            elif arousal < 0.8:
                phase_goals.append("focus")
            else:
                phase_goals.append("energy")

        return {
            "schedule": sched,
            "trajectory": traj,
            "phase_goals": Answer(
                values={
                    "goals": phase_goals,
                    "phases_minutes": sched.values.get("phases_minutes", np.array([])),
                    "trajectory": trajectory,
                },
                problem_type=ProblemType.VADE_TRAJECTORY,
            ),
        }

    def disease_forecast(
        self,
        disease: str,
        baseline_scores: dict[str, float],
        annual_rates: dict[str, float],
        months: int = 60,
    ) -> dict[str, Answer]:
        """Multi-biomarker disease forecast."""
        results = {}
        for biomarker, baseline in baseline_scores.items():
            rate = annual_rates.get(biomarker, 0.0)
            answer = self.solver.solve(Problem(
                type=ProblemType.PROGRESSION_LINEAR,
                params={
                    "baseline": baseline,
                    "annual_rate": rate,
                    "rate_std": abs(rate) * 0.3,
                },
                query={"months": months, "interval": 6},
            ))
            results[biomarker] = answer
        return results

    def therapeutic_session(
        self,
        patient_vad: np.ndarray,
        goal_id: str,
        duration_minutes: float = 30.0,
    ) -> dict[str, Answer]:
        """Full therapeutic session: plan → EIF → binaural targets per phase."""

        plan = self.solver.solve(Problem(
            type=ProblemType.THERAPEUTIC_SESSION_PLAN,
            params={"goal_id": goal_id, "duration_minutes": duration_minutes},
            query={"n_phases": 5},
        ))

        eif = self.solver.solve(Problem(
            type=ProblemType.THERAPEUTIC_EIF,
            params={
                "valence": float(patient_vad[0]),
                "arousal": float(patient_vad[1]),
                "dominance": float(patient_vad[2]) if len(patient_vad) > 2 else 0.5,
            },
        ))

        phase_names = plan.values.get("phase_names", [])
        binaural_targets = []
        for name in phase_names:
            band = "alpha"
            nl = name.lower()
            if "sleep" in nl or "delta" in nl:
                band = "delta"
            elif "relax" in nl or "wind" in nl:
                band = "theta"
            elif "focus" in nl or "deep" in nl:
                band = "beta"
            elif "energy" in nl or "lift" in nl:
                band = "beta"

            beat = self.solver.solve(Problem(
                type=ProblemType.BINAURAL_BEAT,
                query={"target_band": band},
            ))
            binaural_targets.append(beat)

        return {
            "plan": plan,
            "eif": eif,
            "binaural_targets": binaural_targets,
        }
