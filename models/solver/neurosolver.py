"""NeuroSolver — math-first problem → answer engine.

Routes structured Problem specifications to the appropriate mathematical
modules and returns structured Answer objects.

Usage:
    solver = NeuroSolver()
    problem = Problem(
        problem_type=ProblemType.SIMULATE_PKPD,
        drug=DrugSpec(drug_idx=0, dose_mg=10.0),
        time_points=torch.linspace(0, 24, 100),
    )
    answer = solver.solve(problem)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.config import MODALITY_DIMS, NUM_DISEASE_STATES
from models.math.pk_model import OneCompartmentPK
from models.math.pd_model import EmaxPD, PKPDCoupled
from models.math.progression_model import DiseaseTrajectory
from models.math.grn_dynamics import GRNNetwork, GRNSimulateEliminate, SelfOrganizingMap
from models.math.correlation_engine import CorrelationEngine
from models.math.vade_geometry import VADENavigator
from models.solver.problem import Problem, ProblemType
from models.solver.answer import Answer


TOTAL_BIOMARKERS = sum(MODALITY_DIMS.values())


class DiagnosisHead(nn.Module):
    """Classifies patient state into disease probabilities."""

    def __init__(self, input_dim: int, n_diseases: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_diseases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuroSolver(nn.Module):
    """Math-first solver that routes problems to equation modules.

    Modules:
        pk: 1-compartment pharmacokinetics
        pd: Emax pharmacodynamics
        pkpd: Coupled PK/PD
        progression: Disease trajectory (linear + AR(1))
        grn: Generative Relational Network dynamics
        grn_sim: GRN simulate→eliminate→identify
        som: Self-Organizing Map for dimensionality reduction
        correlation: Multivariate correlation engine
        vade: VADE Glome S³ navigator
        diagnosis: Disease classification from biomarkers
    """

    def __init__(
        self,
        n_drugs: int = 11,
        n_diseases: int = NUM_DISEASE_STATES,
        n_biomarkers: int = TOTAL_BIOMARKERS,
        n_grn_entities: int = 20,
    ):
        super().__init__()
        self.pk = OneCompartmentPK(n_drugs)
        self.pd = EmaxPD(n_drugs)
        self.pkpd = PKPDCoupled(self.pk, self.pd)
        self.progression = DiseaseTrajectory(n_diseases, n_biomarkers)
        self.grn = GRNNetwork(n_grn_entities)
        self.grn_sim = GRNSimulateEliminate(self.grn)
        self.som = SelfOrganizingMap(n_biomarkers, grid_h=10, grid_w=10)
        self.correlation = CorrelationEngine(MODALITY_DIMS)
        self.vade = VADENavigator(n_biomarkers, vade_dim=4)
        self.diagnosis = DiagnosisHead(n_biomarkers, n_diseases)

        self._dispatch = {
            ProblemType.DIAGNOSE: self._solve_diagnose,
            ProblemType.PREDICT_PROGRESSION: self._solve_progression,
            ProblemType.PREDICT_TREATMENT: self._solve_treatment,
            ProblemType.IMPUTE: self._solve_impute,
            ProblemType.SIMULATE_PK: self._solve_pk,
            ProblemType.SIMULATE_PD: self._solve_pd,
            ProblemType.SIMULATE_PKPD: self._solve_pkpd,
            ProblemType.GRN_DYNAMICS: self._solve_grn,
            ProblemType.VADE_NAVIGATE: self._solve_vade,
            ProblemType.SAMPLE_COHORT: self._solve_sample,
        }

    def solve(self, problem: Problem) -> Answer:
        handler = self._dispatch[problem.problem_type]
        return handler(problem)

    def forward(self, problem: Problem) -> Answer:
        return self.solve(problem)

    def _concat_biomarkers(self, patient) -> torch.Tensor:
        parts = []
        for name in MODALITY_DIMS:
            if name in patient.biomarkers:
                parts.append(patient.biomarkers[name])
            else:
                parts.append(torch.zeros(patient.biomarkers[next(iter(patient.biomarkers))].shape[0],
                                         MODALITY_DIMS[name]))
        return torch.cat(parts, dim=-1)

    def _solve_diagnose(self, p: Problem) -> Answer:
        x = self._concat_biomarkers(p.patient)
        logits = self.diagnosis(x)
        probs = logits.softmax(dim=-1)
        return Answer(
            result={"logits": logits, "probabilities": probs, "predicted_class": probs.argmax(dim=-1)},
            problem_type="diagnose",
        )

    def _solve_progression(self, p: Problem) -> Answer:
        x = self._concat_biomarkers(p.patient)
        disease_idx = torch.full((x.shape[0],), p.disease_idx or p.patient.disease_label or 0, dtype=torch.long)
        trajectory = self.progression(x, p.time_points, disease_idx, stochastic=p.stochastic)
        return Answer(
            result={"trajectory": trajectory, "time_points": p.time_points},
            problem_type="predict_progression",
        )

    def _solve_treatment(self, p: Problem) -> Answer:
        pkpd_out = self._solve_pkpd(p)
        effect = pkpd_out.result["effect"]
        if p.patient is not None:
            x = self._concat_biomarkers(p.patient)
            disease_idx = torch.full((x.shape[0],), p.disease_idx or 0, dtype=torch.long)
            treatment_effect = effect.unsqueeze(-1).expand_as(x.unsqueeze(1) if p.time_points.dim() > 0 else x)
            trajectory = self.progression(
                x, p.time_points, disease_idx, treatment_effect=treatment_effect, stochastic=p.stochastic,
            )
            return Answer(
                result={**pkpd_out.result, "trajectory": trajectory},
                problem_type="predict_treatment",
            )
        return Answer(result=pkpd_out.result, problem_type="predict_treatment")

    def _solve_impute(self, p: Problem) -> Answer:
        log_probs = self.correlation(p.patient.biomarkers)
        imputed = {}
        for name in MODALITY_DIMS:
            if name not in p.patient.biomarkers or (name in p.patient.mask and p.patient.mask[name].any()):
                imputed[name] = self.correlation.sample(name, p.n_samples)
        return Answer(
            result={"imputed": imputed, "log_probs": log_probs},
            problem_type="impute",
        )

    def _solve_pk(self, p: Problem) -> Answer:
        n = p.time_points.shape[0]
        dose = torch.full((n,), p.drug.dose_mg)
        drug_idx = torch.full((n,), p.drug.drug_idx, dtype=torch.long)
        route = torch.full((n,), p.drug.route, dtype=torch.long)
        conc = self.pk(dose, p.time_points, drug_idx, route)
        return Answer(
            result={"concentration": conc, "time": p.time_points},
            problem_type="simulate_pk",
        )

    def _solve_pd(self, p: Problem) -> Answer:
        conc = p.extra.get("concentration", torch.linspace(0, 100, 50))
        drug_idx = torch.full((conc.shape[0],), p.drug.drug_idx, dtype=torch.long)
        effect = self.pd(conc, drug_idx)
        return Answer(
            result={"effect": effect, "concentration": conc},
            problem_type="simulate_pd",
        )

    def _solve_pkpd(self, p: Problem) -> Answer:
        n = p.time_points.shape[0]
        dose = torch.full((n,), p.drug.dose_mg)
        drug_idx = torch.full((n,), p.drug.drug_idx, dtype=torch.long)
        route = torch.full((n,), p.drug.route, dtype=torch.long)
        out = self.pkpd(dose, p.time_points, drug_idx, route)
        return Answer(
            result={**out, "time": p.time_points},
            problem_type="simulate_pkpd",
        )

    def _solve_grn(self, p: Problem) -> Answer:
        result = self.grn_sim(p.grn_signal, n_sim_steps=p.grn_steps)
        return Answer(
            result=result,
            problem_type="grn_dynamics",
        )

    def _solve_vade(self, p: Problem) -> Answer:
        x = self._concat_biomarkers(p.patient) if p.patient else p.extra["features"]
        proj = self.vade.project(x)
        session = self.vade.plan_session(proj["p_sphere"], p.vade_target, p.vade_n_steps)
        constraint_ok = self.vade.check_therapeutic_constraint(session["trajectory"])
        return Answer(
            result={**session, "projection": proj, "therapeutic_constraint": constraint_ok},
            problem_type="vade_navigate",
        )

    def _solve_sample(self, p: Problem) -> Answer:
        samples = {}
        for name in (p.target_modalities or list(MODALITY_DIMS.keys())):
            if name in self.correlation.group_names:
                samples[name] = self.correlation.sample(name, p.n_samples)
        return Answer(
            result={"samples": samples},
            problem_type="sample_cohort",
        )
