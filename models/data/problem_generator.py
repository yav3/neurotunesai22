"""Generate (problem, exact_answer) training pairs from the math_core solvers.

The neural foundation model is trained on these pairs — it learns to
approximate the exact solvers. Math first: the training signal comes from
equations, not language.

Each generator produces a batch of problems with known answers, sampled
from the parameter ranges in the clinical schemas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.interface.problem import Problem, ProblemType, Answer
from models.interface.solver import UnifiedSolver
from models.math_core.pk_solver import PKSolver, PKParams
from models.math_core.pd_solver import PDSolver, PDParams
from models.math_core.vade_geometry import VADEGeometry
from models.math_core.spectral import SpectralSolver, EEG_BANDS


SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic_data" / "schemas"


def _load_drug_names() -> list[str]:
    path = SCHEMA_DIR / "drug_parameters.json"
    if path.exists():
        with open(path) as f:
            return list(json.load(f).get("drugs", {}).keys())
    return ["donepezil"]


class MathProblemDataset(Dataset):
    """Generates (problem_encoding, answer_value) pairs on the fly.

    Each sample is:
        x: np.float32 vector encoding the problem (type + params + query)
        y: np.float32 scalar or vector — the exact numerical answer
    """

    def __init__(
        self,
        n_samples: int = 100_000,
        seed: int = 42,
        answer_dim: int = 8,
    ):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.solver = UnifiedSolver()
        self.drug_names = _load_drug_names()
        self.answer_dim = answer_dim

        # Pre-generate all problems for reproducibility
        self.problems: list[tuple[np.ndarray, np.ndarray]] = []
        self._generate_all()

    def _generate_all(self):
        generators = [
            self._gen_pk_problems,
            self._gen_pd_problems,
            self._gen_vade_problems,
            self._gen_spectral_problems,
        ]
        per_gen = self.n_samples // len(generators)

        for gen in generators:
            for _ in range(per_gen):
                try:
                    x, y = gen()
                    self.problems.append((x, y))
                except Exception:
                    pass

        # Shuffle
        self.rng.shuffle(self.problems)

    def _gen_pk_problems(self) -> tuple[np.ndarray, np.ndarray]:
        drug = self.rng.choice(self.drug_names)
        weight = self.rng.uniform(50, 120)
        time_hr = self.rng.uniform(0.5, 72)

        problem = Problem(
            type=ProblemType.PK_CONCENTRATION,
            params={"drug": drug, "weight_kg": weight},
            query={"time_hr": time_hr},
        )
        answer = self.solver.solve(problem)
        x = problem.encode()
        y_val = answer.values.get("concentration_ng_ml", 0.0)
        y = np.zeros(self.answer_dim, dtype=np.float32)
        y[0] = float(y_val)
        return x, y

    def _gen_pd_problems(self) -> tuple[np.ndarray, np.ndarray]:
        drug = self.rng.choice(self.drug_names)
        conc = self.rng.lognormal(2, 1)
        baseline = self.rng.uniform(0, 30)

        problem = Problem(
            type=ProblemType.PD_EFFECT,
            params={"drug": drug, "baseline": baseline},
            query={"concentration_ng_ml": conc},
        )
        answer = self.solver.solve(problem)
        x = problem.encode()
        y = np.zeros(self.answer_dim, dtype=np.float32)
        y[0] = float(answer.values.get("effect", 0.0))
        return x, y

    def _gen_vade_problems(self) -> tuple[np.ndarray, np.ndarray]:
        # Random points on S³
        p = self.rng.normal(size=4)
        p = p / np.linalg.norm(p)
        q = self.rng.normal(size=4)
        q = q / np.linalg.norm(q)

        problem = Problem(
            type=ProblemType.VADE_GEODESIC,
            params={"point": p.tolist()},
            query={"target": q.tolist()},
        )
        answer = self.solver.solve(problem)
        x = problem.encode()
        y = np.zeros(self.answer_dim, dtype=np.float32)
        y[0] = float(answer.values.get("geodesic_distance", 0.0))
        return x, y

    def _gen_spectral_problems(self) -> tuple[np.ndarray, np.ndarray]:
        # Synthetic PSD
        freqs = np.linspace(0.5, 100, 200)
        psd = self.rng.exponential(1.0, size=200) / (freqs + 1)  # 1/f-like

        powers = SpectralSolver.all_band_powers(psd, freqs)
        H = SpectralSolver.spectral_entropy(psd)
        tar = SpectralSolver.theta_alpha_ratio(powers)

        x = Problem(
            type=ProblemType.SPECTRAL_ENTROPY,
            params={"psd": psd[:16].tolist()},
            query={},
        ).encode()

        y = np.zeros(self.answer_dim, dtype=np.float32)
        y[0] = H
        y[1] = tar
        y[2] = powers.get("delta", 0)
        y[3] = powers.get("theta", 0)
        y[4] = powers.get("alpha", 0)
        y[5] = powers.get("beta", 0)
        y[6] = powers.get("gamma", 0)
        return x, y

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x, y = self.problems[idx]
        return {
            "input": torch.from_numpy(x),
            "target": torch.from_numpy(y),
        }

    @property
    def input_dim(self) -> int:
        return Problem.encoding_dim()
