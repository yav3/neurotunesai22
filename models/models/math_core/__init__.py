"""Exact mathematical solvers — the ground-truth computation layer.

Every equation from the patent portfolio and clinical parameter schemas
is implemented here as a deterministic solver. The neural foundation model
is trained to approximate and generalize these solvers, not the other way
around. Math first, learning second.
"""
from models.math_core.pk_solver import PKSolver
from models.math_core.pd_solver import PDSolver
from models.math_core.progression_solver import ProgressionSolver
from models.math_core.grn_solver import GRNSolver
from models.math_core.vade_geometry import VADEGeometry
from models.math_core.spectral import SpectralSolver
from models.math_core.audio_solver import AudioSolver
from models.math_core.therapeutic_solver import TherapeuticSolver
