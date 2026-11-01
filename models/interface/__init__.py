"""Problem → Answer interface.

Pose a mathematical/clinical problem, get a numerical answer.
"""
from models.interface.problem import Problem, ProblemType
from models.interface.solver import UnifiedSolver
from models.interface.composer import PipelineComposer
