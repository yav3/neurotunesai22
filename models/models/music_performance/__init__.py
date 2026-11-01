"""Music performance analysis: expression, tempo, intonation, ensemble, practice."""
from .expression import (
    DynamicLevel,
    ExpressionCurve,
    DynamicsAnalyzer,
    VibratoAnalyzer,
    ArticulationDetector,
)
from .tempo_expression import (
    RubatoDetector,
    AccelerandoDetector,
    AgogicAccent,
    MetricalStrength,
)
from .intonation import (
    PitchTracker,
    IntonationAnalyzer,
    TemperamentComparator,
    MicrotoneDetector,
)
from .ensemble import (
    SynchronyAnalyzer,
    BlendAnalyzer,
    BalanceAnalyzer,
    TuningAnalyzer,
)
from .practice import (
    PracticeSession,
    ProgressTracker,
    DifficultyEstimator,
    PracticeScheduler,
)

__all__ = [
    "DynamicLevel", "ExpressionCurve", "DynamicsAnalyzer",
    "VibratoAnalyzer", "ArticulationDetector",
    "RubatoDetector", "AccelerandoDetector", "AgogicAccent", "MetricalStrength",
    "PitchTracker", "IntonationAnalyzer", "TemperamentComparator", "MicrotoneDetector",
    "SynchronyAnalyzer", "BlendAnalyzer", "BalanceAnalyzer", "TuningAnalyzer",
    "PracticeSession", "ProgressTracker", "DifficultyEstimator", "PracticeScheduler",
    "OrnamentType", "Ornament", "MicroEvent",
    "realize_trill", "realize_mordent", "realize_turn",
    "realize_ornament", "apply_ornaments", "OrnamentDensity", "compute_density",
]

from .ornamentation import (
    OrnamentType, Ornament, MicroEvent,
    realize_trill, realize_mordent, realize_turn,
    realize_ornament, apply_ornaments, OrnamentDensity, compute_density,
)
