"""Music history: period analysis, style evolution, form lineage, instrumentation."""

from .periods import (
    MusicalPeriod, PeriodProfile, PERIOD_PROFILES,
    classify_period_by_year, period_harmonic_signature, style_distance,
)
from .evolution import (
    StyleFeatureVector, style_evolution_trajectory,
    innovation_score, influence_graph,
)
from .form_evolution import (
    MusicalForm, form_period_distribution, form_complexity_score, form_lineage,
)
from .instrumentation_history import (
    InstrumentEra, INSTRUMENT_TIMELINE,
    available_instruments, orchestra_size_by_period, instrument_family_balance,
)

__all__ = [
    "MusicalPeriod", "PeriodProfile", "PERIOD_PROFILES",
    "classify_period_by_year", "period_harmonic_signature", "style_distance",
    "StyleFeatureVector", "style_evolution_trajectory",
    "innovation_score", "influence_graph",
    "MusicalForm", "form_period_distribution", "form_complexity_score", "form_lineage",
    "InstrumentEra", "INSTRUMENT_TIMELINE",
    "available_instruments", "orchestra_size_by_period", "instrument_family_balance",
]
