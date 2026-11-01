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
    MusicalForm, FormType, FormAnalysis, CANONICAL_LABEL_SEQUENCES,
    form_period_distribution, form_complexity_score, form_lineage,
    section_similarity, detect_form, repetition_rate, analyze_form,
    period_from_year,
)
from .instrumentation_history import (
    InstrumentEra, INSTRUMENT_TIMELINE,
    available_instruments, orchestra_size_by_period, instrument_family_balance,
)

from .performance_practice import (
    TuningSystem, ArticulationStyle, PerformancePractice, PERIOD_PRACTICES,
    reference_pitch_for_year, tuning_deviation_cents, ornamentation_density,
)
from .composer_network import (
    ComposerProfile, CANONICAL_COMPOSERS,
    composers_by_period, contemporaries, innovation_chain, stylistic_similarity,
)

__all__ = [
    "MusicalPeriod", "PeriodProfile", "PERIOD_PROFILES",
    "classify_period_by_year", "period_harmonic_signature", "style_distance",
    "StyleFeatureVector", "style_evolution_trajectory",
    "innovation_score", "influence_graph",
    "MusicalForm", "FormType", "FormAnalysis", "CANONICAL_LABEL_SEQUENCES",
    "form_period_distribution", "form_complexity_score", "form_lineage",
    "section_similarity", "detect_form", "repetition_rate", "analyze_form",
    "period_from_year",
    "InstrumentEra", "INSTRUMENT_TIMELINE",
    "available_instruments", "orchestra_size_by_period", "instrument_family_balance",
    "TuningSystem", "ArticulationStyle", "PerformancePractice", "PERIOD_PRACTICES",
    "reference_pitch_for_year", "tuning_deviation_cents", "ornamentation_density",
    "ComposerProfile", "CANONICAL_COMPOSERS",
    "composers_by_period", "contemporaries", "innovation_chain", "stylistic_similarity",
    "PeriodCharacteristics", "PERIOD_DATABASE",
    "classify_period", "period_distance", "evolution_timeline",
]

from .period_style import (
    PeriodCharacteristics, PERIOD_DATABASE,
    classify_period, period_distance, evolution_timeline,
)
