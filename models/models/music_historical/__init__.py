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
from .compositional_techniques import (
    CounterpointSpecies, MotivicTechnique, CompositionalTechnique, TECHNIQUE_CATALOG,
    technique_complexity_by_period, harmonic_rhythm_density,
    motivic_development_score, orchestration_complexity,
)
from .genre_genealogy import (
    Genre, GENRE_TREE,
    genre_distance, genre_influence_chain, genres_by_era,
    genre_fusion_candidates, genre_innovation_index,
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
    "CounterpointSpecies", "MotivicTechnique", "CompositionalTechnique", "TECHNIQUE_CATALOG",
    "technique_complexity_by_period", "harmonic_rhythm_density",
    "motivic_development_score", "orchestration_complexity",
    "Genre", "GENRE_TREE",
    "genre_distance", "genre_influence_chain", "genres_by_era",
    "genre_fusion_candidates", "genre_innovation_index",
    "pythagorean_ratios", "pythagorean_cents", "pythagorean_comma",
    "quarter_comma_meantone_ratios", "meantone_cents",
    "JUST_RATIOS_5LIMIT", "just_intonation_cents", "syntonic_comma",
    "werckmeister_iii_ratios", "werckmeister_iii_cents",
    "TuningComparison", "compare_tunings",
    "cents_to_frequency", "beat_frequency", "interval_purity",
    "PeriodLabel", "StyleFeatures", "PERIOD_PROTOTYPES",
    "period_likelihood", "dominant_period",
    "temporal_evolution", "period_influence_graph",
]

from .period_analysis import (
    PeriodLabel, StyleFeatures, PERIOD_PROTOTYPES,
    period_likelihood, dominant_period,
    style_distance as period_style_distance,
    temporal_evolution,
    influence_graph as period_influence_graph,
)

from .tuning_systems import (
    pythagorean_ratios, pythagorean_cents, pythagorean_comma,
    quarter_comma_meantone_ratios, meantone_cents,
    JUST_RATIOS_5LIMIT, just_intonation_cents, syntonic_comma,
    werckmeister_iii_ratios, werckmeister_iii_cents,
    TuningComparison, compare_tunings,
    cents_to_frequency, beat_frequency, interval_purity,
)
