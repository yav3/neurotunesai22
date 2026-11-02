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
from .harmonic_language_evolution import (
    ROMAN_NUMERALS, ERA_TRANSITION_MATRICES,
    HarmonicProfile, ProgressionAnalysis,
    harmonic_entropy, vocabulary_usage, dominant_resolution_rate,
    analyze_progression, profile_era, era_entropy_trajectory,
    generate_progression, transition_divergence,
)

from .style_classifier import (
    PeriodPrototype, FEATURE_NAMES, DEFAULT_PROTOTYPES,
    mahalanobis_distance, log_likelihood, classify_period,
    period_distances, nearest_periods, transition_probability,
    feature_importance, classify_batch,
    confusion_matrix as sc_confusion_matrix,
)

__all__ = [
    "PeriodPrototype", "FEATURE_NAMES", "DEFAULT_PROTOTYPES",
    "mahalanobis_distance", "log_likelihood",
    "period_distances", "nearest_periods", "transition_probability",
    "feature_importance", "classify_batch",
    "sc_confusion_matrix",
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
    "kirnberger_iii_cents", "vallotti_cents", "young_ii_cents",
    "equal_temperament_cents", "get_temperament",
    "KeyColor", "key_color", "all_key_colors", "key_ranking_by_purity",
    "WolfInterval", "detect_wolf_intervals", "wolf_fifth_location",
    "temperament_distance_matrix", "temperament_key_affinity",
    "AdaptiveState", "adaptive_just_retune", "adaptive_drift_trajectory",
    "StyleFingerprint", "StyleDistance", "EraCharacteristics", "ERA_CENTROIDS",
    "compute_style_fingerprint", "style_distance_transfer",
    "classify_era_transfer", "interpolate_styles",
    "ornamentation_density_hist", "harmonic_vocabulary_size",
    "rhythmic_complexity_score", "melodic_range_profile",
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
from .historical_temperament_analysis import (
    kirnberger_iii_cents, vallotti_cents, young_ii_cents,
    equal_temperament_cents, get_temperament,
    KeyColor, key_color, all_key_colors, key_ranking_by_purity,
    WolfInterval, detect_wolf_intervals, wolf_fifth_location,
    temperament_distance_matrix, temperament_key_affinity,
    AdaptiveState, adaptive_just_retune, adaptive_drift_trajectory,
)
from .historical_instrument_acoustics import (
    InstrumentFamily, HistoricalEra as AcousticEra,
    HarmonicProfile, DecayModel, FormantSpec, HistoricalInstrument,
    INSTRUMENT_CATALOG, instruments_by_era, instruments_by_family,
    spectral_overlap, pitch_range_overlap,
    ensemble_blend_matrix, ensemble_spectral_diversity,
    timbral_evolution, era_average_brightness, masking_potential,
)
from .style_transfer_analysis import (
    StyleFingerprint, StyleDistance, EraCharacteristics, ERA_CENTROIDS,
    compute_style_fingerprint,
    style_distance as style_distance_transfer,
    classify_era as classify_era_transfer, interpolate_styles,
    ornamentation_density as ornamentation_density_hist,
    harmonic_vocabulary_size,
    rhythmic_complexity_score, melodic_range_profile,
)
from .notation_systems import (
    NotationEra, MensuralDuration, MensuralNote, Mensuration, NeumeGroup,
    ShapeNote, NEUME_CATALOG, SHAPE_NOTE_FOUR, SHAPE_NOTE_SEVEN,
    mensuration_ratio, mensural_to_modern_duration, neume_to_intervals,
    intervals_to_neume, TabNote, parse_tab_line, tab_to_pitches,
    notation_complexity, era_pitch_precision,
)
