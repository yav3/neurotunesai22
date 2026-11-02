"""Music cultural analysis: scales, tunings, rhythmic systems, form, ornamentation across world traditions."""
from models.music_cultural.scales import (
    CulturalScale, ScaleDatabase,
    MAQAM_SCALES, RAGA_SCALES, GAMELAN_SCALES,
    interval_pattern, scale_similarity,
)
from models.music_cultural.tuning import (
    TuningSystem as TuningSystemBase, EqualTemperament as EqualTemperamentBase,
    JustIntonation as JustIntonationBase, PythagoreanTuning as PythagoreanTuningBase,
    MeantoneTemperament as MeantoneTemperamentBase, WellTemperament as WellTemperamentBase,
    cents_to_ratio, ratio_to_cents, frequency_ratio,
)
from models.music_cultural.rhythm import *  # noqa: F401,F403

# Deep world-scale library (35+ scales)
from models.music_cultural.world_scales import (
    WorldScale, ScaleMatcher, WORLD_SCALE_REGISTRY,
    MAQAM_RAST, MAQAM_BAYATI, MAQAM_HIJAZ, MAQAM_SABA,
    MAQAM_NAHAWAND, MAQAM_KURD, MAQAM_AJAM,
    RAGA_BILAWAL, RAGA_KAFI, RAGA_BHAIRAVI, RAGA_YAMAN,
    RAGA_KHAMAJ, RAGA_ASAVARI, RAGA_MARWA, RAGA_TODI, RAGA_PURVI,
    HIRAJOSHI, KUMOI, IN_SCALE, YO_SCALE, IWATO, MIYAKO_BUSHI,
    GONG, SHANG, JIAO, ZHI, YU,
    SLENDRO, PELOG,
    TIZITA_MAJOR, TIZITA_MINOR, BATI_MAJOR, BATI_MINOR, AMBASSEL,
    BLUES, WHOLE_TONE, HUNGARIAN_MINOR, PERSIAN, BYZANTINE,
)

# Deep tuning system library
from models.music_cultural.tuning_systems import (
    TuningSystem, TuningComparison,
    pythagorean_tuning, just_intonation, equal_temperament,
    meantone_tuning, maqam_quarter_tones, slendro_tuning,
    pelog_tuning, shruti_tuning, cents_from_ratio,
    ratio_from_cents, tuning_deviation, compare_tunings,
    frequency_from_cents, nearest_12tet_note,
)

# Deep rhythm system library
from models.music_cultural.rhythm_systems import (
    TimeSystem, Tala, Usul, ClaveSyncopation,
    TINTAL, JHAPTAL, RUPAK, EKTAL, DADRA, KEHERWA, TALA_REGISTRY,
    DUYEK, AKSAK, CIFTETELLI, KARSILAMA, USUL_REGISTRY,
    SON_32, SON_23, RUMBA, BOSSA_NOVA, CLAVE_REGISTRY,
    EuclideanRhythm, PolyrhythmGenerator,
)

# Musical form analysis (22 forms)
from models.music_cultural.form_analysis import (
    MusicalForm, FORM_LIBRARY, FormMatcher,
)

# Ornamentation library (28 ornaments, 5 traditions)
from models.music_cultural.ornamentation import (
    Ornament, ORNAMENT_LIBRARY, OrnamentDetector,
    TRILL, MORDENT, UPPER_MORDENT, TURN, APPOGGIATURA, ACCIACCATURA,
    GRUPPETTO, GLISSANDO, TREMOLO,
    GAMAK, MEEND, KAN, MURKI, KHATKA, ANDOLAN,
    RAQS, ZALAQA, RASHA,
    CUT, STRIKE, CRAN, ROLL, SLIDE_IRISH, TREBLE,
    RASGUEADO, PICADO, ALZAPUA, TREMOLO_FLAMENCO,
)

__all__ = [
    # Original scale module
    "CulturalScale", "ScaleDatabase",
    "MAQAM_SCALES", "RAGA_SCALES", "GAMELAN_SCALES",
    "interval_pattern", "scale_similarity",
    # Original tuning module (base classes)
    "TuningSystemBase", "EqualTemperamentBase", "JustIntonationBase",
    "PythagoreanTuningBase", "MeantoneTemperamentBase", "WellTemperamentBase",
    "cents_to_ratio", "ratio_to_cents", "frequency_ratio",
    # World scales (35+)
    "WorldScale", "ScaleMatcher", "WORLD_SCALE_REGISTRY",
    "MAQAM_RAST", "MAQAM_BAYATI", "MAQAM_HIJAZ", "MAQAM_SABA",
    "MAQAM_NAHAWAND", "MAQAM_KURD", "MAQAM_AJAM",
    "RAGA_BILAWAL", "RAGA_KAFI", "RAGA_BHAIRAVI", "RAGA_YAMAN",
    "RAGA_KHAMAJ", "RAGA_ASAVARI", "RAGA_MARWA", "RAGA_TODI", "RAGA_PURVI",
    "HIRAJOSHI", "KUMOI", "IN_SCALE", "YO_SCALE", "IWATO", "MIYAKO_BUSHI",
    "GONG", "SHANG", "JIAO", "ZHI", "YU",
    "SLENDRO", "PELOG",
    "TIZITA_MAJOR", "TIZITA_MINOR", "BATI_MAJOR", "BATI_MINOR", "AMBASSEL",
    "BLUES", "WHOLE_TONE", "HUNGARIAN_MINOR", "PERSIAN", "BYZANTINE",
    # Tuning systems
    "TuningSystem", "TuningComparison",
    "pythagorean_tuning", "just_intonation", "equal_temperament",
    "meantone_tuning", "maqam_quarter_tones", "slendro_tuning",
    "pelog_tuning", "shruti_tuning", "cents_from_ratio",
    "ratio_from_cents", "tuning_deviation", "compare_tunings",
    "frequency_from_cents", "nearest_12tet_note",
    # Rhythm systems
    "TimeSystem", "Tala", "Usul", "ClaveSyncopation",
    "TINTAL", "JHAPTAL", "RUPAK", "EKTAL", "DADRA", "KEHERWA", "TALA_REGISTRY",
    "DUYEK", "AKSAK", "CIFTETELLI", "KARSILAMA", "USUL_REGISTRY",
    "SON_32", "SON_23", "RUMBA", "BOSSA_NOVA", "CLAVE_REGISTRY",
    "EuclideanRhythm", "PolyrhythmGenerator",
    # Form analysis
    "MusicalForm", "FORM_LIBRARY", "FormMatcher",
    # Ornamentation
    "Ornament", "ORNAMENT_LIBRARY", "OrnamentDetector",
    "TRILL", "MORDENT", "UPPER_MORDENT", "TURN", "APPOGGIATURA", "ACCIACCATURA",
    "GRUPPETTO", "GLISSANDO", "TREMOLO",
    "GAMAK", "MEEND", "KAN", "MURKI", "KHATKA", "ANDOLAN",
    "RAQS", "ZALAQA", "RASHA",
    "CUT", "STRIKE", "CRAN", "ROLL", "SLIDE_IRISH", "TREBLE",
    "RASGUEADO", "PICADO", "ALZAPUA", "TREMOLO_FLAMENCO",
    "Tradition", "ScaleProfile", "RhythmProfile",
    "TraditionFingerprint", "TRADITION_FINGERPRINTS",
    "FusionAnalysis",
    "scale_interval_similarity", "tuning_deviation_12tet", "microtonal_content",
    "rhythm_accent_correlation", "tradition_distance", "tradition_similarity_matrix",
    "analyze_fusion_potential", "classify_tradition",
    "GeographicPoint", "StyleRegion",
    "haversine_distance", "geographic_distance_matrix", "feature_distance_matrix",
    "mantel_test", "spatial_autocorrelation", "inverse_distance_weighting",
    "classify_region", "diffusion_gradient", "cultural_boundary_detection",
    "isolation_by_distance",
    "DancePattern", "DANCE_CATALOG", "pattern_to_binary", "accent_to_weight",
    "dance_pattern_similarity", "classify_dance",
    "generate_onset_times", "dance_syncopation_score",
    "swing_feel_from_pattern", "tempo_in_range",
    "ModalFamily", "Mode", "CHURCH_MODES", "MELODIC_MINOR_MODES", "PENTATONIC_MODES",
    "pitch_class_histogram", "mode_template", "correlate_profile", "detect_mode",
    "transpose_mode", "modal_brightness", "modal_interchange_candidates",
    "characteristic_interval_strength", "interval_vector", "mode_distance",
    "brightness_ordering", "mode_subset_of", "common_tones",
    "TransmissionMode", "CulturalNode", "InfluenceEdge", "DiffusionState",
    "cin_build_influence_graph", "cultural_feature_distance",
    "cin_cultural_distance_matrix", "proximity_influence_graph",
    "threshold_diffusion", "continuous_diffusion", "influence_centrality",
    "transmission_pathways", "pathway_strength", "dominant_pathway",
    "cin_cultural_mds", "innovation_rate", "cultural_entropy",
    "syncretism_score", "influence_asymmetry", "net_influence",
    "CulturalDimension", "TraditionProfile", "CulturalDistanceResult",
    "scale_distance", "rhythm_distance", "tuning_distance",
    "ornament_distance", "form_distance", "texture_distance", "timbre_distance",
    "cultural_distance_metric", "pairwise_distance_matrix_cultural",
    "tradition_mds_2d", "tradition_cluster",
    "influence_graph_cultural", "borrowing_score", "fusion_potential",
    "PerformanceTradition", "VibratoType", "VibratoProfile",
    "TempoConvention", "DynamicConvention", "ArticulationProfile",
    "ImprovisationProfile", "PerformancePractice", "PRACTICE_CATALOG",
    "practice_vector", "practice_distance", "practice_similarity",
    "pairwise_practice_matrix", "classify_tradition_perf",
    "vibrato_similarity", "tempo_flexibility_score", "expressive_freedom_score",
]

from models.music_cultural.dance_rhythm import (
    DancePattern, DANCE_CATALOG, pattern_to_binary, accent_to_weight,
    pattern_similarity as dance_pattern_similarity, classify_dance,
    generate_onset_times, syncopation_score as dance_syncopation_score,
    swing_feel_from_pattern, tempo_in_range,
)
from models.music_cultural.cross_cultural_analysis import (
    Tradition, ScaleProfile, RhythmProfile,
    TraditionFingerprint, TRADITION_FINGERPRINTS,
    FusionAnalysis,
    scale_interval_similarity, tuning_deviation_12tet, microtonal_content,
    rhythm_accent_correlation, tradition_distance, tradition_similarity_matrix,
    analyze_fusion_potential, classify_tradition,
)
from models.music_cultural.modal_systems import (
    ModalFamily, Mode, CHURCH_MODES, MELODIC_MINOR_MODES, PENTATONIC_MODES,
    pitch_class_histogram, mode_template, correlate_profile, detect_mode,
    transpose_mode, modal_brightness, modal_interchange_candidates,
    characteristic_interval_strength, interval_vector, mode_distance,
    brightness_ordering, mode_subset_of, common_tones,
)
from models.music_cultural.geographic_mapping import (
    GeographicPoint, StyleRegion,
    haversine_distance, geographic_distance_matrix, feature_distance_matrix,
    mantel_test, spatial_autocorrelation, inverse_distance_weighting,
    classify_region, diffusion_gradient, cultural_boundary_detection,
    isolation_by_distance,
)
from models.music_cultural.cultural_influence_network import (
    TransmissionMode, CulturalNode, InfluenceEdge, DiffusionState,
    build_influence_graph as cin_build_influence_graph,
    cultural_feature_distance, cultural_distance_matrix as cin_cultural_distance_matrix,
    proximity_influence_graph, threshold_diffusion, continuous_diffusion,
    influence_centrality, transmission_pathways, pathway_strength, dominant_pathway,
    cultural_mds as cin_cultural_mds, innovation_rate, cultural_entropy,
    syncretism_score, influence_asymmetry, net_influence,
)
from models.music_cultural.cultural_distance import (
    CulturalDimension, TraditionProfile, CulturalDistanceResult,
    scale_distance, rhythm_distance, tuning_distance,
    ornament_distance, form_distance, texture_distance, timbre_distance,
    cultural_distance as cultural_distance_metric,
    pairwise_distance_matrix as pairwise_distance_matrix_cultural,
    tradition_mds_2d, tradition_cluster,
    influence_graph as influence_graph_cultural,
    borrowing_score, fusion_potential,
)
from models.music_cultural.performance_practice import (
    PerformanceTradition, VibratoType, VibratoProfile,
    TempoConvention, DynamicConvention, ArticulationProfile,
    ImprovisationProfile, PerformancePractice, PRACTICE_CATALOG,
    practice_vector, practice_distance, practice_similarity,
    pairwise_practice_matrix, classify_tradition as classify_tradition_perf,
    vibrato_similarity, tempo_flexibility_score, expressive_freedom_score,
)
