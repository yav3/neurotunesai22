"""Music semantics: genre ontology, instrument taxonomy, form analysis, affect, mood, knowledge graph, concept embedding, mood ontology."""
from models.music_semantics.genre_ontology import (
    GenreNode, GENRE_TREE, genre_distance, genre_ancestors, genre_descendants,
    common_ancestor, features_to_genre,
)
from models.music_semantics.instrument_taxonomy import (
    Instrument, INSTRUMENT_CATALOG, instrument_compatibility, ensemble_suggestion,
)
from models.music_semantics.form_analysis import (
    MUSICAL_FORMS, FormSection, classify_form, form_complexity,
)
from models.music_semantics.musical_affect import (
    AffectPoint, EmotionArc, CircumplexModel, MusicalAffectAnalyzer,
    HEVNER_CLUSTERS, THAYER_QUADRANTS,
    affect_distance, arc_similarity,
)
from models.music_semantics.mood_tags import (
    MoodCategory, MoodTag, MOOD_VAD_MAP, MUSIC_TAG_TAXONOMY,
    TagProfile, vad_to_mood, tag_similarity,
)
from models.music_semantics.music_knowledge_graph import (
    EntityType, RelationType, Entity, Relation, MusicKnowledgeGraph,
)
from models.music_semantics.concept_embedding import (
    MusicalConcept, ConceptEmbedding,
    CONCEPT_CATEGORIES, CONCEPT_REGISTRY, EMBEDDING_DIM,
    concept_similarity, nearest_concepts, analogy,
)
from models.music_semantics.mood_ontology import (
    MoodNode, MoodTrajectory, MOOD_TREE, VONNEGUT_ARCS,
    mood_distance, mood_ancestors, mood_from_vad, mood_from_features,
)
from models.music_semantics.lyrics_analysis import (
    syllable_count, syllable_density,
    rhyme_similarity, detect_rhyme_scheme,
    line_similarity_matrix, detect_repetition, RepetitionStructure,
    sentiment_polarity, sentiment_trajectory, SentimentResult,
    lyric_melody_alignment,
)
from models.music_semantics.narrative_arc import (
    tension_curve, smooth_tension,
    classify_arc, ArcType, ArcClassification,
    emotional_trajectory, vad_distance, emotional_range, emotional_velocity, VADPoint,
    segment_entropy, surprise_curve, structural_surprise_score,
)
__all__ = [
    "GenreNode", "GENRE_TREE", "genre_distance", "genre_ancestors", "genre_descendants",
    "common_ancestor", "features_to_genre",
    "Instrument", "INSTRUMENT_CATALOG", "instrument_compatibility", "ensemble_suggestion",
    "MUSICAL_FORMS", "FormSection", "classify_form", "form_complexity",
    "AffectPoint", "EmotionArc", "CircumplexModel", "MusicalAffectAnalyzer",
    "HEVNER_CLUSTERS", "THAYER_QUADRANTS", "affect_distance", "arc_similarity",
    "MoodCategory", "MoodTag", "MOOD_VAD_MAP", "MUSIC_TAG_TAXONOMY",
    "TagProfile", "vad_to_mood", "tag_similarity",
    "EntityType", "RelationType", "Entity", "Relation", "MusicKnowledgeGraph",
    "MusicalConcept", "ConceptEmbedding",
    "CONCEPT_CATEGORIES", "CONCEPT_REGISTRY", "EMBEDDING_DIM",
    "concept_similarity", "nearest_concepts", "analogy",
    "MoodNode", "MoodTrajectory", "MOOD_TREE", "VONNEGUT_ARCS",
    "mood_distance", "mood_ancestors", "mood_from_vad", "mood_from_features",
    "syllable_count", "syllable_density",
    "rhyme_similarity", "detect_rhyme_scheme",
    "line_similarity_matrix", "detect_repetition", "RepetitionStructure",
    "sentiment_polarity", "sentiment_trajectory", "SentimentResult",
    "lyric_melody_alignment",
    "tension_curve", "smooth_tension",
    "classify_arc", "ArcType", "ArcClassification",
    "emotional_trajectory", "vad_distance", "emotional_range", "emotional_velocity", "VADPoint",
    "segment_entropy", "surprise_curve", "structural_surprise_score",
    "ColourAssociation", "PITCH_HUE_MAP", "pitch_to_colour", "chord_colour", "colour_distance",
    "TextureAssociation", "spectral_to_texture", "texture_distance",
    "ShapeAssociation", "BOUBA_KIKI_MAP", "articulation_to_shape", "dynamics_to_shape", "shape_distance",
    "cross_modal_coherence", "synesthesia_profile",
    "MusicOntology", "MusicKnowledgeBase",
    "OntologyNode", "OntologyEdge",
    "NodeType", "OntologyRelationType",
    "SemanticDimension", "SemanticVector", "SemanticSpace",
    "semantic_distance", "tag_cooccurrence_matrix", "latent_semantic_analysis",
    "concept_graph_sem", "analogy_completion", "interpolate_semantic",
    "semantic_trajectory", "cross_modal_alignment_sem",
    "ConceptType", "MusicConcept", "SimilarityResult",
    "ss_feature_vector", "feature_similarity", "taxonomic_similarity",
    "wu_palmer_similarity", "distributional_similarity",
    "ss_pointwise_mutual_information", "ss_concept_embedding",
    "embedding_similarity_matrix", "metaphor_mapping",
    "combined_similarity", "pairwise_concept_similarity", "concept_clustering",
    "EmotionalDimension", "ETArcType", "EmotionalState", "EmotionalArc", "ETTensionCurve",
    "valence_from_features", "arousal_from_features", "tension_from_harmony",
    "compute_emotional_trajectory", "et_tension_curve",
    "climax_strength", "emotional_distance", "et_emotional_velocity",
    "emotional_smoothness", "et_arc_similarity",
    "narrative_segments", "et_emotional_range",
    "StyleCategory", "sr_HarmonicFunction", "FeatureProfile", "StyleInference",
    "HarmonicRule", "CompositionEvaluation",
    "classify_style", "classify_all_styles",
    "sr_harmonic_function", "is_authentic_cadence", "is_plagal_cadence",
    "is_deceptive_cadence", "is_half_cadence", "functional_progression_score",
    "evaluate_melody", "evaluate_harmonic_progression",
]

from .music_ontology import (
    MusicOntology, MusicKnowledgeBase,
    OntologyNode, OntologyEdge,
    NodeType, RelationType as OntologyRelationType,
)
from .semantic_embedding import (
    SemanticDimension, SemanticVector, SemanticSpace,
    semantic_distance, tag_cooccurrence_matrix, latent_semantic_analysis,
    concept_graph as concept_graph_sem, analogy_completion, interpolate_semantic,
    semantic_trajectory, cross_modal_alignment as cross_modal_alignment_sem,
)
from .cross_modal import (
    ColourAssociation, PITCH_HUE_MAP, pitch_to_colour, chord_colour, colour_distance,
    TextureAssociation, spectral_to_texture, texture_distance,
    ShapeAssociation, BOUBA_KIKI_MAP, articulation_to_shape, dynamics_to_shape, shape_distance,
    cross_modal_coherence, synesthesia_profile,
)
from .emotional_trajectory import (
    EmotionalDimension, ArcType as ETArcType,
    EmotionalState, EmotionalArc, TensionCurve as ETTensionCurve,
    valence_from_features, arousal_from_features, tension_from_harmony,
    emotional_trajectory as compute_emotional_trajectory,
    tension_curve as et_tension_curve,
    climax_strength, emotional_distance, emotional_velocity as et_emotional_velocity,
    emotional_smoothness, arc_similarity as et_arc_similarity,
    narrative_segments, emotional_range as et_emotional_range,
)
from .symbolic_reasoning import (
    StyleCategory, HarmonicFunction as sr_HarmonicFunction,
    FeatureProfile, StyleInference, HarmonicRule, CompositionEvaluation,
    classify_style, classify_all_styles,
    harmonic_function as sr_harmonic_function,
    is_authentic_cadence, is_plagal_cadence, is_deceptive_cadence, is_half_cadence,
    functional_progression_score,
    evaluate_melody, evaluate_harmonic_progression,
)
from .semantic_similarity import (
    ConceptType, MusicConcept, SimilarityResult,
    feature_vector as ss_feature_vector, feature_similarity,
    taxonomic_similarity, wu_palmer_similarity, distributional_similarity,
    pointwise_mutual_information as ss_pointwise_mutual_information,
    concept_embedding as ss_concept_embedding,
    embedding_similarity_matrix, metaphor_mapping,
    combined_similarity, pairwise_concept_similarity, concept_clustering,
)
