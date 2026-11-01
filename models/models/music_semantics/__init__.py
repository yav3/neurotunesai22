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
]

from .cross_modal import (
    ColourAssociation, PITCH_HUE_MAP, pitch_to_colour, chord_colour, colour_distance,
    TextureAssociation, spectral_to_texture, texture_distance,
    ShapeAssociation, BOUBA_KIKI_MAP, articulation_to_shape, dynamics_to_shape, shape_distance,
    cross_modal_coherence, synesthesia_profile,
)
