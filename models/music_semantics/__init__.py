"""Music semantics: genre ontology, instrument taxonomy, form analysis, affect, mood, knowledge graph."""
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
]
