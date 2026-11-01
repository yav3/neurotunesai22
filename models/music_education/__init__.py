"""Music education: pedagogy, ear training, sight-reading, practice planning, assessment."""
from models.music_education.ear_training import (
    IntervalTrainer, ChordTrainer, RhythmDictation,
)
from models.music_education.sight_reading import (
    SightReadingGenerator, DifficultyEstimator,
)
from models.music_education.practice import (
    PracticePlanner, SpacedRepetition, SkillTracker,
)
from models.music_education.assessment import (
    RubricCriterion, PerformanceRubric, SkillAssessment,
)

from models.music_education.exercise_generator import (
    DifficultyLevel, ExerciseType, Exercise, ExerciseGenerator, grade_answer,
)
from models.music_education.progress_tracker import (
    LearningProfile, SkillAttempt,
)
from models.music_education.curriculum import (
    TopicStatus, Topic, LearnerState, LearningPath, CurriculumGraph,
    spaced_review_schedule, forgetting_curve, optimal_review_time,
)
from models.music_education.adaptive_difficulty import (
    SkillDomain, SkillEstimate, ExerciseItem,
    irt_probability, irt_information,
    update_ability_elo, select_next_item, select_max_information,
    ZPDRange, compute_zpd, filter_zpd_items,
    LearnerProfile, DifficultySchedule, AdaptiveSession,
    generate_difficulty_curve,
)

__all__ = [
    "IntervalTrainer", "ChordTrainer", "RhythmDictation",
    "SightReadingGenerator", "DifficultyEstimator",
    "PracticePlanner", "SpacedRepetition", "SkillTracker",
    "RubricCriterion", "PerformanceRubric", "SkillAssessment",
    "DifficultyLevel", "ExerciseType", "Exercise", "ExerciseGenerator", "grade_answer",
    "LearningProfile", "SkillAttempt",
    "SkillDomain", "SkillEstimate", "ExerciseItem",
    "irt_probability", "irt_information",
    "update_ability_elo", "select_next_item", "select_max_information",
    "ZPDRange", "compute_zpd", "filter_zpd_items",
    "LearnerProfile", "DifficultySchedule", "AdaptiveSession",
    "generate_difficulty_curve",
    "TopicStatus", "Topic", "LearnerState", "LearningPath", "CurriculumGraph",
    "spaced_review_schedule", "forgetting_curve", "optimal_review_time",
    "IntervalQuality", "Interval", "INTERVALS", "IntervalLearnerState",
    "generate_exercise", "spaced_repetition_priority", "confusion_matrix",
    "mastery_level", "recommend_session_plan",
    "ClefType", "KeySignatureInfo", "TimeSignatureInfo", "NoteOnStaff",
    "staff_position_to_midi", "midi_to_staff_position",
    "identify_key_signature", "analyze_reading_difficulty",
    "generate_reading_exercise", "common_reading_errors",
    "MasteryLevel", "BloomLevel", "Concept",
    "MUSIC_THEORY_GRAPH", "build_concept_graph", "topological_order",
    "prerequisites_met", "knowledge_gaps", "lesson_sequence",
    "update_mastery", "bloom_level_for_exercise", "recommend_next_concepts",
    "ErrorType", "PerformanceError", "PerformanceRubricScore",
    "dtw_alignment", "classify_errors", "score_performance",
    "difficulty_adjustment", "ProgressRecord",
]

from models.music_education.interval_training import (
    IntervalQuality, Interval, INTERVALS,
    LearnerState as IntervalLearnerState,
    generate_exercise, spaced_repetition_priority, confusion_matrix,
    mastery_level, recommend_session_plan,
)
from models.music_education.spaced_repetition import (
    ReviewItem, ReviewResult, sm2_update,
    retention_probability, optimal_interval,
    LeitnerBox, forgetting_curve,
    schedule_session, difficulty_estimate,
)
from models.music_education.music_theory_tutor import (
    MasteryLevel, BloomLevel, Concept,
    MUSIC_THEORY_GRAPH, build_concept_graph, topological_order,
    prerequisites_met, knowledge_gaps, lesson_sequence,
    update_mastery, bloom_level_for_exercise, recommend_next_concepts,
)
from models.music_education.performance_assessment import (
    ErrorType, PerformanceError, RubricScore as PerformanceRubricScore,
    dtw_alignment, classify_errors, score_performance,
    difficulty_adjustment, ProgressRecord,
)
from models.music_education.score_reading import (
    ClefType, KeySignatureInfo, TimeSignatureInfo, NoteOnStaff,
    staff_position_to_midi, midi_to_staff_position,
    identify_key_signature, analyze_reading_difficulty,
    generate_reading_exercise, common_reading_errors,
)
