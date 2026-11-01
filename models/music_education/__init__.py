"""Music education: pedagogy, ear training, sight-reading, practice planning, assessment."""
from models.music_education.ear_training import (
    IntervalQuality, IntervalQuestion, ChordQuestion, DictationExercise,
    EarTrainingScore, interval_name, interval_difficulty,
    generate_interval_question, score_interval_answer,
    chord_pitches, generate_chord_question, score_chord_answer,
    generate_dictation, score_dictation, contour_score,
    adaptive_difficulty, session_summary,
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
    "PracticeSM2State", "practice_sm2_update", "schedule_passages",
    "PracticeErrorType", "NoteError",
    "detect_pitch_errors", "detect_timing_errors", "detect_dynamic_errors",
    "TempoAnalysis", "analyze_tempo_consistency",
    "PracticeBlock", "segment_practice_blocks",
    "improvement_curve", "diminishing_returns_point",
    "DifficultyEstimate", "estimate_difficulty",
    "cs_SkillDomain", "Difficulty", "Skill", "cs_LearnerState",
    "cs_Exercise", "cs_LessonPlan", "DIFFICULTY_LEVELS",
    "build_prerequisite_graph", "cs_topological_sort",
    "prerequisite_satisfaction", "readiness_score", "next_skills",
    "cs_spaced_repetition_interval", "cs_update_mastery",
    "difficulty_progression", "zone_of_proximal_development",
    "generate_lesson_plan", "learning_efficiency",
    "cs_skill_gap_analysis", "estimated_completion_time", "mastery_decay",
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
from models.music_education.practice_analysis import (
    SM2State as PracticeSM2State,
    sm2_update as practice_sm2_update,
    schedule_passages,
    ErrorType as PracticeErrorType,
    NoteError, detect_pitch_errors, detect_timing_errors, detect_dynamic_errors,
    TempoAnalysis, analyze_tempo_consistency,
    PracticeBlock, segment_practice_blocks,
    improvement_curve, diminishing_returns_point,
    DifficultyEstimate, estimate_difficulty,
)
from models.music_education.curriculum_sequencing import (
    SkillDomain as cs_SkillDomain,
    Difficulty, Skill,
    LearnerState as cs_LearnerState,
    Exercise as cs_Exercise,
    LessonPlan as cs_LessonPlan,
    DIFFICULTY_LEVELS,
    build_prerequisite_graph,
    topological_sort as cs_topological_sort,
    prerequisite_satisfaction, readiness_score, next_skills,
    spaced_repetition_interval as cs_spaced_repetition_interval,
    update_mastery as cs_update_mastery,
    difficulty_progression, zone_of_proximal_development,
    generate_lesson_plan, learning_efficiency,
    skill_gap_analysis as cs_skill_gap_analysis,
    estimated_completion_time, mastery_decay,
)
from models.music_education.adaptive_curriculum import (
    MusicSkill, SKILL_LEVELS, skill_level_label,
    prerequisites_satisfied, recommended_skills,
    ResponseRecord, estimate_difficulty_mle, estimate_discrimination_mle,
    estimate_2pl_parameters, SPRTState, run_sprt,
    power_law_predict, fit_power_law, predict_mastery_trial,
    CurriculumItem, sm2_music_update, schedule_curriculum_session,
    ProgressSnapshot, accuracy_trajectory, time_on_task_summary,
    skill_gap_analysis, compute_progress_snapshot,
)
