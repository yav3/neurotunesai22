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

__all__ = [
    "IntervalTrainer", "ChordTrainer", "RhythmDictation",
    "SightReadingGenerator", "DifficultyEstimator",
    "PracticePlanner", "SpacedRepetition", "SkillTracker",
    "RubricCriterion", "PerformanceRubric", "SkillAssessment",
    "DifficultyLevel", "ExerciseType", "Exercise", "ExerciseGenerator", "grade_answer",
    "LearningProfile", "SkillAttempt",
]
