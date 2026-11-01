"""Music training pipeline: datasets, objectives, evaluation, augmentation, checkpoints."""
from models.music_training.dataset import MusicDataset, FeatureNormalizer
from models.music_training.objectives import ContrastiveLoss, ReconstructionLoss, MultiTaskHead
from models.music_training.scheduler import CosineAnnealingScheduler, WarmupScheduler, OneCycleScheduler
from models.music_training.evaluator import (
    genre_accuracy,
    mood_quadrant_accuracy,
    key_detection_accuracy,
    tempo_estimation_accuracy,
    embedding_quality,
    recommendation_ndcg,
)
from models.music_training.checkpoint import CheckpointManager
from models.music_training.augmentation import (
    pitch_shift_features,
    tempo_scale_features,
    noise_injection,
    feature_dropout,
    mixup,
)

from models.music_training.curriculum import (
    DifficultyMetric, CurriculumStrategy, SampleDifficulty,
    score_by_loss, score_by_entropy, score_by_complexity,
    CurriculumScheduler, SelfPacedWeight,
)

__all__ = [
    "MusicDataset", "FeatureNormalizer",
    "ContrastiveLoss", "ReconstructionLoss", "MultiTaskHead",
    "CosineAnnealingScheduler", "WarmupScheduler", "OneCycleScheduler",
    "genre_accuracy", "mood_quadrant_accuracy", "key_detection_accuracy",
    "tempo_estimation_accuracy", "embedding_quality", "recommendation_ndcg",
    "CheckpointManager",
    "pitch_shift_features", "tempo_scale_features", "noise_injection",
    "feature_dropout", "mixup",
    "DifficultyMetric", "CurriculumStrategy", "SampleDifficulty",
    "score_by_loss", "score_by_entropy", "score_by_complexity",
    "CurriculumScheduler", "SelfPacedWeight",
    "AugmentationType", "ContrastivePair", "TripletSample",
    "cosine_similarity", "pairwise_cosine_similarity",
    "nt_xent_loss", "triplet_loss", "batch_triplet_loss",
    "hard_negative_mining", "MusicAugmentor", "ContrastiveTrainingState",
    "alignment_metric", "uniformity_metric",
    "intonation_error_cents", "intonation_score", "intonation_bias",
    "onset_timing_error", "rhythm_precision_score", "rhythm_consistency",
    "SightReadingFeatures", "sight_reading_difficulty",
    "PracticeItem", "sm2_update", "schedule_practice_session",
    "PerformanceSnapshot", "performance_trend", "improvement_rate",
    "weakness_detection",
]

from .skill_assessment import (
    intonation_error_cents, intonation_score, intonation_bias,
    onset_timing_error, rhythm_precision_score, rhythm_consistency,
    SightReadingFeatures, sight_reading_difficulty,
    PracticeItem, sm2_update, schedule_practice_session,
    PerformanceSnapshot, performance_trend, improvement_rate,
    weakness_detection,
)
from .contrastive_learning import (
    AugmentationType, ContrastivePair, TripletSample,
    cosine_similarity, pairwise_cosine_similarity,
    nt_xent_loss, triplet_loss, batch_triplet_loss,
    hard_negative_mining, MusicAugmentor, ContrastiveTrainingState,
    alignment_metric, uniformity_metric,
)
from .curriculum_learning import (
    DifficultyScorer, CurriculumScheduler as CurriculumScheduler2,
    AntiCurriculumScheduler, self_paced_weight,
    MixupAugmentor, LearningRateSchedule,
)
