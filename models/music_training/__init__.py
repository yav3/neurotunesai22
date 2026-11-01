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
]
