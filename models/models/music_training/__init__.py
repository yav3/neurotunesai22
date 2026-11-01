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

__all__ = [
    "MusicDataset", "FeatureNormalizer",
    "ContrastiveLoss", "ReconstructionLoss", "MultiTaskHead",
    "CosineAnnealingScheduler", "WarmupScheduler", "OneCycleScheduler",
    "genre_accuracy", "mood_quadrant_accuracy", "key_detection_accuracy",
    "tempo_estimation_accuracy", "embedding_quality", "recommendation_ndcg",
    "CheckpointManager",
    "pitch_shift_features", "tempo_scale_features", "noise_injection",
    "feature_dropout", "mixup",
]
