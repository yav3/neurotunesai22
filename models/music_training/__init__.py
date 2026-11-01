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
    "DifficultyLevel", "SR_Metrics", "Exercise",
    "sr_rhythmic_complexity", "intervallic_difficulty",
    "accidental_density", "texture_density", "range_difficulty",
    "tempo_difficulty", "estimate_difficulty",
    "generate_exercise", "adaptive_exercise_sequence",
    "reading_speed_estimate", "sr_weakness_profile",
    "SkillLevel", "LearningObjective", "StudentState", "CurriculumItem",
    "zone_of_proximal_development", "spacing_effect_weight",
    "interleaving_benefit", "mastery_probability",
    "skill_decay", "difficulty_calibration",
    "sequence_curriculum", "practice_schedule", "skill_progress_forecast",
    "LossResult", "spectral_convergence", "log_stft_magnitude_loss",
    "multi_resolution_stft_loss", "mel_weighted_loss", "reconstruction_loss",
    "kl_divergence_gaussian", "vae_loss", "cl_contrastive_loss",
    "feature_matching_loss", "adversarial_loss_generator",
    "adversarial_loss_discriminator", "weighted_mse", "focal_loss",
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
from .practice_analytics import (
    PracticeSession, PracticeReport, PracticeEvent, PracticeStreak,
    WeakSpot, SM2Card,
    tempo_stability, tempo_accuracy as train_tempo_accuracy,
    note_accuracy, note_accuracy_over_time, tempo_accuracy_over_time,
    error_density, identify_problem_spots, learning_curve,
    fit_learning_curve, spaced_repetition_schedule, analyze_practice,
    detect_weak_spots, improvement_trend,
    schedule_review, quality_from_accuracy,
    session_efficiency, passage_mastery, practice_summary,
)
from .ear_training import (
    INTERVAL_NAMES, INTERVAL_SEMITONES, ChordQuality, CHORD_INTERVALS,
    IntervalResponse, ChordResponse,
    score_interval_recognition, interval_confusion_matrix,
    identify_chord_quality, score_chord_identification,
    MelodicDictationResult, HarmonicDictationResult,
    EloSkill, INTERVAL_DIFFICULTY, CHORD_DIFFICULTY,
    select_exercise_difficulty, generate_interval_exercise,
    generate_chord_exercise, generate_rhythm_exercise,
)
from .practice_curriculum import (
    MusicalDimension, PieceDifficulty,
    estimate_rhythmic_difficulty, estimate_harmonic_difficulty,
    estimate_technical_difficulty, MasteryState, ZPDSequencer,
    spaced_repetition_interval, forgetting_curve,
)
from .sight_reading import (
    DifficultyLevel, SightReadingMetrics as SR_Metrics, Exercise,
    rhythmic_complexity as sr_rhythmic_complexity,
    intervallic_difficulty, accidental_density, texture_density,
    range_difficulty, tempo_difficulty, estimate_difficulty,
    generate_exercise, adaptive_exercise_sequence,
    reading_speed_estimate, weakness_profile as sr_weakness_profile,
)
from .curriculum_sequencing import (
    SkillLevel, LearningObjective, StudentState, CurriculumItem,
    zone_of_proximal_development, spacing_effect_weight,
    interleaving_benefit, mastery_probability,
    skill_decay, difficulty_calibration,
    sequence_curriculum, practice_schedule, skill_progress_forecast,
)
from models.music_training.loss_functions import (
    LossResult, spectral_convergence, log_stft_magnitude_loss,
    multi_resolution_stft_loss, mel_weighted_loss, reconstruction_loss,
    kl_divergence_gaussian, vae_loss, contrastive_loss as cl_contrastive_loss,
    triplet_loss, feature_matching_loss, adversarial_loss_generator,
    adversarial_loss_discriminator, hard_negative_mining, weighted_mse, focal_loss,
)
