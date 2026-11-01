"""Music foundation model: two-tower encoder, decoder, tokenizer, diffusion, generation."""
from .encoder import (
    EncoderConfig, LinearLayer, gelu, layer_norm, MultiHeadAttention,
    AudioEncoder, SemanticEncoder, GovernedBridge, MusicFoundationModel,
)
from .pretraining import (
    InfoNCELoss, MaskedFeaturePrediction, CrossModalAlignment, PretrainingObjective,
)
from .decoder import (
    DecoderConfig, MusicDecoder, ReconstructionHead, GenerationHead,
)
from .tokenizer import MusicCodebook, MusicTokenizer
from .optimizer import SGD, Adam, AdamW, GradientClipper
from .gradient import numerical_gradient, GradientAccumulator
from .positional import (
    sinusoidal_encoding, RotaryPositionalEncoding, TemporalEncoding, RelativePositionBias,
)
from .trainer import TrainerConfig, MusicTrainer, EarlyStopping, TrainingHistory
from .inference import (
    InferenceConfig, InferenceResult, GenreHead, MoodHead, VADHead,
    MusicInferencePipeline,
)
from .training import (
    TrainingConfig, TrainingMetrics, Trainer, cosine_lr, clip_gradients,
)
from .spectrogram import (
    SpectrogramEncoderConfig, SpectrogramEncoder, SpectrogramDecoder,
)
from .conditioning import (
    ConditioningConfig, ConditioningModule, FiLMLayer,
    GenreEncoder, MoodEncoder, KeyEncoder, TempoEncoder, TagEncoder,
    GENRES, MOODS, KEYS, MODES,
)
from .diffusion import (
    DiffusionConfig, LatentDiffusionModel, NoisePredictor,
)
from .frontier import FrontierConfig, MusicFrontierModel
from .frontier_model import (
    FrontierConfig as DeepFrontierConfig,
    MusicFrontierModel as DeepFrontierModel,
    ConvFeatureExtractor, TransformerEncoder, TransformerLayer,
    QuantizationModule, ContrastiveHead, ClassificationHead,
    generate_span_mask, contrastive_loss, diversity_loss,
    masked_prediction_loss as frontier_masked_prediction_loss,
    cosine_warmup_schedule, TrainingState, batch_spectrograms,
)
from .deep_trainer import (
    DeepTrainerConfig, DeepAudioDataset, DeepFrontierTrainer,
)
from .evaluation import (
    RetrievalMetrics, ClassificationMetrics, EmbeddingMetrics, EvalReport,
    ModelEvaluator, evaluate_retrieval, evaluate_classification, evaluate_embeddings,
)
from .checkpoint import save_checkpoint, load_checkpoint, save_npz, load_npz
from .frontier_pipeline import (
    FrontierPipelineConfig, FrontierDataset, SpectrogramAugmenter,
    FrontierEvaluator, FrontierEvalResult, LinearProbe,
    FrontierTrainingPipeline, quick_pretrain, pretrain_from_signals,
)
from .frontier_trainer import (
    FrontierTrainerConfig, FrontierTrainer, AudioDataset, cosine_schedule,
)
from .audio_pipeline import (
    AudioPipelineConfig, AudioSegmenter, AudioAugmenter, AudioPreprocessor,
    SyntheticAudioGenerator,
)
from .end_to_end import (
    PipelineConfig as E2EPipelineConfig,
    AudioSample as E2EAudioSample,
    TrainingResult as E2ETrainingResult,
    load_wav as e2e_load_wav,
    generate_synthetic_dataset as e2e_generate_synthetic_dataset,
    MusicFoundationPipeline as E2EMusicFoundationPipeline,
    run_demo as e2e_run_demo,
)
from .representation import (
    PoolingConfig, RepresentationOutput,
    mean_pool, attention_pool, multi_scale_pool,
    Codebook, init_codebook, quantize, update_codebook_ema, codebook_perplexity,
    generate_mask, masked_prediction_loss, contrastive_prediction_loss, sample_negatives,
    barlow_twins_loss, variance_loss, invariance_loss, covariance_loss, vicreg_loss,
    feature_pyramid, pyramid_merge,
    representation_entropy, isotropy_score, effective_rank, alignment_uniformity,
)
from .augmentation import (
    AugType, AugConfig, AugResult,
    gain_jitter, polarity_inversion, additive_noise, soft_clip,
    random_crop, fade_in_out,
    frequency_mask, time_mask, spec_augment, frequency_warp, random_eq,
    pitch_shift_resample, tempo_stretch, mixup as foundation_mixup,
    cutmix_spectrogram,
    build_augmentation_chain, default_pretraining_chain, two_view_augmentation,
)
from .attention import (
    AttentionConfig, AttentionOutput,
    scaled_dot_product_attention, multi_head_attention,
    relative_position_encoding, relative_attention_logits, relative_position_attention,
    local_attention_mask, causal_mask, causal_local_mask,
    cross_attention,
    harmonic_position_encoding, pitch_aware_attention,
    random_fourier_features, performer_attention,
    attention_head_diversity, attention_sparsity, effective_attention_span,
    init_attention_weights,
)
from .feature_integration import (
    FeatureIntegrationConfig, FeatureProjection, GatedFusion, AttentionFusion,
    MultiStreamFeatureIntegrator, AudioFeatureExtractor, build_default_integrator,
    compute_mfcc, compute_chroma, compute_spectral_contrast, compute_onset_strength,
    compute_spectral_centroid, compute_spectral_bandwidth, compute_spectral_rolloff,
    compute_zero_crossing_rate,
)
