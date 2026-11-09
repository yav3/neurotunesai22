"""Music foundation model: two-tower encoder, decoder, tokenizer, diffusion, generation."""
from .encoder import (
    EncoderConfig, LinearLayer, gelu, layer_norm, MultiHeadAttention,
    AudioEncoder, SemanticEncoder, GovernedBridge, MusicFoundationModel,
)
from .pretraining import (
    InfoNCELoss, MaskedFeaturePrediction, CrossModalAlignment, PretrainingObjective,
    VICRegLoss, BarlowTwinsLoss, BYOLLoss, MaskedAcousticModel,
    MultiTaskPretraining, CodebookDiversityLoss, TemporalOrderPrediction,
)
from .decoder import (
    DecoderConfig, SamplingConfig, KVCache, MusicDecoder,
    ReconstructionHead, GenerationHead,
    MaskedMultiHeadAttention, CrossAttention, TransformerDecoderLayer,
    top_k_filter, top_p_filter, min_p_filter, typical_p_filter,
    apply_repetition_penalty, sample_logits, BeamHypothesis, beam_search,
    classifier_free_guidance,
)
from .tokenizer import (
    MusicCodebook, MusicTokenizer,
    EMACodebook, gumbel_softmax_quantize, FiniteScalarQuantizer,
    ProductQuantizer, ResidualVQ,
)
from .optimizer import (
    SGD, Adam, AdamW, GradientClipper,
    LAMB, Adafactor, Lion,
    cosine_annealing, cosine_warmup, linear_warmup_decay, one_cycle,
    inverse_sqrt, exponential_decay,
    ExponentialMovingAverage, ParameterGroupOptimizer,
)
from .gradient import (
    numerical_gradient, GradientAccumulator,
    spsa_gradient, nes_gradient, add_gradient_noise, centralize_gradients,
    GradientVarianceTracker, LookaheadGradient, project_gradient_to_subspace,
    empirical_fisher_diagonal, pcgrad,
)
from .positional import (
    sinusoidal_encoding, RotaryPositionalEncoding, TemporalEncoding, RelativePositionBias,
    alibi_slopes, alibi_bias, ConditionalPositionalEncoding,
    MusicalMeterEncoding, PitchPositionalEncoding, MultiScaleTemporalEncoding,
    log_relative_position_buckets, LogRelativePositionBias, NTKAwareRoPE,
)
from .trainer import TrainerConfig, MusicTrainer, EarlyStopping, TrainingHistory
from .inference import (
    InferenceConfig, InferenceResult, GenreHead, MoodHead, VADHead,
    InstrumentHead, TempoHead, PlattCalibrator,
    MusicInferencePipeline, EnsemblePipeline, EmbeddingIndex,
    embedding_analogy, embedding_interpolate, estimate_uncertainty,
)
from .training import (
    TrainingConfig, TrainingMetrics, Trainer, cosine_lr, clip_gradients,
)
from .spectrogram import (
    SpectrogramEncoderConfig, SpectrogramEncoder, SpectrogramDecoder,
    compute_stft, istft, griffin_lim, pcen,
    compute_cqt, erb_scale, gammatone_filterbank, compute_gammatone_spectrogram,
    multi_resolution_stft_loss, instantaneous_frequency, group_delay,
    spec_augment,
)
from .conditioning import (
    ConditioningConfig, ConditioningModule, FiLMLayer,
    GenreEncoder, MoodEncoder, KeyEncoder, TempoEncoder, TagEncoder,
    AdaINLayer, CrossAttentionConditioning, TemporalConditioning,
    HierarchicalConditioning, ConditionDropout, GlobalStyleTokens,
    condition_interpolate, condition_slerp, mix_conditions,
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
from .multistream_trainer import (
    MultiStreamTrainerConfig, MultiStreamDataset, MultiStreamFeatureCache,
    MultiStreamPretrainer, quick_multistream_train,
)
from .audio_codec import (
    CodecConfig, AudioEncoder as CodecEncoder, AudioDecoder as CodecDecoder,
    ResidualVectorQuantizer, NeuralAudioCodec, CodecTrainer,
    spectral_convergence_loss, log_magnitude_loss, multi_scale_spectral_loss,
    quick_codec_test,
)
from .generation import (
    GenerationConfig, TokenSampler, AutoregressiveGenerator,
    CFGGenerator, BeamSearchGenerator, MusicContinuation,
    ConstrainedGenerator, generate_music_tokens,
)
from .source_separation import (
    SeparationConfig, SourceSeparator, MaskEstimator, WienerFilter,
    UNetSeparator, HarmonicPercussiveSeparation, RepetitionSeparation,
    evaluate_bss,
)
from .evaluation import (
    RetrievalMetrics, ClassificationMetrics, EmbeddingMetrics, EvalReport,
    ModelEvaluator, evaluate_retrieval, evaluate_classification, evaluate_embeddings,
)
from .checkpoint import (
    save_checkpoint, load_checkpoint, save_npz, load_npz,
    save_binary, load_binary, diff_params, apply_diff,
    save_diff_checkpoint, average_checkpoints, exponential_average,
    interpolate_params, param_stats, total_parameters, param_norm,
    params_equal, CheckpointManager,
)
from .frontier_pipeline import (
    FrontierPipelineConfig, FrontierDataset, SpectrogramAugmenter,
    FrontierEvaluator, FrontierEvalResult, LinearProbe,
    FrontierTrainingPipeline, quick_pretrain, pretrain_from_signals,
    FeatureEnhancedPipeline, quick_multi_stream_test,
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
from .data_pipeline import (
    AudioClip, SpectrogramConfig, BatcherConfig, AugScheduleConfig,
    PipelineConfig, StreamingBatcher, OnTheFlyAugmenter, CurriculumSampler,
    AudioPipeline, load_wav, load_directory, generate_synthetic_clips,
    compute_mel_spectrogram, mel_to_audio, process_batch_spectrograms,
    compute_dataset_stats, apply_global_normalization,
    stratified_split, k_fold_indices,
)
from .benchmark import (
    BenchmarkConfig, BenchmarkReport, BenchmarkSuite,
    ProbeClassifier, RetrievalBenchmark, ClusteringBenchmark,
    centered_kernel_alignment, intrinsic_dimensionality_mle,
    isotropy_score, effective_rank, representation_similarity_analysis,
    normalized_mutual_information, adjusted_rand_index, silhouette_score,
    loss_landscape_1d, gradient_signal_to_noise, convergence_analysis,
)
from .uncertainty import (
    EigenstateEnsemble, FeatureSDE, euler_maruyama, sde_uncertainty,
    build_learned_sde, ContradictionDetector, learn_selection_rule,
    SelectionRuleFilter, counterfactual_embedding, causal_sensitivity,
    counterfactual_explanation, UncertaintyWrapper, UncertaintyPrediction,
    EnsemblePrediction, train_ensemble, ensemble_predict,
    STANDARD_AUGMENTATIONS,
)
from .robustness import (
    would_flip, test_permutation_invariance, test_rescaling_invariance,
    test_pitch_shift_equivariance, expected_calibration_error, brier_score,
    reliability_diagram, calibrate_conformal, conformal_predict,
    conformal_coverage, temporal_split, drift_detector, evaluate_under_drift,
    RobustnessReport, run_robustness_suite,
)
from .scaling import (
    TripletMiner, DiversityBatchSampler, CurriculumSampler as ScalingCurriculumSampler,
    StratifiedBatchSampler, prune_triplets, efficient_triplet_loss,
    greedy_mis, benchmark_mis, benchmark_scaling,
)
from .deployment import (
    EngineSpec, EngineVersion, EngineRegistry, ActionLayer,
    CanaryConfig, CanaryDeployment, MethodCard, generate_method_cards,
    ModelSnapshot, SnapshotStore, ModelArtifact, ModelRegistry,
)
from .sdk import (
    NeuroTunesModel, pipeline_embed, pipeline_analyze, pipeline_similarity_matrix,
)
from .data_ingest import (
    IngestConfig, AudioFile, CorpusBuilder,
    decode_audio, scan_local_audio, clips_to_batch,
    compute_corpus_stats,
)
from .agents import (
    AgentResult, MusicAnalysisAgent, QualityAssessmentAgent,
    SimilaritySearchAgent, AnnotationAgent, PipelineAgent,
    AgentRegistry, run_all_agents,
)
from .sandbox import (
    TestCase, SandboxMetrics, SandboxReport, TestingSandbox,
    generate_html_report,
)
from .testing_sandbox import (
    SandboxObserver, ModelTestSuite, run_sandbox,
)
from .webhook_auth import (
    generate_signing_key, sign_payload, verify_signature,
    NonceTracker, APIKeyStore, APIKeyRecord,
    WebhookConfig, create_webhook_middleware,
)
from .interactive_demo import generate_dashboard_html
from .demo_server import create_app
from .demo_client import run_local_demo, run_remote_demo, DemoReport
from .generation_demo import (
    DemoConfig as GenerationDemoConfig,
    GenerationResult, DemoReport as GenerationDemoReport,
    generate_seed_audio, run_generation, run_batch_generation,
    round_trip_fidelity, run_full_demo as run_generation_demo,
    save_wav, compute_mel, mel_to_magnitude,
)
from .train_and_generate import (
    TrainGenConfig, TrainGenReport, train_and_generate,
    CHECKPOINT_VERSION, reconstruct_from_checkpoint, verify_checkpoint,
)
