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
from .evaluation import (
    RetrievalMetrics, ClassificationMetrics, EmbeddingMetrics, EvalReport,
    ModelEvaluator, evaluate_retrieval, evaluate_classification, evaluate_embeddings,
)
from .checkpoint import save_checkpoint, load_checkpoint, save_npz, load_npz
