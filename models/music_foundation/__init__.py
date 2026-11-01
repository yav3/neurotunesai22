"""Music foundation model: two-tower encoder, decoder, tokenizer, training."""
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
