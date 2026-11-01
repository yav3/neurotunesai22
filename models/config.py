"""Configuration for the NeuroFoundation model and all downstream heads."""
from __future__ import annotations
from dataclasses import dataclass, field


MODALITY_DIMS: dict[str, int] = {
    "eeg_spectral": 5,       # delta, theta, alpha, beta, gamma
    "eeg_derived": 3,        # theta/alpha ratio, spectral entropy, alpha peak freq
    "fmri_bold": 8,          # DMN, PFC, hippocampus, amygdala, motor, cerebellum, thalamus, BG
    "fmri_connectivity": 4,  # DMN-PFC, hipp-amyg, motor-BG, thal-cortical
    "blood": 7,              # NfL, GFAP, BDNF, S100B, CRP, IL6, TNFα
    "csf": 4,                # Aβ42, total tau, p-tau181, NfL-CSF
    "genomic": 2,            # APOE ε4 carrier, polygenic risk score
    "clinical": 4,           # MMSE, MoCA, Hamilton, UPDRS
    "demographics": 3,       # age, sex, BMI
}

NUM_MODALITIES = len(MODALITY_DIMS)
TOTAL_FEATURES = sum(MODALITY_DIMS.values())  # 40

NUM_DISEASE_STATES = 14
DISEASE_CODES = [
    "HC", "AD", "PD", "MDD", "EP", "MS", "ALS",
    "HD", "TBI", "IS", "SZ", "BD", "ASD", "MCI",
]


@dataclass
class FoundationConfig:
    """Hyperparameters for the NeuroFoundation transformer."""
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    use_cls_token: bool = True
    max_modalities: int = NUM_MODALITIES + 1  # +1 for [CLS]


@dataclass
class PretrainConfig:
    """Pretraining hyperparameters."""
    mask_modality_prob: float = 0.25
    mask_feature_prob: float = 0.15
    noise_std: float = 0.1
    contrastive_temperature: float = 0.07
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "mmm": 1.0,       # masked modality modeling
        "mfm": 0.5,       # masked feature modeling
        "contrastive": 0.3,
    })
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    batch_size: int = 512
    grad_accum_steps: int = 1
    fp16: bool = True
    seed: int = 42


@dataclass
class FinetuneConfig:
    """Finetuning hyperparameters for downstream tasks."""
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 50
    batch_size: int = 256
    patience: int = 10
    freeze_encoder_epochs: int = 5
    head_hidden_dim: int = 128
    head_dropout: float = 0.2
    seed: int = 42
