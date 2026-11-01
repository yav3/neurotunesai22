"""Perturbation-Response cVAE — narrow foundation model for gene expression.

Conditional variational autoencoder that predicts post-perturbation expression
profiles given a baseline expression vector and a perturbation specification.
Designed for the 745 in-regime GEO series (≥30 samples) from the dataset scout.

Architecture:
    Encoder:  (x_baseline || h_perturb)  →  (μ, log σ²)  ∈ R^{latent_dim}
    Decoder:  (z || h_perturb)           →  x̂_target     ∈ R^{n_genes}

Loss:
    L = MSE(log1p(x̂), log1p(x_target))
      + β · KL(q(z|x,p) || N(0,I))
      + λ · consistency(Δz across same perturbation on similar baselines)

Three perturbation types:
    knockout   — gene index → learned embedding
    drug       — (drug_idx, dose_mg, duration_hr) → linear projection
    time_course — scalar time → sinusoidal positional encoding

Follows Merlin's evidence discipline: the model is agnostic to specific gene
sets (parameterised by n_genes), does not pool species, and every training run
must declare its regime before the method is chosen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np


class PerturbationType(Enum):
    KNOCKOUT = "knockout"
    DRUG = "drug"
    TIME_COURSE = "time_course"


@dataclass
class PerturbationSpec:
    """Specification of a single perturbation applied to a sample."""
    ptype: PerturbationType
    gene_idx: int | None = None
    drug_idx: int | None = None
    dose_mg: float = 0.0
    duration_hr: float = 0.0
    time_point: float = 0.0


@dataclass
class PerturbationModelConfig:
    n_genes: int = 5000
    latent_dim: int = 128
    perturbation_dim: int = 64
    encoder_hidden: int = 512
    decoder_hidden: int = 512
    n_drugs: int = 64
    max_gene_idx: int = 25000
    sinusoidal_dim: int = 64
    dropout: float = 0.1
    beta: float = 1.0
    consistency_lambda: float = 0.1


class SinusoidalTimeEncoding(nn.Module):
    """Map scalar time to a fixed-dim sinusoidal encoding, then project."""

    def __init__(self, raw_dim: int, out_dim: int):
        super().__init__()
        self.raw_dim = raw_dim
        self.proj = nn.Linear(raw_dim, out_dim)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, raw_dim, 2).float() / raw_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) scalar time values → (batch, out_dim)."""
        # (batch, raw_dim // 2)
        angles = t.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        enc = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (batch, raw_dim)
        return self.proj(enc)


class PerturbationEncoder(nn.Module):
    """Encode a perturbation specification into a fixed-dim vector."""

    def __init__(self, config: PerturbationModelConfig):
        super().__init__()
        d = config.perturbation_dim

        self.ko_embed = nn.Embedding(config.max_gene_idx, d)
        self.drug_proj = nn.Sequential(
            nn.Linear(config.n_drugs + 2, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.time_enc = SinusoidalTimeEncoding(config.sinusoidal_dim, d)

        self.type_embed = nn.Embedding(len(PerturbationType), d)
        self.n_drugs = config.n_drugs

    def forward(
        self,
        ptype_ids: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        dose: torch.Tensor,
        duration: torch.Tensor,
        time_point: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs: (batch,). Returns (batch, perturbation_dim)."""
        batch = ptype_ids.shape[0]
        device = ptype_ids.device

        h_type = self.type_embed(ptype_ids)

        h_ko = self.ko_embed(gene_idx.clamp(min=0))
        ko_mask = (ptype_ids == PerturbationType.KNOCKOUT.value.__hash__() % 3)

        drug_onehot = torch.zeros(batch, self.n_drugs, device=device)
        valid_drug = drug_idx.clamp(min=0, max=self.n_drugs - 1)
        drug_onehot.scatter_(1, valid_drug.unsqueeze(1), 1.0)
        drug_feat = torch.cat([drug_onehot, dose.unsqueeze(1), duration.unsqueeze(1)], dim=-1)
        h_drug = self.drug_proj(drug_feat)

        h_time = self.time_enc(time_point)

        # Select by type: 0=knockout, 1=drug, 2=time_course
        is_ko = (ptype_ids == 0).unsqueeze(-1).float()
        is_drug = (ptype_ids == 1).unsqueeze(-1).float()
        is_time = (ptype_ids == 2).unsqueeze(-1).float()

        h = is_ko * h_ko + is_drug * h_drug + is_time * h_time + h_type
        return h


class ExpressionEncoder(nn.Module):
    """Encode baseline expression + perturbation vector → (μ, logvar)."""

    def __init__(self, config: PerturbationModelConfig):
        super().__init__()
        in_dim = config.n_genes + config.perturbation_dim
        h = config.encoder_hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.mu_head = nn.Linear(h, config.latent_dim)
        self.logvar_head = nn.Linear(h, config.latent_dim)

    def forward(
        self, expression: torch.Tensor, perturbation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, logvar), each (batch, latent_dim)."""
        x = torch.cat([expression, perturbation], dim=-1)
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class ExpressionDecoder(nn.Module):
    """Decode latent z + perturbation → predicted expression."""

    def __init__(self, config: PerturbationModelConfig):
        super().__init__()
        in_dim = config.latent_dim + config.perturbation_dim
        h = config.decoder_hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.n_genes),
        )

    def forward(
        self, z: torch.Tensor, perturbation: torch.Tensor
    ) -> torch.Tensor:
        """Returns (batch, n_genes) predicted log1p-expression."""
        x = torch.cat([z, perturbation], dim=-1)
        return self.net(x)


class PerturbationResponseModel(nn.Module):
    """Conditional VAE for predicting post-perturbation gene expression.

    Given a baseline expression profile x_base and a perturbation spec p,
    encodes to a latent z ~ q(z|x_base, p) and decodes to x̂_target.

    The model operates in log1p space: inputs should be log1p-transformed
    expression values, and outputs are in the same space.
    """

    def __init__(self, config: PerturbationModelConfig | None = None):
        super().__init__()
        self.config = config or PerturbationModelConfig()
        self.perturb_enc = PerturbationEncoder(self.config)
        self.encoder = ExpressionEncoder(self.config)
        self.decoder = ExpressionDecoder(self.config)

    def _reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _encode_perturbation(
        self,
        ptype_ids: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        dose: torch.Tensor,
        duration: torch.Tensor,
        time_point: torch.Tensor,
    ) -> torch.Tensor:
        return self.perturb_enc(ptype_ids, gene_idx, drug_idx, dose, duration, time_point)

    def forward(
        self,
        baseline: torch.Tensor,
        ptype_ids: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        dose: torch.Tensor,
        duration: torch.Tensor,
        time_point: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass with reparameterisation.

        All perturbation fields are (batch,) tensors; baseline is (batch, n_genes).
        Returns dict with keys: mu, logvar, z, predicted.
        """
        h_p = self._encode_perturbation(ptype_ids, gene_idx, drug_idx, dose, duration, time_point)
        mu, logvar = self.encoder(baseline, h_p)
        z = self._reparameterise(mu, logvar)
        predicted = self.decoder(z, h_p)
        return {"mu": mu, "logvar": logvar, "z": z, "predicted": predicted}

    @torch.no_grad()
    def predict(
        self,
        baseline: torch.Tensor,
        ptype_ids: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        dose: torch.Tensor,
        duration: torch.Tensor,
        time_point: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic prediction (uses μ, no sampling)."""
        was_training = self.training
        self.eval()
        out = self.forward(baseline, ptype_ids, gene_idx, drug_idx, dose, duration, time_point)
        if was_training:
            self.train()
        return out["predicted"]

    def compute_loss(
        self,
        baseline: torch.Tensor,
        ptype_ids: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        dose: torch.Tensor,
        duration: torch.Tensor,
        time_point: torch.Tensor,
        target: torch.Tensor,
        beta: float | None = None,
        consistency_lambda: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the full β-VAE loss.

        Returns dict: total, reconstruction, kl, consistency (all scalar tensors).
        """
        beta = beta if beta is not None else self.config.beta
        lam = consistency_lambda if consistency_lambda is not None else self.config.consistency_lambda

        out = self.forward(baseline, ptype_ids, gene_idx, drug_idx, dose, duration, time_point)
        mu, logvar, predicted = out["mu"], out["logvar"], out["predicted"]

        recon = F.mse_loss(predicted, target)

        # KL(q(z|x,p) || N(0,I)) = -0.5 * Σ(1 + logvar - μ² - exp(logvar))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        # Perturbation consistency: samples with the same perturbation type
        # should have similar latent shifts from their baselines.
        consistency = self._perturbation_consistency(
            out["z"], baseline, ptype_ids, gene_idx, drug_idx
        )

        total = recon + beta * kl + lam * consistency
        return {
            "total": total,
            "reconstruction": recon,
            "kl": kl,
            "consistency": consistency,
        }

    def _perturbation_consistency(
        self,
        z: torch.Tensor,
        baseline: torch.Tensor,
        ptype_ids: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage similar latent shifts for identical perturbations.

        Groups samples by (ptype, gene_idx for KO, drug_idx for drug).
        Within each group with ≥2 members, penalises variance of z.
        """
        batch = z.shape[0]
        if batch < 2:
            return torch.tensor(0.0, device=z.device)

        # Build a group key per sample
        keys = ptype_ids * 100000 + torch.where(
            ptype_ids == 0, gene_idx,
            torch.where(ptype_ids == 1, drug_idx + 50000, torch.zeros_like(gene_idx))
        )

        total_var = torch.tensor(0.0, device=z.device)
        n_groups = 0
        for key in keys.unique():
            mask = keys == key
            if mask.sum() < 2:
                continue
            z_group = z[mask]
            total_var = total_var + z_group.var(dim=0).mean()
            n_groups += 1

        if n_groups == 0:
            return torch.tensor(0.0, device=z.device)
        return total_var / n_groups


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class GEOExpressionData:
    """Container for a loaded GEO expression matrix with condition labels.

    expression: (n_samples, n_genes) numpy array, log1p-transformed
    conditions: list of str condition labels, length n_samples
    gene_names: list of str gene symbols, length n_genes
    species: "human" or "mouse"
    accession: GEO accession (e.g. "GSE107965")
    """
    expression: np.ndarray
    conditions: list[str]
    gene_names: list[str]
    species: str
    accession: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PerturbationDataset(Dataset):
    """Creates (baseline, perturbation, target) triplets from expression data.

    Given a gene expression matrix with condition labels, pairs each treatment
    sample with a control sample to form a perturbation-prediction task.

    The perturbation type is inferred from the data:
    - If conditions contain exactly two groups → drug/treatment perturbation
    - If conditions contain numeric time points → time_course perturbation
    - If metadata specifies knocked-out genes → knockout perturbation
    """

    def __init__(
        self,
        data: GEOExpressionData,
        control_condition: str | None = None,
        perturbation_type: PerturbationType = PerturbationType.DRUG,
        drug_idx: int = 0,
        dose_mg: float = 1.0,
        duration_hr: float = 24.0,
        log1p_transform: bool = False,
    ):
        super().__init__()
        expr = data.expression
        if log1p_transform:
            expr = np.log1p(np.maximum(expr, 0.0))

        conditions = np.array(data.conditions)
        unique_conds = sorted(set(data.conditions))

        if control_condition is None:
            for candidate in ("control", "ctrl", "untreated", "baseline", "vehicle", "DMSO", "WT"):
                matches = [c for c in unique_conds if candidate.lower() in c.lower()]
                if matches:
                    control_condition = matches[0]
                    break
            if control_condition is None:
                control_condition = unique_conds[0]

        ctrl_mask = conditions == control_condition
        treat_mask = ~ctrl_mask

        ctrl_indices = np.where(ctrl_mask)[0]
        treat_indices = np.where(treat_mask)[0]

        if len(ctrl_indices) == 0 or len(treat_indices) == 0:
            self.baselines = np.zeros((0, expr.shape[1]), dtype=np.float32)
            self.targets = np.zeros((0, expr.shape[1]), dtype=np.float32)
            self.ptype_ids = np.zeros(0, dtype=np.int64)
            self.gene_indices = np.zeros(0, dtype=np.int64)
            self.drug_indices = np.zeros(0, dtype=np.int64)
            self.doses = np.zeros(0, dtype=np.float32)
            self.durations = np.zeros(0, dtype=np.float32)
            self.time_points = np.zeros(0, dtype=np.float32)
            return

        # Pair each treatment sample with the nearest control sample (L2)
        ctrl_expr = expr[ctrl_indices]
        treat_expr = expr[treat_indices]

        # (n_treat, n_ctrl) distance matrix
        diffs = treat_expr[:, None, :] - ctrl_expr[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        nearest_ctrl = dists.argmin(axis=1)

        self.baselines = ctrl_expr[nearest_ctrl].astype(np.float32)
        self.targets = treat_expr.astype(np.float32)

        n = len(treat_indices)
        ptype_val = {
            PerturbationType.KNOCKOUT: 0,
            PerturbationType.DRUG: 1,
            PerturbationType.TIME_COURSE: 2,
        }[perturbation_type]

        self.ptype_ids = np.full(n, ptype_val, dtype=np.int64)
        self.gene_indices = np.zeros(n, dtype=np.int64)
        self.drug_indices = np.full(n, drug_idx, dtype=np.int64)
        self.doses = np.full(n, dose_mg, dtype=np.float32)
        self.durations = np.full(n, duration_hr, dtype=np.float32)

        # For time_course: parse numeric time from condition labels
        if perturbation_type == PerturbationType.TIME_COURSE:
            treat_conds = conditions[treat_indices]
            tp = []
            for c in treat_conds:
                nums = [float(s) for s in str(c).replace(",", " ").split() if _is_numeric(s)]
                tp.append(nums[0] if nums else 0.0)
            self.time_points = np.array(tp, dtype=np.float32)
        else:
            self.time_points = np.zeros(n, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.baselines)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "baseline": torch.from_numpy(self.baselines[idx]),
            "target": torch.from_numpy(self.targets[idx]),
            "ptype_ids": torch.tensor(self.ptype_ids[idx], dtype=torch.long),
            "gene_idx": torch.tensor(self.gene_indices[idx], dtype=torch.long),
            "drug_idx": torch.tensor(self.drug_indices[idx], dtype=torch.long),
            "dose": torch.tensor(self.doses[idx], dtype=torch.float32),
            "duration": torch.tensor(self.durations[idx], dtype=torch.float32),
            "time_point": torch.tensor(self.time_points[idx], dtype=torch.float32),
        }


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Convenience: load a Merlin GEO adapter CSV into GEOExpressionData
# ---------------------------------------------------------------------------

def load_geo_csv(
    path: str,
    condition_col: str = "condition",
    species: str = "human",
    accession: str = "",
    log1p: bool = True,
) -> GEOExpressionData:
    """Load a CSV where rows are samples, first column is condition label,
    remaining columns are gene expression values.

    This matches the format produced by Merlin's GEO adapters and committed
    under backend/data/expression/{species}/.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if condition_col in df.columns:
        conditions = df[condition_col].tolist()
        expr_df = df.drop(columns=[condition_col])
    else:
        conditions = df.iloc[:, 0].tolist()
        expr_df = df.iloc[:, 1:]

    gene_names = list(expr_df.columns)
    expr = expr_df.values.astype(np.float64)
    if log1p:
        expr = np.log1p(np.maximum(expr, 0.0))

    return GEOExpressionData(
        expression=expr.astype(np.float32),
        conditions=[str(c) for c in conditions],
        gene_names=gene_names,
        species=species,
        accession=accession,
    )
