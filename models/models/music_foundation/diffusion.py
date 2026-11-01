"""Diffusion-based generative model for music latents.

Implements a denoising diffusion probabilistic model (DDPM) in the latent
space of the spectrogram encoder. Supports classifier-free guidance via
the conditioning module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .encoder import LinearLayer, gelu, layer_norm, MultiHeadAttention
from .positional import sinusoidal_encoding


@dataclass
class DiffusionConfig:
    latent_dim: int = 128
    hidden_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    n_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    cond_dim: int = 128
    cfg_scale: float = 3.0
    schedule: str = "linear"


def _make_schedule(config: DiffusionConfig) -> dict[str, np.ndarray]:
    T = config.n_timesteps
    if config.schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1) / T
        alpha_bar = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0, 0.999)
    else:
        betas = np.linspace(config.beta_start, config.beta_end, T)

    alphas = 1.0 - betas
    alpha_cumprod = np.cumprod(alphas)
    alpha_cumprod_prev = np.concatenate([np.array([1.0]), alpha_cumprod[:-1]])

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
        "alpha_cumprod_prev": alpha_cumprod_prev,
        "sqrt_alpha_cumprod": np.sqrt(alpha_cumprod),
        "sqrt_one_minus_alpha_cumprod": np.sqrt(1.0 - alpha_cumprod),
        "posterior_variance": betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod + 1e-20),
    }


class TimestepEmbedding:
    def __init__(self, dim: int, seed: int = 0) -> None:
        self.proj1 = LinearLayer(dim, dim * 4, seed)
        self.proj2 = LinearLayer(dim * 4, dim, seed + 1)
        self.dim = dim

    def forward(self, t: int | np.ndarray) -> np.ndarray:
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        half = self.dim // 2
        freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
        args = t_arr[:, None] * freqs[None, :]
        emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        if emb.shape[-1] < self.dim:
            emb = np.pad(emb, ((0, 0), (0, self.dim - emb.shape[-1])))
        h = gelu(self.proj1.forward(emb))
        return self.proj2.forward(h)


class DenoisingBlock:
    def __init__(self, dim: int, n_heads: int, cond_dim: int, seed: int = 0) -> None:
        self.attn = MultiHeadAttention(dim, n_heads, seed)
        self.ff1 = LinearLayer(dim, dim * 4, seed + 100)
        self.ff2 = LinearLayer(dim * 4, dim, seed + 200)
        self.t_proj = LinearLayer(dim, dim, seed + 300)
        self.c_proj = LinearLayer(cond_dim, dim, seed + 400)

    def forward(self, x: np.ndarray, t_emb: np.ndarray, cond: np.ndarray | None = None) -> np.ndarray:
        t_mod = self.t_proj.forward(t_emb)
        if t_mod.ndim == 1:
            t_mod = t_mod.reshape(1, -1)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        h = x + t_mod
        if cond is not None:
            c_mod = self.c_proj.forward(cond.reshape(1, -1) if cond.ndim == 1 else cond)
            h = h + c_mod

        h = layer_norm(h + self.attn.forward(h))
        ff = self.ff2.forward(gelu(self.ff1.forward(h)))
        return layer_norm(h + ff)


class NoisePredictor:
    """U-Net-style noise predictor operating on latent sequences."""

    def __init__(self, config: DiffusionConfig, seed: int = 0) -> None:
        self.config = config
        self.time_emb = TimestepEmbedding(config.hidden_dim, seed)
        self.input_proj = LinearLayer(config.latent_dim, config.hidden_dim, seed + 10)
        self.blocks = [
            DenoisingBlock(config.hidden_dim, config.n_heads, config.cond_dim, seed + i * 500)
            for i in range(config.n_layers)
        ]
        self.output_proj = LinearLayer(config.hidden_dim, config.latent_dim, seed + 9999)

    def forward(
        self, x_t: np.ndarray, t: int | np.ndarray, cond: np.ndarray | None = None,
    ) -> np.ndarray:
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)
        t_emb = self.time_emb.forward(t)
        if t_emb.ndim == 2:
            t_emb = t_emb[0]

        h = gelu(self.input_proj.forward(x_t))
        for block in self.blocks:
            h = block.forward(h, t_emb, cond)
        return self.output_proj.forward(h)

    def params(self) -> dict[str, np.ndarray]:
        p = {}
        p.update({f"diff_input_{k}": v for k, v in self.input_proj.params.items()})
        for i, block in enumerate(self.blocks):
            p.update({f"diff_block{i}_ff1_{k}": v for k, v in block.ff1.params.items()})
            p.update({f"diff_block{i}_ff2_{k}": v for k, v in block.ff2.params.items()})
        p.update({f"diff_output_{k}": v for k, v in self.output_proj.params.items()})
        return p


class LatentDiffusionModel:
    """DDPM in the latent space with classifier-free guidance."""

    def __init__(self, config: DiffusionConfig | None = None, seed: int = 0) -> None:
        self.config = config or DiffusionConfig()
        self.schedule = _make_schedule(self.config)
        self.noise_pred = NoisePredictor(self.config, seed)
        self.rng = np.random.RandomState(seed)

    def q_sample(self, x_0: np.ndarray, t: int, noise: np.ndarray | None = None) -> np.ndarray:
        if noise is None:
            noise = self.rng.randn(*x_0.shape)
        sqrt_alpha = self.schedule["sqrt_alpha_cumprod"][t]
        sqrt_one_minus = self.schedule["sqrt_one_minus_alpha_cumprod"][t]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def training_loss(
        self, x_0: np.ndarray, cond: np.ndarray | None = None,
    ) -> float:
        t = self.rng.randint(0, self.config.n_timesteps)
        noise = self.rng.randn(*x_0.shape)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.noise_pred.forward(x_t, t, cond)

        if predicted_noise.shape != noise.shape:
            m = min(predicted_noise.shape[-1], noise.shape[-1])
            predicted_noise = predicted_noise[..., :m]
            noise = noise[..., :m]

        return float(np.mean((predicted_noise - noise) ** 2))

    def p_sample(
        self,
        x_t: np.ndarray,
        t: int,
        cond: np.ndarray | None = None,
    ) -> np.ndarray:
        pred_noise = self.noise_pred.forward(x_t, t, cond)

        if cond is not None and self.config.cfg_scale > 1.0:
            uncond_noise = self.noise_pred.forward(x_t, t, None)
            pred_noise = uncond_noise + self.config.cfg_scale * (pred_noise - uncond_noise)

        alpha = self.schedule["alphas"][t]
        alpha_bar = self.schedule["alpha_cumprod"][t]
        beta = self.schedule["betas"][t]

        mean = (1.0 / np.sqrt(alpha)) * (
            x_t - (beta / np.sqrt(1.0 - alpha_bar + 1e-20)) * pred_noise
        )

        if t > 0:
            var = self.schedule["posterior_variance"][t]
            noise = self.rng.randn(*x_t.shape)
            return mean + np.sqrt(var) * noise
        return mean

    def sample(
        self,
        shape: tuple[int, ...],
        cond: np.ndarray | None = None,
        n_steps: int | None = None,
    ) -> np.ndarray:
        n_steps = n_steps or self.config.n_timesteps
        step_size = max(1, self.config.n_timesteps // n_steps)
        timesteps = list(range(self.config.n_timesteps - 1, -1, -step_size))

        x_t = self.rng.randn(*shape)
        for t in timesteps:
            x_t = self.p_sample(x_t, t, cond)
        return x_t

    def sample_progressive(
        self,
        shape: tuple[int, ...],
        cond: np.ndarray | None = None,
        save_every: int = 100,
    ) -> list[np.ndarray]:
        intermediates = []
        x_t = self.rng.randn(*shape)
        for t in range(self.config.n_timesteps - 1, -1, -1):
            x_t = self.p_sample(x_t, t, cond)
            if (self.config.n_timesteps - 1 - t) % save_every == 0:
                intermediates.append(x_t.copy())
        intermediates.append(x_t.copy())
        return intermediates

    def params(self) -> dict[str, np.ndarray]:
        return self.noise_pred.params()
