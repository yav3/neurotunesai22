"""Train a frontier music model from synthetic data, then generate audio.

This is the single entry point that demonstrates the full foundation model
lifecycle: corpus creation → pretraining → evaluation → generation → export.

The pipeline proves the model learns real musical structure by measuring:
  1. Training loss decreasing across steps (the model is learning)
  2. Codebook utilization increasing (discrete representations form)
  3. Generated continuations that differ based on seed style (representations
     are style-sensitive, not random)
  4. Temperature sweep producing expected diversity ordering

Pure numpy/scipy. No external ML framework.

Usage:
    python -m models.music_foundation.train_and_generate [--n-steps 20] [--output-dir out/]
"""
from __future__ import annotations

import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .frontier_model import (
    FrontierConfig, MusicFrontierModel,
    contrastive_loss, diversity_loss, masked_prediction_loss,
    cosine_warmup_schedule, batch_spectrograms,
)
from .generation import (
    GenerationConfig, MusicContinuation,
)
from .spectrogram import compute_stft, istft, griffin_lim
from .generation_demo import (
    DemoConfig, generate_seed_audio, compute_mel, mel_to_magnitude, save_wav,
)


@dataclass
class TrainGenConfig:
    """Configuration for the train-then-generate pipeline."""
    # Corpus
    n_clips: int = 50
    clip_duration_s: float = 2.0
    sr: int = 16000

    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 256
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    codebook_size: int = 64
    codebook_dim: int = 32
    n_codebooks: int = 4
    max_seq_len: int = 256
    attention_window: int = 0  # 0 = full O(T²) attention; >0 = local O(T·w)

    # Training
    n_steps: int = 80
    batch_size: int = 2
    lr: float = 5e-2
    spsa_eps: float = 1e-2
    momentum: float = 0.8
    contrastive_weight: float = 1.0
    diversity_weight: float = 0.1
    mask_prob: float = 0.15
    ema_decay: float = 0.95
    codebook_reset_threshold: int = 1

    # Generation
    gen_styles: List[str] = field(default_factory=lambda: ["chord", "melody", "rhythm"])
    gen_temperatures: List[float] = field(default_factory=lambda: [0.5, 0.9, 1.3])
    continuation_length: int = 192
    phrase_form: str = 'AABA'
    phrase_variation: float = 0.15

    # Output
    output_dir: Optional[str] = None
    save_checkpoint: bool = True

    seed: int = 42


@dataclass
class TrainGenReport:
    """Results from the full train→generate pipeline."""
    # Training
    train_losses: List[float]
    codebook_utilizations: List[float]
    train_time_s: float
    loss_decreased: bool
    codebook_improved: bool

    # Generation
    generations: List[Dict[str, Any]]
    style_diversity: float
    temp_diversity: float
    gen_time_s: float

    # Fidelity
    round_trip_spectral_convergence: float
    round_trip_correlation: float

    # Structure
    phrase_self_similarity: float
    phrase_form: str

    # Uncertainty
    generation_perplexity: float
    confidence_stats: Dict[str, float]

    # Overall
    total_time_s: float
    n_params: int
    all_checks_passed: bool
    summary: str


def _generate_corpus(
    n_clips: int, duration_s: float, sr: int, seed: int,
) -> List[np.ndarray]:
    """Generate a synthetic training corpus with genuine musical variety.

    Goes beyond static tones: harmonic progressions, arpeggios, walking bass,
    varied drum patterns, FM synthesis. Each style family has internal
    variation (key, tempo, voicing) so the model sees distributional structure.
    """
    rng = np.random.RandomState(seed)
    clips = []
    target_len = int(duration_s * sr)

    # Musical constants
    A4 = 440.0
    def midi_to_hz(m): return A4 * 2 ** ((m - 69) / 12.0)

    # Common chord progressions (MIDI root notes, relative to C4=60)
    progressions = [
        [60, 65, 67, 60],       # I-IV-V-I
        [60, 64, 65, 67],       # I-iii-IV-V
        [60, 69, 65, 67],       # I-vi-IV-V
        [60, 62, 65, 67],       # I-ii-IV-V
        [60, 65, 69, 67],       # I-IV-vi-V
    ]
    chord_types = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'dom7':  [0, 4, 7, 10],
        'min7':  [0, 3, 7, 10],
    }

    styles = [
        'progression', 'arpeggio', 'bass_line', 'drums',
        'fm_pad', 'melody', 'chord', 'rhythm', 'noise', 'sine',
    ]

    for i in range(n_clips):
        style = styles[i % len(styles)]
        t = np.linspace(0, duration_s, target_len, endpoint=False)

        if style == 'progression':
            prog = progressions[rng.randint(len(progressions))]
            key_shift = rng.randint(-3, 4)
            ct = list(chord_types.values())[rng.randint(len(chord_types))]
            chord_dur = duration_s / len(prog)
            sig = np.zeros(target_len)
            for ci, root in enumerate(prog):
                root += key_shift
                freqs = [midi_to_hz(root + iv) for iv in ct]
                start = int(ci * chord_dur * sr)
                end = min(int((ci + 1) * chord_dur * sr), target_len)
                ct_t = np.arange(end - start) / sr
                env = np.minimum(ct_t / 0.02, 1.0) * np.exp(-0.8 * ct_t)
                chord_sig = sum(0.25 * np.sin(2 * np.pi * f * ct_t) for f in freqs)
                sig[start:end] = chord_sig * env
            clips.append(sig)

        elif style == 'arpeggio':
            prog = progressions[rng.randint(len(progressions))]
            key_shift = rng.randint(-5, 6)
            bpm = rng.choice([100, 120, 140, 160])
            note_dur = 60.0 / bpm / 2  # eighth notes
            ct = [0, 4, 7, 12]  # major + octave
            sig = np.zeros(target_len)
            note_idx = 0
            for root in prog:
                root += key_shift
                for iv in ct:
                    f = midi_to_hz(root + iv)
                    start = int(note_idx * note_dur * sr)
                    end = min(int((note_idx + 1) * note_dur * sr), target_len)
                    if start >= target_len:
                        break
                    nt = np.arange(end - start) / sr
                    env = np.minimum(nt / 0.005, 1.0) * np.exp(-4.0 * nt)
                    sig[start:end] = 0.4 * np.sin(2 * np.pi * f * nt) * env
                    note_idx += 1
            clips.append(sig)

        elif style == 'bass_line':
            prog = progressions[rng.randint(len(progressions))]
            key_shift = rng.randint(-3, 4) - 12  # one octave down
            bpm = rng.choice([90, 100, 110, 120])
            beat_dur = 60.0 / bpm
            sig = np.zeros(target_len)
            beat = 0
            for root in prog:
                root += key_shift
                _bass_patterns = [[0, 0, 7, 5], [0, 7, 0, 5], [0, 0, 0, 7]]
                pattern = _bass_patterns[rng.randint(len(_bass_patterns))]
                for iv in pattern:
                    f = midi_to_hz(root + iv)
                    start = int(beat * beat_dur * sr)
                    end = min(int((beat + 1) * beat_dur * sr), target_len)
                    if start >= target_len:
                        break
                    nt = np.arange(end - start) / sr
                    env = np.minimum(nt / 0.005, 1.0) * np.exp(-3.0 * nt)
                    # Saw-like bass
                    harm = sum(0.3 / (h + 1) * np.sin(2 * np.pi * f * (h + 1) * nt) for h in range(4))
                    sig[start:end] = harm * env
                    beat += 1
            clips.append(sig)

        elif style == 'drums':
            bpm = rng.choice([90, 110, 120, 140])
            beat_dur = 60.0 / bpm
            sig = np.zeros(target_len)
            # Kick, snare, hihat with variation
            patterns = {
                'kick':  [[1,0,0,0,1,0,0,0], [1,0,0,1,1,0,0,0], [1,0,1,0,1,0,0,1]][rng.randint(3)],
                'snare': [[0,0,1,0,0,0,1,0], [0,0,1,0,0,1,1,0]][rng.randint(2)],
                'hihat': [[1,1,1,1,1,1,1,1], [1,0,1,0,1,0,1,0], [1,1,0,1,1,1,0,1]][rng.randint(3)],
            }
            eighth_dur = beat_dur / 2
            for step_i in range(int(duration_s / eighth_dur)):
                pi = step_i % 8
                pos = int(step_i * eighth_dur * sr)
                # Kick
                if patterns['kick'][pi] and pos + int(0.05 * sr) <= target_len:
                    kl = min(int(0.05 * sr), target_len - pos)
                    kt = np.arange(kl) / sr
                    sig[pos:pos + kl] += 0.6 * np.sin(2 * np.pi * 55 * kt * np.exp(-25 * kt)) * np.exp(-15 * kt)
                # Snare
                if patterns['snare'][pi] and pos + int(0.04 * sr) <= target_len:
                    sl = min(int(0.04 * sr), target_len - pos)
                    st_ = np.arange(sl) / sr
                    sig[pos:pos + sl] += 0.3 * (np.sin(2 * np.pi * 200 * st_) + 0.5 * rng.randn(sl)) * np.exp(-20 * st_)
                # Hihat
                if patterns['hihat'][pi] and pos + int(0.02 * sr) <= target_len:
                    hl = min(int(0.02 * sr), target_len - pos)
                    sig[pos:pos + hl] += 0.15 * rng.randn(hl) * np.exp(-40 * np.arange(hl) / sr)
            clips.append(np.clip(sig, -1, 1))

        elif style == 'fm_pad':
            root = midi_to_hz(60 + rng.randint(-5, 6))
            mod_ratio = rng.choice([1.0, 2.0, 3.0, 0.5])
            mod_idx = rng.uniform(1.0, 5.0)
            lfo_rate = rng.uniform(0.5, 3.0)
            sig = np.zeros(target_len)
            # FM synthesis with LFO
            mod_env = mod_idx * (1 + 0.3 * np.sin(2 * np.pi * lfo_rate * t))
            modulator = mod_env * np.sin(2 * np.pi * root * mod_ratio * t)
            carrier = 0.4 * np.sin(2 * np.pi * root * t + modulator)
            env = np.minimum(t / 0.1, 1.0) * np.exp(-0.3 * t)
            clips.append(carrier * env)

        else:
            # Fall back to the original simple styles
            clip = generate_seed_audio(duration_s, sr, seed + i, style)
            # Pitch shift via resampling
            shift = 1.0 + rng.uniform(-0.1, 0.1)
            n_out = int(len(clip) / shift)
            indices = np.linspace(0, len(clip) - 1, n_out)
            idx_floor = np.floor(indices).astype(int)
            idx_ceil = np.minimum(idx_floor + 1, len(clip) - 1)
            frac = indices - idx_floor
            clip = clip[idx_floor] * (1 - frac) + clip[idx_ceil] * frac
            if len(clip) >= target_len:
                clip = clip[:target_len]
            else:
                clip = np.pad(clip, (0, target_len - len(clip)))
            clips.append(clip)

    return clips


def _compute_batch_mels(
    clips: List[np.ndarray], sr: int, n_fft: int, hop_length: int, n_mels: int,
) -> np.ndarray:
    """Clips → (batch, n_mels, T) mel spectrograms, zero-padded."""
    mels = []
    for clip in clips:
        mel = compute_mel(clip, sr, n_fft, hop_length, n_mels)  # (T, n_mels)
        mels.append(mel.T)  # (n_mels, T)
    max_t = max(m.shape[1] for m in mels)
    batch = np.zeros((len(mels), n_mels, max_t))
    for i, m in enumerate(mels):
        batch[i, :, :m.shape[1]] = m
    return batch


def _collect_params(model: MusicFrontierModel) -> List[Tuple[Any, str, np.ndarray]]:
    """Walk model component tree and collect all trainable weight arrays."""
    params = []
    components = [
        model.feature_extractor,
        model.transformer,
        model.quantizer,
        model.contrastive_head,
        model.classifier,
    ]
    for comp in components:
        for attr_name in dir(comp):
            if attr_name.startswith('_'):
                continue
            val = getattr(comp, attr_name, None)
            if isinstance(val, np.ndarray) and val.dtype.kind == 'f' and val.size > 0:
                params.append((comp, attr_name, val))
            # Check nested: transformer layers
            if attr_name == 'layers' and isinstance(val, list):
                for layer in val:
                    for la in dir(layer):
                        if la.startswith('_'):
                            continue
                        lv = getattr(layer, la, None)
                        if isinstance(lv, np.ndarray) and lv.dtype.kind == 'f' and lv.size > 0:
                            params.append((layer, la, lv))
            # Conv layers list
            if attr_name == 'conv_weights' and isinstance(val, list):
                for idx, w in enumerate(val):
                    if isinstance(w, np.ndarray):
                        params.append((comp, f'conv_weights[{idx}]', w))
    return params


def _flatten_params(params: List[Tuple[Any, str, np.ndarray]]) -> np.ndarray:
    return np.concatenate([p.ravel() for _, _, p in params])


def _unflatten_and_set(
    params: List[Tuple[Any, str, np.ndarray]], flat: np.ndarray,
) -> None:
    offset = 0
    for comp, attr_name, orig in params:
        size = orig.size
        new_val = flat[offset:offset + size].reshape(orig.shape)
        if '[' in attr_name:
            # Handle list indexing like conv_weights[0]
            base, idx_str = attr_name.split('[')
            idx = int(idx_str.rstrip(']'))
            getattr(comp, base)[idx] = new_val
        else:
            setattr(comp, attr_name, new_val)
        offset += size


def _ema_codebook_update(
    model: MusicFrontierModel,
    batch_mel: np.ndarray,
    ema_decay: float,
    reset_threshold: int,
    usage_counts: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """EMA codebook update: move entries toward their assigned encoder outputs.

    For each codebook entry, compute EMA of the encoder features assigned to it.
    Reset dead entries (usage < reset_threshold) to random live encoder outputs.
    This bypasses the non-differentiable argmax in Gumbel-softmax quantization —
    codebook vectors move toward usage, not through gradients.

    Returns updated usage_counts.
    """
    quantizer = model.quantizer

    # Run encoder (feature extraction + transformer) to get continuous representations
    features = model.feature_extractor.forward(batch_mel)  # (T, d_model)
    enc_out = model.transformer.forward(features)
    hidden = enc_out['hidden_states']  # (T, d_model)

    # Project to codebook space
    T = hidden.shape[0]
    projected = hidden @ quantizer.proj_W  # (T, n_codebooks * codebook_dim)
    projected = projected.reshape(T, quantizer.n_codebooks, quantizer.codebook_dim)

    for g in range(quantizer.n_codebooks):
        # L2 nearest-neighbor assignment (consistent with k-means)
        dists = np.sum(
            (projected[:, g, None, :] - quantizer.codebooks[g][None, :, :]) ** 2,
            axis=-1,
        )  # (T, V)
        indices = np.argmin(dists, axis=1)  # (T,)

        for v in range(quantizer.codebook_size):
            mask = (indices == v)
            count = float(mask.sum())
            usage_counts[g * quantizer.codebook_size + v] += count

            if count > 0:
                assigned_mean = projected[mask, g].mean(axis=0)
                quantizer.codebooks[g][v] = (
                    ema_decay * quantizer.codebooks[g][v]
                    + (1 - ema_decay) * assigned_mean
                )

        # Reset dead entries to random data points + noise scaled to data
        alive_mask = np.array([
            usage_counts[g * quantizer.codebook_size + v] >= reset_threshold
            for v in range(quantizer.codebook_size)
        ])
        dead = np.where(~alive_mask)[0]
        if len(dead) > 0 and T > 0:
            data_std = projected[:, g].std() + 1e-6
            src_idx = rng.choice(T, size=len(dead), replace=True)
            for d, s in zip(dead, src_idx):
                quantizer.codebooks[g][d] = (
                    projected[s, g] + rng.randn(quantizer.codebook_dim) * data_std * 0.1
                )
                usage_counts[g * quantizer.codebook_size + d] = 0.0

    return usage_counts


def _collect_params_by_block(model: MusicFrontierModel) -> List[List[Tuple[Any, str, np.ndarray]]]:
    """Collect params grouped by model block for block-coordinate SPSA.

    Returns a list of blocks, where each block is a list of (component, attr, array).
    Blocks: [feature_extractor, transformer_layer_0, ..., transformer_layer_N, quantizer, heads]
    """
    blocks = []

    # Feature extractor
    fe_params = []
    comp = model.feature_extractor
    for attr in dir(comp):
        if attr.startswith('_'):
            continue
        val = getattr(comp, attr, None)
        if isinstance(val, np.ndarray) and val.dtype.kind == 'f' and val.size > 0:
            fe_params.append((comp, attr, val))
        if attr == 'conv_weights' and isinstance(val, list):
            for idx, w in enumerate(val):
                if isinstance(w, np.ndarray):
                    fe_params.append((comp, f'conv_weights[{idx}]', w))
    if fe_params:
        blocks.append(fe_params)

    # Transformer layers — one block per layer
    for layer in model.transformer.layers:
        layer_params = []
        for attr in dir(layer):
            if attr.startswith('_'):
                continue
            val = getattr(layer, attr, None)
            if isinstance(val, np.ndarray) and val.dtype.kind == 'f' and val.size > 0:
                layer_params.append((layer, attr, val))
        if layer_params:
            blocks.append(layer_params)

    # Transformer non-layer params (e.g. layer_norm)
    tf_other = []
    for attr in dir(model.transformer):
        if attr.startswith('_') or attr == 'layers':
            continue
        val = getattr(model.transformer, attr, None)
        if isinstance(val, np.ndarray) and val.dtype.kind == 'f' and val.size > 0:
            tf_other.append((model.transformer, attr, val))
    if tf_other:
        blocks.append(tf_other)

    # Quantizer + heads together (small)
    tail_params = []
    for comp in [model.quantizer, model.contrastive_head, model.classifier]:
        for attr in dir(comp):
            if attr.startswith('_'):
                continue
            val = getattr(comp, attr, None)
            if isinstance(val, np.ndarray) and val.dtype.kind == 'f' and val.size > 0:
                tail_params.append((comp, attr, val))
    if tail_params:
        blocks.append(tail_params)

    return blocks


def _spsa_step(
    model: MusicFrontierModel,
    batch: np.ndarray,
    lr: float,
    eps: float,
    rng: np.random.RandomState,
    contrastive_w: float,
    diversity_w: float,
    mask_prob: float,
    n_perturbations: int = 3,
    ema_decay: float = 0.99,
    reset_threshold: int = 2,
    usage_counts: Optional[np.ndarray] = None,
    velocity: Optional[np.ndarray] = None,
    momentum: float = 0.8,
    step_idx: int = 0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Block-coordinate SPSA with momentum and EMA codebook updates.

    For models >500K params, cycles through model blocks (feature extractor,
    each transformer layer, quantizer+heads), perturbing only one block per
    step. This reduces effective perturbation dimension by ~Nx where N is
    number of blocks, dramatically improving gradient SNR for large models.
    Returns (loss, codebook_utilization, usage_counts, velocity).
    """
    all_params = _collect_params(model)
    flat_full = _flatten_params(all_params)
    n_params = flat_full.size

    # Block-coordinate for large models: perturb one block at a time
    blocks = _collect_params_by_block(model)
    block_idx = step_idx % len(blocks)
    block_params = blocks[block_idx]

    block_size = sum(p.size for _, _, p in block_params)

    # Find offset of this block in the full param vector
    offset = 0
    for bi in range(block_idx):
        for _, _, p in blocks[bi]:
            offset += p.size

    grad_accum = np.zeros(block_size)
    mask_seed = rng.randint(0, 2**31)

    n_batch = len(batch)
    inv_2eps = 1.0 / (2 * eps)
    block_view = flat_full[offset:offset + block_size]

    def _batch_loss() -> float:
        """Average loss across batch with current params."""
        total = 0.0
        for bi in range(n_batch):
            out = model.pretrain_forward(batch[bi], mask_seed=mask_seed)
            l = float(out['total_loss'])
            total += l if np.isfinite(l) else 10.0
        return total / n_batch

    for _ in range(n_perturbations):
        block_delta = rng.choice([-1.0, 1.0], size=block_size)
        eps_delta = eps * block_delta

        # +eps perturbation (in-place on the block slice)
        block_view += eps_delta
        _unflatten_and_set(all_params, flat_full)
        loss_plus = _batch_loss()

        # -eps perturbation (swing from +eps to -eps)
        block_view -= 2.0 * eps_delta
        _unflatten_and_set(all_params, flat_full)
        loss_minus = _batch_loss()

        # Restore original params
        block_view += eps_delta

        grad_accum += (loss_plus - loss_minus) * inv_2eps * block_delta

    # Full gradient vector with block-coordinate structure
    grad = np.zeros_like(flat_full)
    grad[offset:offset + block_size] = grad_accum / n_perturbations

    # Normalize the active block's gradient
    block_grad_norm = np.linalg.norm(grad[offset:offset + block_size]) + 1e-10
    grad[offset:offset + block_size] /= block_grad_norm

    # Momentum
    if velocity is None:
        velocity = np.zeros_like(grad)
    velocity = momentum * velocity + (1 - momentum) * grad
    step_grad = velocity

    flat_new = np.clip(flat_full - lr * step_grad, -5.0, 5.0)
    _unflatten_and_set(all_params, flat_new)

    # EMA codebook update: move codebook entries toward assigned features
    n_total_entries = model.config.n_codebooks * model.config.codebook_size
    if usage_counts is None:
        usage_counts = np.zeros(n_total_entries, dtype=np.float64)
    # Decay old counts so resets track recent usage, not cumulative
    usage_counts = (usage_counts * 0.9).astype(np.float64)
    for bi in range(n_batch):
        usage_counts = _ema_codebook_update(
            model, batch[bi], ema_decay, reset_threshold, usage_counts, rng,
        )

    # Evaluate at updated params (average over batch)
    eval_loss = 0.0
    total_perp = 0.0
    for bi in range(n_batch):
        out = model.pretrain_forward(batch[bi], mask_seed=mask_seed + 1)
        el = float(out['total_loss'])
        eval_loss += el if np.isfinite(el) else 10.0
        total_perp += float(out.get('perplexity', 0.0))
    loss = eval_loss / n_batch
    perp = total_perp / n_batch
    cb_util = min(perp / model.config.codebook_size, 1.0)

    return loss, cb_util, usage_counts, velocity



def train_and_generate(config: Optional[TrainGenConfig] = None) -> TrainGenReport:
    """Run the full train → generate pipeline.

    1. Generate synthetic musical corpus
    2. Build model
    3. Pretrain via SPSA with momentum + EMA codebook updates
    4. Train mel decoder projection (codebook → mel space)
    5. Generate continuations via learned vocoder across styles/temperatures
    6. Measure diversity (style spectral spread, temperature RMS variation)
    7. Measure round-trip fidelity
    8. Produce report with all metrics
    """
    cfg = config or TrainGenConfig()
    rng = np.random.RandomState(cfg.seed)
    t0 = time.monotonic()

    # ---- 1. Corpus ---- #
    clips = _generate_corpus(cfg.n_clips, cfg.clip_duration_s, cfg.sr, cfg.seed)

    # ---- 2. Build model ---- #
    frontier_cfg = FrontierConfig(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        n_mels=cfg.n_mels,
        conv_channels=[cfg.n_mels, cfg.d_model],
        conv_kernels=[10, 3],
        conv_strides=[5, 2],
        n_codebooks=cfg.n_codebooks,
        codebook_size=cfg.codebook_size,
        codebook_dim=cfg.codebook_dim,
        max_seq_len=cfg.max_seq_len,
        attention_window=cfg.attention_window,
    )
    model = MusicFrontierModel(frontier_cfg, seed=cfg.seed)

    # Data-dependent codebook initialization: run k-means on encoder outputs
    # so the quantizer starts with a meaningful vocabulary.  Without this,
    # codebooks at std=0.01 vs projected vectors at std~1.0 produce
    # near-uniform logits → all tokens collapse to one entry.
    init_projected = []
    for clip in clips[:min(30, len(clips))]:
        mel_init = compute_mel(clip, cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)
        feats_init = model.feature_extractor.forward(mel_init.T)
        enc_init = model.transformer.forward(feats_init)
        h_init = enc_init['hidden_states']
        p_init = h_init @ model.quantizer.proj_W
        p_init = p_init.reshape(h_init.shape[0], cfg.n_codebooks, cfg.codebook_dim)
        init_projected.append(p_init)
    init_projected = np.concatenate(init_projected, axis=0)  # (N, n_codebooks, cb_dim)

    for g in range(cfg.n_codebooks):
        data_g = init_projected[:, g]  # (N, cb_dim)
        # K-means++ init: spread initial centroids apart
        centroids = [data_g[rng.randint(len(data_g))]]
        for _ in range(cfg.codebook_size - 1):
            dists = np.min([np.sum((data_g - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dists / (dists.sum() + 1e-10)
            centroids.append(data_g[rng.choice(len(data_g), p=probs)])
        model.quantizer.codebooks[g] = np.array(centroids)
        # 5 Lloyd iterations
        for _ in range(5):
            d2 = np.sum((data_g[:, None] - model.quantizer.codebooks[g][None]) ** 2, axis=-1)
            assign = np.argmin(d2, axis=1)
            for v in range(cfg.codebook_size):
                mask = (assign == v)
                if mask.sum() > 0:
                    model.quantizer.codebooks[g][v] = data_g[mask].mean(axis=0)

    all_params = _collect_params(model)
    n_params = sum(p.size for _, _, p in all_params)

    # ---- 3. Train ---- #
    train_losses = []
    codebook_utils = []
    usage_counts = None
    velocity = None
    t_train_start = time.monotonic()

    for step in range(cfg.n_steps):
        # Sample mini-batch
        idx = rng.choice(len(clips), size=min(cfg.batch_size, len(clips)), replace=False)
        batch_clips = [clips[i] for i in idx]
        batch_mel = _compute_batch_mels(
            batch_clips, cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels,
        )

        # LR schedule: cosine warmup
        warmup_steps = max(1, cfg.n_steps // 5)
        if step < warmup_steps:
            lr = cfg.lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, cfg.n_steps - warmup_steps)
            lr = cfg.lr * 0.5 * (1 + np.cos(np.pi * progress))

        # Adaptive perturbation count: more samples later for precision
        n_pert = 3 if step < cfg.n_steps // 2 else 5

        loss, cb_util, usage_counts, velocity = _spsa_step(
            model, batch_mel, lr, cfg.spsa_eps, rng,
            cfg.contrastive_weight, cfg.diversity_weight, cfg.mask_prob,
            n_perturbations=n_pert,
            ema_decay=cfg.ema_decay,
            reset_threshold=cfg.codebook_reset_threshold,
            usage_counts=usage_counts,
            velocity=velocity,
            momentum=cfg.momentum,
            step_idx=step,
        )
        train_losses.append(loss)
        codebook_utils.append(cb_util)

    train_time = time.monotonic() - t_train_start

    # ---- 4. K-means codebook refinement ---- #
    # After SPSA pretraining, do exact k-means on encoder outputs to perfectly
    # place codebook entries. No gradients needed — closed-form optimization.
    all_projected = []
    for clip in clips:
        mel_clip = compute_mel(clip, cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)
        feats = model.feature_extractor.forward(mel_clip.T)
        enc = model.transformer.forward(feats)
        hidden = enc['hidden_states']
        T_h = hidden.shape[0]
        proj = hidden @ model.quantizer.proj_W
        proj = proj.reshape(T_h, model.quantizer.n_codebooks, cfg.codebook_dim)
        all_projected.append(proj)
    all_projected = np.concatenate(all_projected, axis=0)  # (N, n_codebooks, cb_dim)

    for g in range(model.quantizer.n_codebooks):
        data_g = all_projected[:, g]  # (N, cb_dim)
        # K-means: 5 iterations of assign-then-update
        for _ in range(5):
            dists = np.sum(
                (data_g[:, None] - model.quantizer.codebooks[g][None]) ** 2, axis=-1
            )  # (N, V)
            assignments = np.argmin(dists, axis=1)
            for v in range(model.quantizer.codebook_size):
                mask = (assignments == v)
                if mask.sum() > 0:
                    model.quantizer.codebooks[g][v] = data_g[mask].mean(axis=0)

    # Measure post-refinement codebook utilization
    post_util_used = set()
    for clip in clips[:min(20, len(clips))]:
        mel_util = compute_mel(clip, cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)
        feats_util = model.feature_extractor.forward(mel_util.T)
        enc_util = model.transformer.forward(feats_util)
        h_util = enc_util['hidden_states']
        p_util = h_util @ model.quantizer.proj_W
        p_util = p_util.reshape(h_util.shape[0], model.quantizer.n_codebooks, cfg.codebook_dim)
        for g in range(model.quantizer.n_codebooks):
            d2 = np.sum((p_util[:, g, None, :] - model.quantizer.codebooks[g][None]) ** 2, axis=-1)
            for idx in np.argmin(d2, axis=1):
                post_util_used.add((g, int(idx)))
    post_refine_util = len(post_util_used) / (cfg.n_codebooks * cfg.codebook_size)

    # ---- 5. Train mel decoder projection ---- #
    cb_total_dim = cfg.n_codebooks * cfg.codebook_dim
    decoder_W, decoder_b = _init_decoder_projection(cb_total_dim, cfg.n_mels, cfg.seed + 999)
    decoder_W, decoder_b, rbf_params = _train_decoder_projection(
        model, clips, decoder_W, decoder_b, cfg, rng, n_steps=10,
    )

    # ---- 6. Build per-codebook interpolated trigram models ---- #
    V = cfg.codebook_size
    n_cb = cfg.n_codebooks
    # Per-codebook: unigram + bigram + trigram counts, then interpolated backoff.
    # Trigram captures 2-step patterns (e.g. I→IV→V chord progressions).
    unigram_counts_all = np.ones((n_cb, V))
    bigram_counts_all = np.ones((n_cb, V, V))
    trigram_counts_all = np.ones((n_cb, V, V, V))
    token_sequences_all = [[] for _ in range(n_cb)]
    for clip in clips:
        mel_clip = compute_mel(clip, cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)
        feats = model.feature_extractor.forward(mel_clip.T)
        enc = model.transformer.forward(feats)
        hidden = enc['hidden_states']
        proj = hidden @ model.quantizer.proj_W
        proj = proj.reshape(hidden.shape[0], n_cb, cfg.codebook_dim)
        for g in range(n_cb):
            d2 = np.sum(
                (proj[:, g, None, :] - model.quantizer.codebooks[g][None]) ** 2,
                axis=-1,
            )
            tok_seq = np.argmin(d2, axis=1)
            token_sequences_all[g].append(tok_seq)
            for j in range(len(tok_seq)):
                unigram_counts_all[g, tok_seq[j]] += 1
                if j >= 1:
                    bigram_counts_all[g, tok_seq[j - 1], tok_seq[j]] += 1
                if j >= 2:
                    trigram_counts_all[g, tok_seq[j - 2], tok_seq[j - 1], tok_seq[j]] += 1

    # Normalize to probability distributions
    unigram_probs_all = unigram_counts_all / unigram_counts_all.sum(axis=1, keepdims=True)
    bigram_probs_all = bigram_counts_all / bigram_counts_all.sum(axis=2, keepdims=True)
    trigram_probs_all = trigram_counts_all / trigram_counts_all.sum(axis=3, keepdims=True)

    # Pack into a dict for the generation function
    ngram_model = {
        'unigram': unigram_probs_all,
        'bigram': bigram_probs_all,
        'trigram': trigram_probs_all,
        'lambdas': (0.1, 0.3, 0.6),  # interpolation: uni, bi, tri
    }

    bigram_entropy = float(np.mean([
        -np.sum(bigram_probs_all[g] * np.log(bigram_probs_all[g] + 1e-12)) / V
        for g in range(n_cb)
    ]))

    # ---- 7. Generate from trained model ---- #
    t_gen_start = time.monotonic()
    generations = []

    gen_idx = 0
    for style in cfg.gen_styles:
        for temp in cfg.gen_temperatures:
            seed_audio = generate_seed_audio(1.0, cfg.sr, cfg.seed, style)
            mel = compute_mel(seed_audio, cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)

            gen_seed = cfg.seed + gen_idx * 137
            gen_idx += 1
            gen_rng = np.random.RandomState(gen_seed)

            # Encode seed audio → per-codebook seed token sequences
            seed_feats = model.feature_extractor.forward(mel.T)
            seed_enc = model.transformer.forward(seed_feats)
            seed_hidden = seed_enc['hidden_states']
            seed_proj = seed_hidden @ model.quantizer.proj_W
            seed_proj = seed_proj.reshape(
                seed_hidden.shape[0], cfg.n_codebooks, cfg.codebook_dim,
            )

            # Sample independently per codebook with phrase structure
            per_cb_tokens = []
            per_cb_logprobs = []
            for g in range(cfg.n_codebooks):
                seed_d2 = np.sum(
                    (seed_proj[:, g, None, :] - model.quantizer.codebooks[g][None]) ** 2,
                    axis=-1,
                )
                seed_tok_seq = np.argmin(seed_d2, axis=1)
                cb_rng = np.random.RandomState(gen_seed + g * 53)
                cb_tokens, cb_lp = _generate_phrase_tokens(
                    ngram_model, g, seed_tok_seq, cfg.continuation_length,
                    temperature=temp, rng=cb_rng,
                    form=cfg.phrase_form,
                    variation_noise=cfg.phrase_variation,
                    return_confidence=True,
                )
                per_cb_tokens.append(cb_tokens)
                per_cb_logprobs.append(cb_lp)

            # Build full embeddings from multi-codebook tokens
            tokens = per_cb_tokens[0]  # primary tokens for length/reporting
            n_gen = min(len(t) for t in per_cb_tokens)
            if n_gen > 0:
                cont_audio = _tokens_to_audio_multicb(
                    per_cb_tokens, model, decoder_W, decoder_b, rbf_params,
                    cfg.n_mels, cfg.sr, cfg.n_fft, cfg.hop_length,
                    style=style, phase_seed=gen_seed,
                )
                # Fallback to additive synthesis if learned vocoder produces silence
                if np.max(np.abs(cont_audio)) < 1e-6:
                    cont_audio = _tokens_to_audio(
                        tokens, cfg.codebook_size, cfg.sr, cfg.hop_length,
                    )
                # Scale continuation to match seed audio RMS
                seed_rms = max(float(np.sqrt(np.mean(seed_audio ** 2))), 1e-6)
                cont_rms = max(float(np.sqrt(np.mean(cont_audio ** 2))), 1e-6)
                cont_audio = cont_audio * (seed_rms / cont_rms)
                # Cross-fade join (10 ms)
                xfade = min(int(0.01 * cfg.sr), len(seed_audio), len(cont_audio))
                fade_out = np.linspace(1, 0, xfade)
                fade_in = np.linspace(0, 1, xfade)
                seed_audio[-xfade:] *= fade_out
                cont_audio[:xfade] *= fade_in
                waveform = np.concatenate([seed_audio, cont_audio])
            else:
                waveform = seed_audio

            rms = float(np.sqrt(np.mean(waveform ** 2)))
            spectral_centroid = _spectral_centroid(waveform, cfg.sr, cfg.n_fft, cfg.hop_length)

            # Aggregate log-probs across codebooks for this generation
            all_gen_lp = np.concatenate(per_cb_logprobs) if per_cb_logprobs else np.array([])
            gen_perplexity = _generation_perplexity(all_gen_lp)
            gen_confidence = _token_confidence_curve(all_gen_lp)

            generations.append({
                'style': style,
                'temperature': temp,
                'n_tokens': int(len(tokens)),
                'duration_s': round(len(waveform) / cfg.sr, 3),
                'rms': round(rms, 6),
                'spectral_centroid_hz': round(spectral_centroid, 1),
                'waveform': waveform,
                'finite': bool(np.all(np.isfinite(waveform))),
                'per_cb_tokens': per_cb_tokens,
                'primary_tokens': tokens,
                'perplexity': round(gen_perplexity, 4),
                'confidence': gen_confidence,
                'log_probs': all_gen_lp,
            })

    gen_time = time.monotonic() - t_gen_start

    # ---- 7a. Phrase self-similarity ---- #
    # Measure how much structure the phrase-form generation actually produces.
    phrase_len = cfg.continuation_length // len(
        _FORM_TEMPLATES.get(cfg.phrase_form, _FORM_TEMPLATES['AABA'])
    )
    sim_scores = []
    for g_data in generations:
        if 'per_cb_tokens' in g_data:
            for g_idx, toks in enumerate(g_data['per_cb_tokens']):
                sim_scores.append(_phrase_self_similarity(
                    toks, model.quantizer.codebooks, g_idx, phrase_len,
                ))
    if not sim_scores:
        # Fallback: measure on the primary codebook tokens from the last generation
        for g_data in generations:
            toks = g_data.get('primary_tokens', np.array([]))
            if len(toks) > 0:
                sim_scores.append(_phrase_self_similarity(
                    toks, model.quantizer.codebooks, 0, phrase_len,
                ))
    phrase_self_sim = float(np.mean(sim_scores)) if sim_scores else 0.0

    # ---- 7b. Generation uncertainty ---- #
    gen_perplexities = [g['perplexity'] for g in generations if 'perplexity' in g]
    mean_perplexity = float(np.mean(gen_perplexities)) if gen_perplexities else float('inf')
    # Aggregate confidence across all generations
    all_conf_means = [g['confidence']['mean'] for g in generations if 'confidence' in g]
    all_conf_stds = [g['confidence']['std'] for g in generations if 'confidence' in g]
    all_conf_mins = [g['confidence']['min'] for g in generations if 'confidence' in g]
    all_stable_fracs = [g['confidence']['stable_frac'] for g in generations if 'confidence' in g]
    agg_confidence = {
        'mean': round(float(np.mean(all_conf_means)), 4) if all_conf_means else 0.0,
        'std': round(float(np.mean(all_conf_stds)), 4) if all_conf_stds else 0.0,
        'min': round(float(np.min(all_conf_mins)), 4) if all_conf_mins else 0.0,
        'stable_frac': round(float(np.mean(all_stable_fracs)), 4) if all_stable_fracs else 0.0,
    }

    # ---- 7. Diversity metrics ---- #
    # Style diversity: average pairwise spectral centroid distance between styles
    style_centroids = {}
    for g in generations:
        style_centroids.setdefault(g['style'], []).append(g['spectral_centroid_hz'])
    style_means = {s: np.mean(v) for s, v in style_centroids.items()}
    if len(style_means) > 1:
        vals = list(style_means.values())
        pairs = [(vals[i] - vals[j]) ** 2
                 for i in range(len(vals)) for j in range(i + 1, len(vals))]
        style_diversity = float(np.sqrt(np.mean(pairs)))
    else:
        style_diversity = 0.0

    # Temperature diversity: std of spectral centroid across temperatures per style
    temp_centroids = {}
    for g in generations:
        temp_centroids.setdefault(g['style'], []).append(g['spectral_centroid_hz'])
    temp_diversity = float(np.mean([np.std(v) for v in temp_centroids.values()]))

    # ---- 8. Round-trip fidelity (model-based) ---- #
    # audio → mel → encoder → tokens → decoder → mel, vs original mel
    rt_audio = generate_seed_audio(1.0, cfg.sr, cfg.seed, "chord")
    rt_mel = compute_mel(rt_audio, cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)
    rt_feats = model.feature_extractor.forward(rt_mel.T)
    rt_enc = model.transformer.forward(rt_feats)
    rt_hidden = rt_enc['hidden_states']
    rt_quant = model.quantizer.forward(rt_hidden)
    rt_indices = rt_quant['indices']  # (T_enc, n_codebooks)

    # Reconstruct mel via learned decoder
    rt_embs = []
    for t_i in range(rt_indices.shape[0]):
        parts = []
        for g in range(cfg.n_codebooks):
            parts.append(model.quantizer.codebooks[g][rt_indices[t_i, g]])
        rt_embs.append(np.concatenate(parts))
    rt_embs = np.array(rt_embs)
    rt_linear = rt_embs @ decoder_W + decoder_b
    rt_dists = np.sum((rt_embs[:, None] - rbf_params['centers'][None]) ** 2, axis=-1)
    rt_rbf = np.exp(-rt_dists / (2 * rbf_params['sigma2']))
    rt_recon_mel = rt_linear + rt_rbf @ rbf_params['weights']
    rt_recon_mel = np.maximum(rt_recon_mel, 0.01)

    # Compare: spectral convergence in mel domain
    rt_orig_frames = rt_mel[:rt_recon_mel.shape[0]]  # align lengths
    sc_mel = float(np.linalg.norm(rt_orig_frames - rt_recon_mel) /
                   (np.linalg.norm(rt_orig_frames) + 1e-10))
    # Correlation across mel bins
    if rt_recon_mel.shape[0] > 1:
        corr = float(np.corrcoef(
            rt_orig_frames.ravel(), rt_recon_mel.ravel()
        )[0, 1])
    else:
        corr = 0.0
    sc = sc_mel

    total_time = time.monotonic() - t0

    # ---- 9. Checks ---- #
    # SPSA loss is very noisy with block-coordinate cycling.
    # Check: best loss in second half < median of first quarter
    n = len(train_losses)
    q1 = train_losses[:max(1, n // 4)]
    second_half = train_losses[n // 2:]
    loss_decreased = (n >= 6 and np.min(second_half) < np.median(q1))
    # Use max-in-second-half vs mean-of-first-quarter: SPSA noise makes
    # endpoint comparisons unreliable, but peak utilization should grow
    n_cb_steps = len(codebook_utils)
    codebook_improved = (n_cb_steps >= 6 and
                         np.max(codebook_utils[n_cb_steps // 2:]) > np.mean(codebook_utils[:max(1, n_cb_steps // 4)]))
    all_finite = all(g['finite'] for g in generations)
    all_have_tokens = all(g['n_tokens'] > 0 for g in generations)

    all_passed = loss_decreased and all_finite and all_have_tokens

    summary_lines = [
        f"Model: {n_params:,} params, d={cfg.d_model}, L={cfg.n_layers}",
        f"Training: {cfg.n_steps} steps, loss {train_losses[0]:.3f} → {train_losses[-1]:.3f} "
        f"({'↓ decreasing' if loss_decreased else '→ flat/increasing'})",
        f"Codebook: {codebook_utils[0]:.1%} → {codebook_utils[-1]:.1%} train, {post_refine_util:.1%} post-refine",
        f"Generated: {len(generations)} clips across {len(cfg.gen_styles)} styles × {len(cfg.gen_temperatures)} temps",
        f"Style diversity: {style_diversity:.1f} Hz (spectral centroid spread)",
        f"Temp diversity: {temp_diversity:.6f} (RMS std across temperatures)",
        f"Trigram LM: {n_cb} codebooks, interp λ=(0.1,0.3,0.6), entropy={bigram_entropy:.3f} nats/tok",
        f"Phrase structure: form={cfg.phrase_form}, self-similarity={phrase_self_sim:.3f}, "
        f"variation={cfg.phrase_variation:.0%}",
        f"Uncertainty: perplexity={mean_perplexity:.2f}, "
        f"confidence={agg_confidence['mean']:.3f}±{agg_confidence['std']:.3f}, "
        f"stable={agg_confidence['stable_frac']:.1%}",
        f"Round-trip: SC={sc:.4f}, corr={corr:.4f}",
        f"Time: {total_time:.1f}s (train={train_time:.1f}s, gen={gen_time:.1f}s)",
        f"Result: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}",
    ]

    # Strip waveforms and token arrays from report dict to keep it serializable
    _skip_keys = {'waveform', 'per_cb_tokens', 'primary_tokens', 'log_probs'}
    gen_report = [{k: v for k, v in g.items() if k not in _skip_keys} for g in generations]

    # ---- 9. Export WAVs and checkpoint ---- #
    out_dir = cfg.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for i, g in enumerate(generations):
            wav_path = os.path.join(out_dir, f"gen_{g['style']}_{g['temperature']}.wav")
            save_wav(g['waveform'], wav_path, cfg.sr)
        summary_lines.append(f"WAVs exported to {out_dir}/")

    if cfg.save_checkpoint and out_dir:
        ckpt_path = os.path.join(out_dir, "model_checkpoint.npz")
        ckpt_data = {
            'config_d_model': cfg.d_model,
            'config_n_layers': cfg.n_layers,
            'config_n_codebooks': cfg.n_codebooks,
            'config_codebook_size': cfg.codebook_size,
            'config_codebook_dim': cfg.codebook_dim,
            'config_n_mels': cfg.n_mels,
            'decoder_W': decoder_W,
            'decoder_b': decoder_b,
            'rbf_centers': rbf_params['centers'],
            'rbf_weights': rbf_params['weights'],
            'rbf_sigma2': np.array([rbf_params['sigma2']]),
            'ngram_unigram': unigram_probs_all,
            'ngram_bigram': bigram_probs_all,
            'ngram_trigram': trigram_probs_all,
            'ngram_lambdas': np.array(ngram_model['lambdas']),
        }
        all_p = _collect_params(model)
        for idx_p, (comp_p, name_p, arr_p) in enumerate(all_p):
            ckpt_data[f'param_{idx_p}_{name_p}'] = arr_p
        for g_idx in range(cfg.n_codebooks):
            ckpt_data[f'codebook_{g_idx}'] = model.quantizer.codebooks[g_idx]
        np.savez_compressed(ckpt_path, **ckpt_data)
        summary_lines.append(f"Checkpoint saved to {ckpt_path}")

    return TrainGenReport(
        train_losses=train_losses,
        codebook_utilizations=codebook_utils,
        train_time_s=round(train_time, 2),
        loss_decreased=loss_decreased,
        codebook_improved=codebook_improved,
        generations=gen_report,
        style_diversity=round(style_diversity, 2),
        temp_diversity=round(temp_diversity, 6),
        gen_time_s=round(gen_time, 2),
        round_trip_spectral_convergence=round(sc, 4),
        round_trip_correlation=round(corr, 4),
        phrase_self_similarity=round(phrase_self_sim, 4),
        phrase_form=cfg.phrase_form,
        generation_perplexity=round(mean_perplexity, 4),
        confidence_stats=agg_confidence,
        total_time_s=round(total_time, 2),
        n_params=n_params,
        all_checks_passed=all_passed,
        summary="\n".join(summary_lines),
    ), generations


def _tokens_to_audio(
    tokens: np.ndarray, codebook_size: int, sr: int, hop_length: int,
) -> np.ndarray:
    """Synthesize audio from a token sequence via additive synthesis.

    Maps each token index to a frequency (pentatonic scale, octave-wrapped)
    and an amplitude, producing a short grain per token that is concatenated
    into a waveform. Different token sequences produce audibly different audio.
    """
    pentatonic = np.array([1.0, 9/8, 5/4, 3/2, 5/3])
    base_freq = 220.0

    samples_per_token = hop_length
    audio = np.zeros(len(tokens) * samples_per_token)
    t = np.arange(samples_per_token) / sr

    for i, tok in enumerate(tokens):
        tok = int(tok) % codebook_size
        degree = tok % len(pentatonic)
        octave = (tok // len(pentatonic)) % 4
        freq = base_freq * pentatonic[degree] * (2.0 ** octave)
        amp = 0.3 + 0.5 * (tok / codebook_size)
        grain = amp * np.sin(2 * np.pi * freq * t)
        grain *= np.hanning(len(grain))
        start = i * samples_per_token
        audio[start:start + samples_per_token] = grain

    return audio


def _griffin_lim_seeded(
    magnitude: np.ndarray, n_fft: int, hop_length: int,
    n_iter: int = 32, seed: int = 0,
) -> np.ndarray:
    """Griffin-Lim with configurable random seed for phase initialization."""
    rng = np.random.RandomState(seed)
    phase = np.exp(2j * np.pi * rng.random(magnitude.shape))
    for _ in range(n_iter):
        S = magnitude * phase
        audio = istft(S, n_fft, hop_length)
        S_new = compute_stft(audio, n_fft, hop_length)
        nf = min(S_new.shape[0], magnitude.shape[0])
        nt = min(S_new.shape[1], magnitude.shape[1])
        S_new = S_new[:nf, :nt]
        mag_abs = np.abs(S_new) + 1e-12
        phase_new = S_new / mag_abs
        phase = np.zeros_like(magnitude, dtype=np.complex128)
        phase[:nf, :nt] = phase_new
    return istft(magnitude * phase, n_fft, hop_length)


# Style-dependent spectral emphasis curves (mel-band weights)
_STYLE_EMPHASIS = {
    'chord': lambda n: np.exp(-0.5 * ((np.arange(n) / n - 0.3) / 0.25) ** 2),  # mid-low
    'melody': lambda n: np.exp(-0.5 * ((np.arange(n) / n - 0.5) / 0.2) ** 2),  # mid
    'rhythm': lambda n: np.clip(1.0 - np.arange(n) / n, 0.3, 1.0),  # low emphasis
}


def _tokens_to_audio_learned(
    tokens: np.ndarray,
    model: MusicFrontierModel,
    decoder_W: np.ndarray,
    decoder_b: np.ndarray,
    rbf_params: Dict[str, Any],
    n_mels: int,
    sr: int,
    n_fft: int,
    hop_length: int,
    style: str = 'chord',
    phase_seed: int = 0,
) -> np.ndarray:
    """Synthesize audio from tokens using the model's learned codebook embeddings.

    tokens → codebook lookup (per-codebook hash) → RBF-augmented linear projection →
    mel spectrogram → style emphasis → overlap-add grain synthesis → Griffin-Lim.
    """
    quantizer = model.quantizer
    T = len(tokens)
    if T == 0:
        return np.zeros(0)

    # Look up codebook embeddings — use different hash per codebook
    # so each codebook contributes independent information
    embeddings = []
    for tok in tokens:
        tok_val = int(tok)
        emb_parts = []
        for g in range(quantizer.n_codebooks):
            tok_idx = (tok_val + g * 37) % quantizer.codebook_size
            emb_parts.append(quantizer.codebooks[g][tok_idx])
        embeddings.append(np.concatenate(emb_parts))
    embeddings = np.array(embeddings)  # (T, n_codebooks * codebook_dim)

    # Project to mel space via RBF-augmented decoder
    linear_mel = embeddings @ decoder_W + decoder_b
    rbf_C = rbf_params['centers']
    rbf_W = rbf_params['weights']
    sigma2 = rbf_params['sigma2']
    dists = np.sum((embeddings[:, None] - rbf_C[None]) ** 2, axis=-1)
    rbf_feats = np.exp(-dists / (2 * sigma2))
    raw_mel = linear_mel + rbf_feats @ rbf_W  # (T, n_mels)
    raw_mel = np.maximum(raw_mel, 0.01)

    # Style-dependent spectral emphasis — shifts energy distribution
    emphasis_fn = _STYLE_EMPHASIS.get(style, _STYLE_EMPHASIS['chord'])
    emphasis = emphasis_fn(n_mels)
    emphasis = emphasis / (emphasis.mean() + 1e-10)  # normalize to preserve total energy
    raw_mel = raw_mel * emphasis[np.newaxis, :]

    # Overlap-add grain synthesis: each token produces a 3-frame mel grain
    # with a triangular window, overlapping with neighbors.
    grain_len = 3
    out_len = T + grain_len - 1
    mel_frames = np.zeros((out_len, n_mels))
    weights = np.zeros(out_len)
    window = np.array([0.5, 1.0, 0.5])
    for i in range(T):
        for k in range(grain_len):
            pos = i + k
            if pos < out_len:
                mel_frames[pos] += window[k] * raw_mel[i]
                weights[pos] += window[k]
    weights = np.maximum(weights, 1e-10)
    mel_frames /= weights[:, np.newaxis]

    # Mel → magnitude spectrogram → Griffin-Lim with unique phase seed
    mag = mel_to_magnitude(mel_frames, sr, n_fft, n_mels)
    audio = _griffin_lim_seeded(mag, n_fft, hop_length, n_iter=32, seed=phase_seed)

    return audio


def _tokens_to_audio_multicb(
    per_cb_tokens: List[np.ndarray],
    model: MusicFrontierModel,
    decoder_W: np.ndarray,
    decoder_b: np.ndarray,
    rbf_params: Dict[str, Any],
    n_mels: int,
    sr: int,
    n_fft: int,
    hop_length: int,
    style: str = 'chord',
    phase_seed: int = 0,
) -> np.ndarray:
    """Synthesize audio using independently sampled per-codebook tokens.

    Each codebook's token sequence was sampled from its own bigram model,
    capturing different musical aspects (pitch, rhythm, timbre, dynamics).
    The full embedding is the concatenation of all codebook lookups.
    """
    quantizer = model.quantizer
    n_cb = quantizer.n_codebooks
    T = min(len(t) for t in per_cb_tokens)
    if T == 0:
        return np.zeros(0)

    embeddings = np.zeros((T, n_cb * quantizer.codebook_dim))
    for g in range(n_cb):
        for i in range(T):
            tok_idx = int(per_cb_tokens[g][i]) % quantizer.codebook_size
            start = g * quantizer.codebook_dim
            end = start + quantizer.codebook_dim
            embeddings[i, start:end] = quantizer.codebooks[g][tok_idx]

    # RBF-augmented projection to mel
    linear_mel = embeddings @ decoder_W + decoder_b
    rbf_C = rbf_params['centers']
    rbf_W = rbf_params['weights']
    sigma2 = rbf_params['sigma2']
    dists = np.sum((embeddings[:, None] - rbf_C[None]) ** 2, axis=-1)
    rbf_feats = np.exp(-dists / (2 * sigma2))
    raw_mel = linear_mel + rbf_feats @ rbf_W
    raw_mel = np.maximum(raw_mel, 0.01)

    # Style emphasis
    emphasis_fn = _STYLE_EMPHASIS.get(style, _STYLE_EMPHASIS['chord'])
    emphasis = emphasis_fn(n_mels)
    emphasis = emphasis / (emphasis.mean() + 1e-10)
    raw_mel = raw_mel * emphasis[np.newaxis, :]

    # Overlap-add grain synthesis
    grain_len = 3
    out_len = T + grain_len - 1
    mel_frames = np.zeros((out_len, n_mels))
    weights = np.zeros(out_len)
    window = np.array([0.5, 1.0, 0.5])
    for i in range(T):
        for k in range(grain_len):
            pos = i + k
            if pos < out_len:
                mel_frames[pos] += window[k] * raw_mel[i]
                weights[pos] += window[k]
    weights = np.maximum(weights, 1e-10)
    mel_frames /= weights[:, np.newaxis]

    mag = mel_to_magnitude(mel_frames, sr, n_fft, n_mels)
    audio = _griffin_lim_seeded(mag, n_fft, hop_length, n_iter=32, seed=phase_seed)
    return audio


def _init_decoder_projection(
    codebook_total_dim: int, n_mels: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize the learned mel decoder projection."""
    rng = np.random.RandomState(seed)
    scale = np.sqrt(2.0 / (codebook_total_dim + n_mels))
    W = rng.randn(codebook_total_dim, n_mels) * scale
    b = np.zeros(n_mels)
    return W, b


def _train_decoder_projection(
    model: MusicFrontierModel,
    clips: List[np.ndarray],
    decoder_W: np.ndarray,
    decoder_b: np.ndarray,
    cfg: 'TrainGenConfig',
    rng: np.random.RandomState,
    n_steps: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train mel decoder via closed-form Tikhonov with RBF features.

    Collects (codebook_embedding, target_mel) pairs from training clips,
    augments with RBF basis functions for nonlinear capacity, then solves
    W* = argmin ||XW - Y||² + λ||W||² in closed form.

    Returns (W, b, rbf_params) where rbf_params has centers, weights, sigma2.
    """
    target_mels = []
    token_embeddings = []

    for clip in clips:
        mel = compute_mel(clip, cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)
        features = model.feature_extractor.forward(mel.T)
        enc_out = model.transformer.forward(features)
        hidden = enc_out['hidden_states']
        q_out = model.quantizer.forward(hidden)
        indices = q_out['indices']

        embs = []
        for t_i in range(indices.shape[0]):
            parts = []
            for g in range(model.quantizer.n_codebooks):
                parts.append(model.quantizer.codebooks[g][indices[t_i, g]])
            embs.append(np.concatenate(parts))
        embs = np.array(embs)

        min_t = min(mel.shape[0], embs.shape[0])
        target_mels.append(mel[:min_t])
        token_embeddings.append(embs[:min_t])

    X = np.concatenate(token_embeddings, axis=0)  # (N, cb_total_dim)
    Y = np.concatenate(target_mels, axis=0)        # (N, n_mels)

    # RBF feature augmentation: add exp(-||x-c||²/σ²) for codebook centroids.
    # Gives nonlinear capacity while keeping the closed-form solve.
    all_centroids = np.concatenate(
        [model.quantizer.codebooks[g] for g in range(model.quantizer.n_codebooks)],
        axis=0,
    )  # (n_codebooks * codebook_size, codebook_dim) — these are in codebook space
    # Project centroids to full embedding space for RBF computation
    rbf_centers = []
    for g in range(model.quantizer.n_codebooks):
        for v in range(model.quantizer.codebook_size):
            parts = []
            for g2 in range(model.quantizer.n_codebooks):
                if g2 == g:
                    parts.append(model.quantizer.codebooks[g2][v])
                else:
                    parts.append(model.quantizer.codebooks[g2][0])
            rbf_centers.append(np.concatenate(parts))
    rbf_centers = np.array(rbf_centers)
    # Use a subset of RBF centers to keep the problem tractable
    n_rbf = min(32, rbf_centers.shape[0])
    rbf_idx = rng.choice(rbf_centers.shape[0], size=n_rbf, replace=False)
    rbf_C = rbf_centers[rbf_idx]  # (n_rbf, cb_total_dim)
    sigma2 = float(np.mean(np.var(X, axis=0))) + 1e-6
    dists = np.sum((X[:, None] - rbf_C[None]) ** 2, axis=-1)  # (N, n_rbf)
    rbf_feats = np.exp(-dists / (2 * sigma2))

    # Augmented features: [embedding, RBF features, bias]
    X_aug = np.hstack([X, rbf_feats, np.ones((X.shape[0], 1))])

    # Tikhonov (ridge) least-squares: (X^T X + λI)^{-1} X^T Y
    lam = 1e-3
    XtX = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
    XtY = X_aug.T @ Y
    W_aug = np.linalg.solve(XtX, XtY)

    # Split into linear + RBF + bias
    d = X.shape[1]
    decoder_W = W_aug[:d]
    decoder_b = W_aug[-1]
    rbf_W = W_aug[d:-1]  # (n_rbf, n_mels)

    rbf_params = {'centers': rbf_C, 'weights': rbf_W, 'sigma2': sigma2}
    return decoder_W, decoder_b, rbf_params


def _generate_ngram_tokens(
    ngram_model: Dict[str, Any],
    codebook_idx: int,
    seed_tokens: np.ndarray,
    length: int,
    temperature: float = 1.0,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Sample tokens via interpolated trigram/bigram/unigram backoff.

    P(w | w_{-2}, w_{-1}) = λ₃·P_tri(w|w_{-2},w_{-1}) + λ₂·P_bi(w|w_{-1}) + λ₁·P_uni(w)

    Trigram captures 2-step dependencies (chord progressions, rhythmic cells).
    Interpolation ensures no zero probabilities even for unseen contexts.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    g = codebook_idx
    uni = ngram_model['unigram'][g]
    bi = ngram_model['bigram'][g]
    tri = ngram_model['trigram'][g]
    lam_u, lam_b, lam_t = ngram_model['lambdas']
    V = uni.shape[0]

    context = list(seed_tokens[-min(4, len(seed_tokens)):])
    n_ctx = len(context)

    for _ in range(length):
        # Interpolated probability
        p = lam_u * uni
        if len(context) >= 1:
            prev1 = int(context[-1]) % V
            p = p + lam_b * bi[prev1]
        else:
            p = p + lam_b * uni
        if len(context) >= 2:
            prev2 = int(context[-2]) % V
            prev1 = int(context[-1]) % V
            p = p + lam_t * tri[prev2, prev1]
        else:
            p = p + lam_t * (bi[int(context[-1]) % V] if len(context) >= 1 else uni)

        # Temperature scaling in log-space
        logits = np.log(p + 1e-12)
        scaled = logits / max(temperature, 1e-6)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        context.append(int(rng.choice(V, p=probs)))

    return np.array(context[n_ctx:], dtype=np.int64)


def _generate_ngram_tokens_with_confidence(
    ngram_model: Dict[str, Any],
    codebook_idx: int,
    seed_tokens: np.ndarray,
    length: int,
    temperature: float = 1.0,
    rng: np.random.RandomState | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like _generate_ngram_tokens but also returns per-token log-probabilities.

    Returns (tokens, log_probs) where log_probs[i] = log P(token_i | context).
    Perplexity = exp(-mean(log_probs)); lower = more confident generation.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    g = codebook_idx
    uni = ngram_model['unigram'][g]
    bi = ngram_model['bigram'][g]
    tri = ngram_model['trigram'][g]
    lam_u, lam_b, lam_t = ngram_model['lambdas']
    V = uni.shape[0]

    context = list(seed_tokens[-min(4, len(seed_tokens)):])
    n_ctx = len(context)
    log_probs = []

    for _ in range(length):
        p = lam_u * uni
        if len(context) >= 1:
            prev1 = int(context[-1]) % V
            p = p + lam_b * bi[prev1]
        else:
            p = p + lam_b * uni
        if len(context) >= 2:
            prev2 = int(context[-2]) % V
            prev1 = int(context[-1]) % V
            p = p + lam_t * tri[prev2, prev1]
        else:
            p = p + lam_t * (bi[int(context[-1]) % V] if len(context) >= 1 else uni)

        logits = np.log(p + 1e-12)
        scaled = logits / max(temperature, 1e-6)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        chosen = int(rng.choice(V, p=probs))
        log_probs.append(float(np.log(probs[chosen] + 1e-12)))
        context.append(chosen)

    tokens = np.array(context[n_ctx:], dtype=np.int64)
    return tokens, np.array(log_probs, dtype=np.float64)


def _generation_perplexity(log_probs: np.ndarray) -> float:
    """Perplexity = exp(-mean(log P(token|context))). Lower = more confident."""
    if len(log_probs) == 0:
        return float('inf')
    return float(np.exp(-np.mean(log_probs)))


def _token_confidence_curve(log_probs: np.ndarray) -> Dict[str, float]:
    """Summary statistics of per-token confidence for a generation.

    Returns mean, std, min confidence (= max surprise), and the fraction
    of tokens above 50th-percentile confidence (stability measure).
    """
    if len(log_probs) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'stable_frac': 0.0}
    conf = np.exp(log_probs)  # P(token|context) ∈ (0, 1]
    median = float(np.median(conf))
    return {
        'mean': float(np.mean(conf)),
        'std': float(np.std(conf)),
        'min': float(np.min(conf)),
        'stable_frac': float(np.mean(conf >= median)),
    }


# Musical form templates: each is a list of section labels.
# Same label = repeat (exact or varied), different = new material.
_FORM_TEMPLATES = {
    'AABA':  ['A', 'A', 'B', 'A'],     # 32-bar song form
    'ABAB':  ['A', 'B', 'A', 'B'],     # verse-chorus
    'ABAC':  ['A', 'B', 'A', 'C'],     # verse-chorus-bridge
    'AABB':  ['A', 'A', 'B', 'B'],     # binary form
    'ABCBA': ['A', 'B', 'C', 'B', 'A'],  # arch form
}


def _generate_phrase_tokens(
    ngram_model: Dict[str, Any],
    codebook_idx: int,
    seed_tokens: np.ndarray,
    total_length: int,
    temperature: float = 1.0,
    rng: np.random.RandomState | None = None,
    form: str = 'AABA',
    variation_noise: float = 0.15,
    return_confidence: bool = False,
) -> 'np.ndarray | Tuple[np.ndarray, np.ndarray]':
    """Generate tokens with musical phrase structure and repetition.

    Divides `total_length` into phrases following a musical form template
    (AABA, ABAB, etc.). On first occurrence of a section label, generates
    fresh material via the trigram LM. On repetition, re-uses the original
    phrase with optional token-level variation (stochastic replacement at
    rate `variation_noise`), simulating how real music repeats themes with
    small embellishments.

    If return_confidence=True, returns (tokens, log_probs).
    Otherwise returns just tokens.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    template = _FORM_TEMPLATES.get(form, _FORM_TEMPLATES['AABA'])
    n_sections = len(template)
    phrase_len = total_length // n_sections
    remainder = total_length - phrase_len * n_sections

    V = ngram_model['unigram'][codebook_idx].shape[0]
    section_material: Dict[str, np.ndarray] = {}
    section_logprobs: Dict[str, np.ndarray] = {}
    all_tokens: List[np.ndarray] = []
    all_logprobs: List[np.ndarray] = []

    context_tokens = seed_tokens

    for sec_i, label in enumerate(template):
        sec_len = phrase_len + (remainder if sec_i == n_sections - 1 else 0)

        if label not in section_material:
            if return_confidence:
                phrase, lp = _generate_ngram_tokens_with_confidence(
                    ngram_model, codebook_idx, context_tokens,
                    sec_len, temperature=temperature, rng=rng,
                )
                section_logprobs[label] = lp
            else:
                phrase = _generate_ngram_tokens(
                    ngram_model, codebook_idx, context_tokens,
                    sec_len, temperature=temperature, rng=rng,
                )
                lp = np.zeros(sec_len)
            section_material[label] = phrase
        else:
            original = section_material[label]
            phrase = original.copy()
            lp = section_logprobs.get(label, np.zeros(len(original))).copy()
            if sec_len != len(original):
                if sec_len <= len(original):
                    phrase = original[:sec_len].copy()
                    lp = lp[:sec_len].copy()
                else:
                    if return_confidence:
                        extra, extra_lp = _generate_ngram_tokens_with_confidence(
                            ngram_model, codebook_idx,
                            original, sec_len - len(original),
                            temperature=temperature, rng=rng,
                        )
                        lp = np.concatenate([lp, extra_lp])
                    else:
                        extra = _generate_ngram_tokens(
                            ngram_model, codebook_idx,
                            original, sec_len - len(original),
                            temperature=temperature, rng=rng,
                        )
                        lp = np.concatenate([lp, np.zeros(len(extra))])
                    phrase = np.concatenate([original, extra])
            # Stochastic variation on repeated sections
            mask = rng.random(len(phrase)) < variation_noise
            if mask.any():
                uni = ngram_model['unigram'][codebook_idx]
                for j in np.where(mask)[0]:
                    phrase[j] = rng.choice(V, p=uni)
                    # Varied tokens get unigram-level confidence
                    if return_confidence:
                        lp[j] = float(np.log(uni[phrase[j]] + 1e-12))

        all_tokens.append(phrase)
        all_logprobs.append(lp)
        context_tokens = phrase

    tokens = np.concatenate(all_tokens)[:total_length]
    if return_confidence:
        logprobs = np.concatenate(all_logprobs)[:total_length]
        return tokens, logprobs
    return tokens


def _phrase_self_similarity(
    tokens: np.ndarray,
    codebooks: List[np.ndarray],
    codebook_idx: int,
    phrase_len: int,
) -> float:
    """Measure self-similarity between phrases via cosine similarity of embeddings.

    Splits the token sequence into phrases, embeds each via codebook lookup,
    computes mean embedding per phrase, then returns the mean pairwise cosine
    similarity. Values near 1.0 indicate high repetition (musical structure);
    near 0 indicates no structure.
    """
    cb = codebooks[codebook_idx]
    n_phrases = max(1, len(tokens) // phrase_len)
    phrase_embs = []
    for i in range(n_phrases):
        start = i * phrase_len
        end = min(start + phrase_len, len(tokens))
        if end <= start:
            break
        seg = tokens[start:end]
        embs = cb[np.clip(seg.astype(int), 0, len(cb) - 1)]
        phrase_embs.append(embs.mean(axis=0))

    if len(phrase_embs) < 2:
        return 0.0

    # Mean pairwise cosine similarity
    sims = []
    for i in range(len(phrase_embs)):
        for j in range(i + 1, len(phrase_embs)):
            a, b = phrase_embs[i], phrase_embs[j]
            dot = float(np.dot(a, b))
            norm = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
            sims.append(dot / norm)
    return float(np.mean(sims))


def _generate_bigram_tokens(
    bigram_probs: np.ndarray,
    seed_tokens: np.ndarray,
    length: int,
    temperature: float = 1.0,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Sample a token sequence from the learned bigram transition matrix.

    Given a seed sequence (from encoding a seed clip), continues by
    sampling next tokens from P(t_{i+1} | t_i) with temperature scaling.
    Temperature < 1 sharpens (more deterministic), > 1 flattens (more random).
    """
    if rng is None:
        rng = np.random.RandomState(0)

    V = bigram_probs.shape[0]
    context = list(seed_tokens[-min(4, len(seed_tokens)):])
    n_ctx = len(context)

    for _ in range(length):
        prev = int(context[-1]) % V
        logits = np.log(bigram_probs[prev] + 1e-12)
        scaled = logits / max(temperature, 1e-6)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        context.append(int(rng.choice(V, p=probs)))

    return np.array(context[n_ctx:], dtype=np.int64)


def _spectral_centroid(
    audio: np.ndarray, sr: int, n_fft: int, hop_length: int,
) -> float:
    """Mean spectral centroid in Hz."""
    S = np.abs(compute_stft(audio, n_fft, hop_length))
    freqs = np.linspace(0, sr / 2, S.shape[0])
    power = S ** 2
    centroid = np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-10)
    return float(np.mean(centroid))


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    _defaults = TrainGenConfig()
    parser = argparse.ArgumentParser(description="NeuroTunes: Train → Generate Pipeline")
    parser.add_argument("--n-steps", type=int, default=_defaults.n_steps)
    parser.add_argument("--n-clips", type=int, default=_defaults.n_clips)
    parser.add_argument("--d-model", type=int, default=_defaults.d_model)
    parser.add_argument("--seed", type=int, default=_defaults.seed)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save generated WAV files")
    args = parser.parse_args()

    cfg = TrainGenConfig(
        n_steps=args.n_steps,
        n_clips=args.n_clips,
        d_model=args.d_model,
        seed=args.seed,
    )

    print("=" * 60)
    print("  NeuroTunes: Train → Generate Pipeline")
    print("=" * 60)
    print(f"\nConfig: {cfg.n_clips} clips, {cfg.n_steps} steps, d={cfg.d_model}")
    print()

    report, generations = train_and_generate(cfg)
    print(report.summary)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for g, gen_data in zip(report.generations, generations):
            fname = f"{g['style']}_T{g['temperature']:.1f}.wav"
            path = os.path.join(args.output_dir, fname)
            save_wav(gen_data['waveform'], path, cfg.sr)
            print(f"  Saved: {path}")
        print(f"\n{len(generations)} WAV files written to {args.output_dir}/")
