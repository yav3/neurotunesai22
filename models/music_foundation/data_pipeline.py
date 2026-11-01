"""Audio data pipeline for the frontier music foundation model.

End-to-end pipeline: load audio → preprocess → batch → augment → spectrogram.
Supports streaming iteration, curriculum sampling, on-the-fly augmentation,
and contrastive two-view generation for self-supervised pretraining.

Pure numpy/scipy. No torch, librosa, or torchaudio.
"""
from __future__ import annotations

import os
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass
class AudioClip:
    signal: np.ndarray
    sr: int
    path: Optional[str] = None

    @property
    def duration(self) -> float:
        return len(self.signal) / self.sr

    @property
    def n_samples(self) -> int:
        return len(self.signal)


def load_wav(path: str, target_sr: Optional[int] = None, mono: bool = True) -> AudioClip:
    """Load a WAV file. Handles 16-bit PCM and 32-bit float."""
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    if mono and data.ndim > 1:
        data = data.mean(axis=1)
    if target_sr is not None and target_sr != sr:
        from scipy.signal import resample
        n_out = int(len(data) * target_sr / sr)
        data = resample(data, n_out)
        sr = target_sr
    return AudioClip(signal=data, sr=sr, path=path)


def load_directory(
    dir_path: str,
    target_sr: int = 22050,
    extensions: Tuple[str, ...] = (".wav",),
    mono: bool = True,
    max_files: Optional[int] = None,
) -> List[AudioClip]:
    """Load all audio files from a directory."""
    clips = []
    for fname in sorted(os.listdir(dir_path)):
        if not any(fname.lower().endswith(ext) for ext in extensions):
            continue
        fpath = os.path.join(dir_path, fname)
        try:
            clip = load_wav(fpath, target_sr, mono)
            clips.append(clip)
        except Exception:
            continue
        if max_files is not None and len(clips) >= max_files:
            break
    return clips


def generate_synthetic_clips(
    n_clips: int = 100,
    sr: int = 22050,
    duration: float = 3.0,
    seed: int = 42,
) -> List[AudioClip]:
    """Generate synthetic audio clips for testing/development."""
    rng = np.random.RandomState(seed)
    clips = []
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr
    for i in range(n_clips):
        n_harmonics = rng.randint(3, 12)
        f0 = rng.uniform(80, 800)
        signal = np.zeros(n_samples)
        for h in range(1, n_harmonics + 1):
            amp = rng.uniform(0.1, 1.0) / h
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * f0 * h * t + phase)
        am_rate = rng.uniform(1, 8)
        am_depth = rng.uniform(0, 0.5)
        signal *= 1 + am_depth * np.sin(2 * np.pi * am_rate * t)
        noise_level = rng.uniform(0.001, 0.05)
        signal += rng.randn(n_samples) * noise_level
        signal /= np.max(np.abs(signal)) + 1e-8
        clips.append(AudioClip(signal=signal, sr=sr))
    return clips


# ---------- spectrogram pipeline ----------

@dataclass
class SpectrogramConfig:
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: float = 20.0
    fmax: Optional[float] = None
    power: float = 2.0
    log_offset: float = 1e-6
    normalize: str = "per_sample"  # "per_sample", "global", "none"
    ref_db: float = -20.0


def _hann_window(n: int) -> np.ndarray:
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    fmax = fmax or sr / 2.0
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel_lo = hz2mel(fmin)
    mel_hi = hz2mel(fmax)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs))
    for m in range(n_mels):
        lo, mid, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        for k in range(lo, mid):
            if mid > lo:
                fb[m, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi > mid:
                fb[m, k] = (hi - k) / (hi - mid)
    return fb


def compute_mel_spectrogram(
    signal: np.ndarray,
    sr: int,
    config: Optional[SpectrogramConfig] = None,
) -> np.ndarray:
    """Signal → log-mel spectrogram. Returns (n_mels, T)."""
    if config is None:
        config = SpectrogramConfig()
    n_fft = config.n_fft
    hop = config.hop_length
    window = _hann_window(n_fft)

    pad_len = n_fft // 2
    padded = np.concatenate([np.zeros(pad_len), signal, np.zeros(pad_len)])
    n_frames = 1 + (len(padded) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop, padded.strides[0]),
    ).copy()
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=n_fft)
    power_spec = np.abs(spectrum) ** config.power

    fb = _mel_filterbank(sr, n_fft, config.n_mels, config.fmin, config.fmax or sr / 2)
    mel_spec = fb @ power_spec.T  # (n_mels, T)
    log_mel = np.log(mel_spec + config.log_offset)

    if config.normalize == "per_sample":
        mu = log_mel.mean()
        std = log_mel.std() + 1e-8
        log_mel = (log_mel - mu) / std
    elif config.normalize == "global":
        log_mel = log_mel - config.ref_db

    return log_mel


def mel_to_audio(
    mel_spec: np.ndarray,
    sr: int,
    config: Optional[SpectrogramConfig] = None,
    n_iter: int = 60,
) -> np.ndarray:
    """Approximate inverse via Griffin-Lim (Griffin & Lim, 1984)."""
    if config is None:
        config = SpectrogramConfig()
    n_fft = config.n_fft
    hop = config.hop_length
    n_freqs = n_fft // 2 + 1
    _, n_frames = mel_spec.shape

    fb = _mel_filterbank(sr, n_fft, config.n_mels, config.fmin, config.fmax or sr / 2)
    fb_pinv = np.linalg.pinv(fb)
    linear = np.maximum(fb_pinv @ mel_spec, 0)
    magnitude = np.sqrt(linear + 1e-10)

    rng = np.random.RandomState(0)
    phase = np.exp(2j * np.pi * rng.rand(n_freqs, n_frames))
    window = _hann_window(n_fft)
    out_len = (n_frames - 1) * hop + n_fft

    for _ in range(n_iter):
        complex_spec = magnitude * phase
        frames = np.fft.irfft(complex_spec.T, n=n_fft)
        frames *= window
        signal = np.zeros(out_len)
        win_sum = np.zeros(out_len)
        for i in range(n_frames):
            start = i * hop
            signal[start:start + n_fft] += frames[i]
            win_sum[start:start + n_fft] += window ** 2
        win_sum = np.maximum(win_sum, 1e-8)
        signal /= win_sum
        stft = np.fft.rfft(
            np.lib.stride_tricks.as_strided(
                signal,
                shape=(n_frames, n_fft),
                strides=(signal.strides[0] * hop, signal.strides[0]),
            ).copy() * window,
            n=n_fft,
        )
        phase = np.exp(1j * np.angle(stft.T))

    return signal[:out_len]


def process_batch_spectrograms(
    signals: List[np.ndarray],
    sr: int,
    config: Optional[SpectrogramConfig] = None,
) -> np.ndarray:
    """Process a batch of signals → stacked spectrograms for the frontier model.

    Returns (batch, n_mels, max_T), zero-padded to the longest in the batch.
    """
    specs = [compute_mel_spectrogram(s, sr, config) for s in signals]
    max_t = max(s.shape[1] for s in specs)
    n_mels = specs[0].shape[0]
    batch = np.zeros((len(specs), n_mels, max_t))
    for i, s in enumerate(specs):
        batch[i, :, :s.shape[1]] = s
    return batch


# ---------- streaming batcher ----------

@dataclass
class BatcherConfig:
    batch_size: int = 8
    segment_length: int = 88200  # 4 seconds at 22050 Hz
    overlap: float = 0.0
    shuffle: bool = True
    drop_last: bool = False


class StreamingBatcher:
    """Yields fixed-length audio segments in batches."""

    def __init__(
        self,
        clips: List[AudioClip],
        config: Optional[BatcherConfig] = None,
        seed: int = 42,
    ):
        self.clips = clips
        self.config = config or BatcherConfig()
        self.seed = seed
        self._segments: List[np.ndarray] = []
        self._prepare_segments()
        self._epoch = 0
        self._idx = 0

    def _prepare_segments(self) -> None:
        seg_len = self.config.segment_length
        hop = int(seg_len * (1 - self.config.overlap))
        hop = max(hop, 1)
        self._segments = []
        for clip in self.clips:
            sig = clip.signal
            if len(sig) < seg_len:
                padded = np.zeros(seg_len)
                padded[:len(sig)] = sig
                self._segments.append(padded)
            else:
                start = 0
                while start + seg_len <= len(sig):
                    self._segments.append(sig[start:start + seg_len])
                    start += hop

    def __len__(self) -> int:
        n = len(self._segments)
        bs = self.config.batch_size
        if self.config.drop_last:
            return n // bs
        return (n + bs - 1) // bs

    def epoch(self, epoch_num: Optional[int] = None) -> "StreamingBatcher":
        self._epoch = epoch_num if epoch_num is not None else self._epoch + 1
        self._idx = 0
        return self

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.RandomState(self.seed + self._epoch)
        order = np.arange(len(self._segments))
        if self.config.shuffle:
            rng.shuffle(order)
        bs = self.config.batch_size
        for start in range(0, len(order), bs):
            end = start + bs
            if end > len(order) and self.config.drop_last:
                break
            indices = order[start:min(end, len(order))]
            batch = np.stack([self._segments[i] for i in indices])
            yield batch


# ---------- on-the-fly augmentation ----------

@dataclass
class AugScheduleConfig:
    initial_prob: float = 0.3
    final_prob: float = 0.8
    warmup_steps: int = 1000
    noise_snr_range: Tuple[float, float] = (15.0, 30.0)
    gain_db_range: Tuple[float, float] = (-6.0, 6.0)
    pitch_semitone_range: Tuple[float, float] = (-2.0, 2.0)


def _augment_signal(
    signal: np.ndarray,
    prob: float,
    config: AugScheduleConfig,
    rng: np.random.RandomState,
) -> np.ndarray:
    result = signal.copy()
    if rng.random() < prob:
        gain_db = rng.uniform(*config.gain_db_range)
        result = result * 10 ** (gain_db / 20.0)
    if rng.random() < prob * 0.5:
        snr = rng.uniform(*config.noise_snr_range)
        power = np.mean(result ** 2)
        if power > 1e-10:
            noise_power = power / (10 ** (snr / 10))
            result = result + rng.randn(len(result)) * np.sqrt(noise_power)
    if rng.random() < prob * 0.3:
        result = -result
    return result


class OnTheFlyAugmenter:
    """Applies augmentation during iteration, optionally generating two views."""

    def __init__(
        self,
        batcher: StreamingBatcher,
        aug_config: Optional[AugScheduleConfig] = None,
        two_view: bool = False,
        seed: int = 42,
    ):
        self.batcher = batcher
        self.aug_config = aug_config or AugScheduleConfig()
        self.two_view = two_view
        self.seed = seed
        self._step = 0
        self._total_steps = 10000

    def set_schedule(self, total_steps: int) -> None:
        self._total_steps = total_steps

    def _current_prob(self) -> float:
        t = min(self._step / max(self.aug_config.warmup_steps, 1), 1.0)
        return self.aug_config.initial_prob + t * (
            self.aug_config.final_prob - self.aug_config.initial_prob
        )

    def __iter__(self):
        prob = self._current_prob()
        for batch in self.batcher:
            rng = np.random.RandomState(self.seed + self._step)
            if self.two_view:
                view_a = np.stack([
                    _augment_signal(batch[i], prob, self.aug_config, rng)
                    for i in range(len(batch))
                ])
                rng2 = np.random.RandomState(self.seed + self._step + 7919)
                view_b = np.stack([
                    _augment_signal(batch[i], prob, self.aug_config, rng2)
                    for i in range(len(batch))
                ])
                yield view_a, view_b
            else:
                augmented = np.stack([
                    _augment_signal(batch[i], prob, self.aug_config, rng)
                    for i in range(len(batch))
                ])
                yield augmented
            self._step += 1


# ---------- dataset splitting ----------

def stratified_split(
    n_samples: int,
    labels: np.ndarray,
    ratios: Tuple[float, ...] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    """Stratified split into train/val/test (or arbitrary number of folds)."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    splits = [[] for _ in ratios]

    for lab in unique_labels:
        indices = np.where(labels == lab)[0]
        rng.shuffle(indices)
        n = len(indices)
        boundaries = np.cumsum([0] + [int(r * n) for r in ratios[:-1]])
        boundaries = np.append(boundaries, n)
        for i in range(len(ratios)):
            splits[i].extend(indices[boundaries[i]:boundaries[i + 1]])

    return tuple(np.array(s) for s in splits)


def k_fold_indices(
    n_samples: int,
    k: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate k-fold cross-validation index pairs."""
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_size = n_samples // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, val_idx))
    return folds


# ---------- curriculum sampling ----------

def spectral_complexity(signal: np.ndarray, sr: int, n_fft: int = 1024) -> float:
    """Estimate spectral complexity: spectral entropy averaged over time."""
    hop = n_fft // 4
    n_frames = max(1, (len(signal) - n_fft) // hop)
    window = _hann_window(n_fft)
    entropy_sum = 0.0
    for i in range(n_frames):
        frame = signal[i * hop:i * hop + n_fft] * window
        mag = np.abs(np.fft.rfft(frame))
        mag = mag / (mag.sum() + 1e-10)
        ent = -np.sum(mag * np.log(mag + 1e-10))
        entropy_sum += ent
    return entropy_sum / max(n_frames, 1)


def dynamic_range(signal: np.ndarray, frame_size: int = 2048) -> float:
    """Estimate dynamic range in dB."""
    n_frames = max(1, len(signal) // frame_size)
    rms_values = []
    for i in range(n_frames):
        frame = signal[i * frame_size:(i + 1) * frame_size]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        rms_values.append(rms)
    rms_arr = np.array(rms_values)
    return float(20 * np.log10(rms_arr.max() / (rms_arr.min() + 1e-10) + 1e-10))


def difficulty_score(signal: np.ndarray, sr: int) -> float:
    """Combined difficulty score in [0, 1] for curriculum sampling."""
    complexity = spectral_complexity(signal, sr)
    dr = dynamic_range(signal)
    complexity_norm = np.clip(complexity / 8.0, 0, 1)
    dr_norm = np.clip(dr / 60.0, 0, 1)
    return 0.6 * complexity_norm + 0.4 * dr_norm


class CurriculumSampler:
    """Sample batches with increasing difficulty over training."""

    def __init__(
        self,
        clips: List[AudioClip],
        batch_size: int = 8,
        seed: int = 42,
    ):
        self.clips = clips
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self._scores = np.array([difficulty_score(c.signal, c.sr) for c in clips])
        self._order = np.argsort(self._scores)

    def sample(
        self,
        step: int,
        total_steps: int,
    ) -> List[np.ndarray]:
        """Return a batch appropriate for the current training step."""
        progress = min(step / max(total_steps, 1), 1.0)
        max_difficulty = 0.3 + 0.7 * progress
        eligible = self._order[self._scores[self._order] <= max_difficulty]
        if len(eligible) < self.batch_size:
            eligible = self._order[:max(self.batch_size, len(self._order))]
        chosen = self.rng.choice(eligible, size=min(self.batch_size, len(eligible)), replace=False)
        return [self.clips[i].signal for i in chosen]


# ---------- end-to-end pipeline ----------

@dataclass
class PipelineConfig:
    sr: int = 22050
    segment_length: float = 4.0
    batch_size: int = 8
    spec_config: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    batcher_config: BatcherConfig = field(default_factory=BatcherConfig)
    aug_config: AugScheduleConfig = field(default_factory=AugScheduleConfig)
    two_view: bool = True
    curriculum: bool = False
    shuffle: bool = True


class AudioPipeline:
    """End-to-end: clips → batched spectrograms ready for the frontier model."""

    def __init__(
        self,
        clips: List[AudioClip],
        config: Optional[PipelineConfig] = None,
        seed: int = 42,
    ):
        self.clips = clips
        self.config = config or PipelineConfig()
        self.seed = seed
        bc = self.config.batcher_config
        bc.segment_length = int(self.config.segment_length * self.config.sr)
        bc.batch_size = self.config.batch_size
        bc.shuffle = self.config.shuffle
        self.batcher = StreamingBatcher(clips, bc, seed)
        self.augmenter = OnTheFlyAugmenter(
            self.batcher,
            self.config.aug_config,
            two_view=self.config.two_view,
            seed=seed,
        )
        self.curriculum_sampler = (
            CurriculumSampler(clips, self.config.batch_size, seed)
            if self.config.curriculum else None
        )

    def epoch(self, epoch_num: int) -> None:
        self.batcher.epoch(epoch_num)

    def __iter__(self):
        sc = self.config.spec_config
        sr = self.config.sr
        for item in self.augmenter:
            if self.config.two_view:
                view_a, view_b = item
                spec_a = process_batch_spectrograms(
                    [view_a[i] for i in range(len(view_a))], sr, sc
                )
                spec_b = process_batch_spectrograms(
                    [view_b[i] for i in range(len(view_b))], sr, sc
                )
                yield spec_a, spec_b
            else:
                batch = item
                specs = process_batch_spectrograms(
                    [batch[i] for i in range(len(batch))], sr, sc
                )
                yield specs

    def get_curriculum_batch(self, step: int, total_steps: int) -> np.ndarray:
        """Get a curriculum-sampled batch as spectrograms."""
        if self.curriculum_sampler is None:
            raise ValueError("Curriculum sampling not enabled")
        signals = self.curriculum_sampler.sample(step, total_steps)
        return process_batch_spectrograms(signals, self.config.sr, self.config.spec_config)


# ---------- normalization utilities ----------

def compute_dataset_stats(
    clips: List[AudioClip],
    config: Optional[SpectrogramConfig] = None,
) -> Dict[str, np.ndarray]:
    """Compute per-frequency-bin mean and std across the dataset."""
    if config is None:
        config = SpectrogramConfig()
    specs = [compute_mel_spectrogram(c.signal, c.sr, SpectrogramConfig(
        n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels,
        fmin=config.fmin, fmax=config.fmax, normalize="none",
    )) for c in clips]
    all_frames = np.concatenate(specs, axis=1)  # (n_mels, total_T)
    return {
        "mean": all_frames.mean(axis=1),
        "std": all_frames.std(axis=1) + 1e-8,
        "min": all_frames.min(axis=1),
        "max": all_frames.max(axis=1),
    }


def apply_global_normalization(
    spectrogram: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """Apply dataset-level normalization using precomputed stats."""
    return (spectrogram - stats["mean"][:, np.newaxis]) / stats["std"][:, np.newaxis]
