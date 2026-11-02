"""Tests for audio pipeline: segmentation, augmentation, preprocessing, synthetic generation."""
import numpy as np
import pytest


class TestAudioSegmenter:
    def test_short_audio_padded(self):
        from models.music_foundation.audio_pipeline import AudioSegmenter
        seg = AudioSegmenter(segment_samples=1000, overlap_ratio=0.5)
        audio = np.ones(500)
        result = seg.segment(audio)
        assert len(result) == 1
        assert result[0].shape == (1000,)
        assert np.allclose(result[0][:500], 1.0)
        assert np.allclose(result[0][500:], 0.0)

    def test_exact_length(self):
        from models.music_foundation.audio_pipeline import AudioSegmenter
        seg = AudioSegmenter(segment_samples=1000, overlap_ratio=0.0)
        audio = np.ones(1000)
        result = seg.segment(audio)
        assert len(result) == 1
        assert result[0].shape == (1000,)

    def test_overlap_produces_more_segments(self):
        from models.music_foundation.audio_pipeline import AudioSegmenter
        seg_no_overlap = AudioSegmenter(segment_samples=1000, overlap_ratio=0.0)
        seg_overlap = AudioSegmenter(segment_samples=1000, overlap_ratio=0.5)
        audio = np.random.randn(3000)
        no_ol = seg_no_overlap.segment(audio)
        ol = seg_overlap.segment(audio)
        assert len(ol) >= len(no_ol)

    def test_long_audio_covers_all_samples(self):
        from models.music_foundation.audio_pipeline import AudioSegmenter
        seg = AudioSegmenter(segment_samples=1000, overlap_ratio=0.0)
        audio = np.random.randn(2500)
        result = seg.segment(audio)
        assert len(result) == 3
        assert result[-1].shape == (1000,)


class TestAudioAugmenter:
    def test_time_stretch_identity(self):
        from models.music_foundation.audio_pipeline import AudioAugmenter
        aug = AudioAugmenter(seed=0)
        audio = np.sin(2 * np.pi * 440 * np.arange(1000) / 22050)
        stretched = aug.time_stretch(audio, 1.0)
        assert np.allclose(audio, stretched)

    def test_time_stretch_slower(self):
        from models.music_foundation.audio_pipeline import AudioAugmenter
        aug = AudioAugmenter(seed=0)
        audio = np.sin(2 * np.pi * 440 * np.arange(1000) / 22050)
        stretched = aug.time_stretch(audio, 0.5)
        assert len(stretched) > len(audio)

    def test_pitch_shift_preserves_length(self):
        from models.music_foundation.audio_pipeline import AudioAugmenter
        aug = AudioAugmenter(seed=0)
        audio = np.sin(2 * np.pi * 440 * np.arange(2000) / 22050)
        shifted = aug.pitch_shift(audio, 2.0)
        assert len(shifted) == len(audio)

    def test_add_noise_changes_signal(self):
        from models.music_foundation.audio_pipeline import AudioAugmenter
        aug = AudioAugmenter(seed=0)
        audio = np.sin(2 * np.pi * 440 * np.arange(1000) / 22050)
        noisy = aug.add_noise(audio, snr_db=10.0)
        assert not np.allclose(audio, noisy)
        assert noisy.shape == audio.shape

    def test_random_gain_changes_amplitude(self):
        from models.music_foundation.audio_pipeline import AudioAugmenter
        aug = AudioAugmenter(seed=0)
        audio = np.ones(100) * 0.5
        gained = aug.random_gain(audio, -6, 6)
        assert not np.allclose(audio, gained)

    def test_augment_composite(self):
        from models.music_foundation.audio_pipeline import AudioAugmenter
        aug = AudioAugmenter(seed=42)
        audio = np.sin(2 * np.pi * 440 * np.arange(2000) / 22050)
        result = aug.augment(audio)
        assert len(result) == len(audio)
        assert np.all(np.isfinite(result))


class TestAudioPreprocessor:
    def test_normalize(self):
        from models.music_foundation.audio_pipeline import AudioPreprocessor, AudioPipelineConfig
        cfg = AudioPipelineConfig(segment_duration_s=1.0, sr=1000, normalize=True, augment=False)
        proc = AudioPreprocessor(cfg)
        audio = np.ones(2000) * 5.0
        segments = proc.process(audio)
        assert np.max(np.abs(segments)) <= 1.0 + 1e-8

    def test_segmentation(self):
        from models.music_foundation.audio_pipeline import AudioPreprocessor, AudioPipelineConfig
        cfg = AudioPipelineConfig(segment_duration_s=1.0, sr=1000, overlap_ratio=0.0,
                                  normalize=False, augment=False)
        proc = AudioPreprocessor(cfg)
        audio = np.random.randn(3000)
        segments = proc.process(audio)
        assert segments.ndim == 2
        assert segments.shape[1] == 1000

    def test_process_batch(self):
        from models.music_foundation.audio_pipeline import AudioPreprocessor, AudioPipelineConfig
        cfg = AudioPipelineConfig(segment_duration_s=0.5, sr=1000, overlap_ratio=0.0,
                                  normalize=False, augment=False)
        proc = AudioPreprocessor(cfg)
        batch = [np.random.randn(1000), np.random.randn(2000)]
        result = proc.process_batch(batch)
        assert result.ndim == 2
        assert result.shape[1] == 500
        assert result.shape[0] >= 4

    def test_augmentation_enabled(self):
        from models.music_foundation.audio_pipeline import AudioPreprocessor, AudioPipelineConfig
        cfg = AudioPipelineConfig(segment_duration_s=1.0, sr=1000, augment=True, seed=42)
        proc = AudioPreprocessor(cfg)
        audio = np.sin(2 * np.pi * 100 * np.arange(2000) / 1000)
        segments = proc.process(audio)
        assert segments.ndim == 2
        assert np.all(np.isfinite(segments))


class TestSyntheticAudioGenerator:
    def test_sine_tone(self):
        from models.music_foundation.audio_pipeline import SyntheticAudioGenerator
        gen = SyntheticAudioGenerator(seed=0)
        tone = gen.sine_tone(440.0, 1.0, 22050)
        assert tone.shape == (22050,)
        assert np.max(np.abs(tone)) <= 1.0 + 1e-8

    def test_chord(self):
        from models.music_foundation.audio_pipeline import SyntheticAudioGenerator
        gen = SyntheticAudioGenerator(seed=0)
        c = gen.chord([261.6, 329.6, 392.0], 0.5, 22050)
        assert c.shape == (11025,)
        assert np.max(np.abs(c)) <= 1.0 + 1e-8

    def test_drum_hit(self):
        from models.music_foundation.audio_pipeline import SyntheticAudioGenerator
        gen = SyntheticAudioGenerator(seed=0)
        hit = gen.drum_hit(0.5, 22050)
        assert hit.shape == (11025,)
        assert np.abs(hit[-1]) < np.abs(hit[0]) or np.abs(hit[0]) < 1e-6

    def test_generate_dataset(self):
        from models.music_foundation.audio_pipeline import SyntheticAudioGenerator
        gen = SyntheticAudioGenerator(seed=42)
        dataset = gen.generate_dataset(10, duration_s=0.5, sr=8000)
        assert len(dataset) == 10
        assert all(len(s) == 4000 for s in dataset)
        assert all(np.all(np.isfinite(s)) for s in dataset)

    def test_empty_chord(self):
        from models.music_foundation.audio_pipeline import SyntheticAudioGenerator
        gen = SyntheticAudioGenerator(seed=0)
        c = gen.chord([], 0.5, 22050)
        assert np.allclose(c, 0.0)
