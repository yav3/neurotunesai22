"""Tests for streaming pipeline module."""
import numpy as np
import pytest
from models.music_pipeline.streaming_pipeline import (
    RingBuffer, AudioChunk, StreamProcessor, FeatureFrame,
    rms_energy, zero_crossing_rate, spectral_centroid, spectral_rolloff,
    spectral_flux, OnsetDetector, chunk_audio, process_full_signal,
    LatencyTracker, overlap_add,
)


class TestRingBuffer:
    def test_basic_write_read(self):
        buf = RingBuffer(10)
        buf.write(np.array([1, 2, 3]))
        assert buf.count == 3
        data = buf.read(3)
        np.testing.assert_array_equal(data, [1, 2, 3])

    def test_wrap_around(self):
        buf = RingBuffer(5)
        buf.write(np.array([1, 2, 3, 4]))
        buf.write(np.array([5, 6, 7]))
        assert buf.count == 5
        data = buf.read()
        assert len(data) == 5

    def test_overflow(self):
        buf = RingBuffer(3)
        buf.write(np.arange(10))
        assert buf.count == 3
        data = buf.read()
        np.testing.assert_array_equal(data, [7, 8, 9])

    def test_clear(self):
        buf = RingBuffer(10)
        buf.write(np.array([1, 2, 3]))
        buf.clear()
        assert buf.count == 0


class TestFeatures:
    def test_rms_silence(self):
        assert rms_energy(np.zeros(100)) == 0.0

    def test_rms_signal(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100)
        r = rms_energy(sig)
        assert 0.5 < r < 1.0

    def test_zcr_sine(self):
        sig = np.sin(2 * np.pi * 100 * np.arange(1000) / 1000)
        z = zero_crossing_rate(sig)
        assert 0.1 < z < 0.5

    def test_centroid_low(self):
        t = np.arange(2048) / 44100
        low = np.sin(2 * np.pi * 200 * t)
        high = np.sin(2 * np.pi * 8000 * t)
        assert spectral_centroid(low, 44100) < spectral_centroid(high, 44100)

    def test_rolloff(self):
        sig = np.random.randn(2048)
        r = spectral_rolloff(sig, 44100)
        assert 0 < r < 22050

    def test_flux_identical(self):
        spec = np.abs(np.fft.rfft(np.random.randn(1024)))
        assert spectral_flux(spec, spec) == 0.0


class TestOnsetDetector:
    def test_detects_onset(self):
        sr = 44100
        chunk_size = 1024
        silence = np.zeros(chunk_size)
        burst = np.random.randn(chunk_size) * 10
        det = OnsetDetector(threshold=0.2)
        for _ in range(5):
            det.process(silence)
        result = det.process(burst)
        assert result is True


class TestStreamProcessor:
    def test_basic_processing(self):
        sr = 44100
        hop = 1024
        proc = StreamProcessor(sample_rate=sr, hop_size=hop)
        sig = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        chunks = chunk_audio(sig, sr, hop * 2)
        all_frames = []
        for c in chunks:
            frames = proc.push(c)
            all_frames.extend(frames)
        assert len(all_frames) > 5
        assert all("rms" in f.features for f in all_frames)

    def test_reset(self):
        proc = StreamProcessor(sample_rate=44100, hop_size=512)
        proc.push(AudioChunk(np.random.randn(1024), 44100, 0.0))
        proc.reset()
        assert len(proc.get_all_features()) == 0


class TestProcessFullSignal:
    def test_basic(self):
        sig = np.random.randn(44100)
        frames = process_full_signal(sig, 44100, hop_size=2048)
        assert len(frames) > 5


class TestLatencyTracker:
    def test_stats(self):
        lt = LatencyTracker()
        for v in [10.0, 12.0, 11.0, 15.0, 10.0]:
            lt.record(v)
        assert 10 < lt.mean_latency() < 15
        assert lt.p99_latency() >= lt.mean_latency()
        assert lt.jitter() > 0

    def test_empty(self):
        lt = LatencyTracker()
        assert lt.mean_latency() == 0.0


class TestOverlapAdd:
    def test_reconstruction(self):
        hop = 256
        frame_len = 1024
        sig = np.random.randn(4096)
        frames = []
        for i in range(0, len(sig) - frame_len, hop):
            frames.append(sig[i:i + frame_len].copy())
        result = overlap_add(frames, hop)
        assert len(result) > 0

    def test_empty(self):
        assert len(overlap_add([], 256)) == 0
