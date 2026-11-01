"""Tests for the music_visualization module."""
import numpy as np
import pytest

from models.music_visualization.waveform import (
    WaveformRegion, WaveformRenderer, WaveformThumbnail,
)
from models.music_visualization.spectrogram import (
    SpectrogramRenderer, MelSpectrogram, LogSpectrogram, SpectrogramColormap,
)
from models.music_visualization.harmonic_circle import (
    CircleOfFifths, ChromaWheel, TonnetzProjection, KeyRegion,
)
from models.music_visualization.piano_roll import (
    NoteLabel, VelocityColorMap, PianoRollRenderer, PianoRollOverlay,
)
from models.music_visualization.rhythm_viz import (
    RhythmGrid, OnsetPlot, TempoGraph, GrooveMap,
)
from models.music_visualization.analysis_plots import (
    LoudnessCurve, DynamicRangePlot, FrequencyResponse, StereoFieldPlot,
)


# --- Waveform ---

class TestWaveformRegion:
    def test_contains(self):
        r = WaveformRegion(1.0, 3.0, "verse")
        assert r.contains(2.0)
        assert not r.contains(0.5)

    def test_duration(self):
        r = WaveformRegion(1.0, 4.5)
        assert abs(r.duration() - 3.5) < 1e-6


class TestWaveformRenderer:
    def test_peak_envelope_shape(self):
        sig = np.random.randn(44100).astype(np.float32)
        pos, neg = WaveformRenderer.peak_envelope(sig, 100)
        assert len(pos) == 100
        assert len(neg) == 100
        assert np.all(pos >= neg)

    def test_rms_envelope_nonneg(self):
        sig = np.sin(np.linspace(0, 10, 4410)).astype(np.float32)
        rms = WaveformRenderer.rms_envelope(sig, 50)
        assert len(rms) == 50
        assert np.all(rms >= 0)

    def test_segment_colors_shape(self):
        sig = np.random.randn(8820).astype(np.float32)
        colors = WaveformRenderer.segment_colors_by_energy(sig, 40)
        assert colors.shape == (40, 3)
        assert colors.dtype == np.uint8

    def test_empty_signal(self):
        pos, neg = WaveformRenderer.peak_envelope(np.array([]), 10)
        assert len(pos) == 0


class TestWaveformThumbnail:
    def test_thumbnail_shape(self):
        sig = np.random.randn(22050).astype(np.float32)
        img = WaveformThumbnail.generate(sig, 100, 30)
        assert img.shape == (30, 100)
        assert img.dtype == np.uint8

    def test_thumbnail_has_content(self):
        sig = np.sin(np.linspace(0, 20, 44100)).astype(np.float32)
        img = WaveformThumbnail.generate(sig, 200, 50)
        assert np.sum(img > 0) > 0


# --- Spectrogram ---

class TestSpectrogramRenderer:
    def test_shape(self):
        sig = np.random.randn(44100).astype(np.float32)
        spec = SpectrogramRenderer.compute(sig, 44100, 1024, 256)
        assert spec.shape[0] == 513  # n_fft/2 + 1
        assert spec.shape[1] > 0

    def test_to_db(self):
        spec = np.ones((10, 5), dtype=np.float32)
        db = SpectrogramRenderer.to_db(spec)
        assert np.allclose(db, 0.0, atol=0.01)

    def test_frequency_axis(self):
        freqs = SpectrogramRenderer.frequency_axis(2048, 44100)
        assert len(freqs) == 1025
        assert freqs[0] == 0.0
        assert abs(freqs[-1] - 22050.0) < 1.0

    def test_time_axis(self):
        times = SpectrogramRenderer.time_axis(10, 512, 44100)
        assert len(times) == 10


class TestMelSpectrogram:
    def test_filterbank_shape(self):
        fb = MelSpectrogram.mel_filterbank(44100, 2048, 40)
        assert fb.shape == (40, 1025)

    def test_compute(self):
        sig = np.random.randn(22050).astype(np.float32)
        mel = MelSpectrogram.compute(sig, 44100, 2048, 512, 40)
        assert mel.shape[0] == 40
        assert mel.shape[1] > 0


class TestLogSpectrogram:
    def test_compute(self):
        spec = np.random.rand(1025, 20).astype(np.float32)
        log_spec = LogSpectrogram.compute(spec, 44100, 2048, 64)
        assert log_spec.shape == (64, 20)


class TestSpectrogramColormap:
    def test_apply_shape(self):
        mag = np.random.rand(10, 20).astype(np.float32)
        rgb = SpectrogramColormap.apply(mag, "viridis")
        assert rgb.shape == (10, 20, 3)
        assert rgb.dtype == np.uint8

    def test_all_colormaps(self):
        mag = np.random.rand(5, 5).astype(np.float32)
        for cm in ["viridis", "magma", "inferno", "plasma"]:
            rgb = SpectrogramColormap.apply(mag, cm)
            assert rgb.shape == (5, 5, 3)


# --- Harmonic circle ---

class TestCircleOfFifths:
    def test_positions_count(self):
        pos = CircleOfFifths.positions()
        assert len(pos) == 12
        assert pos[0]["name"] == "C"

    def test_highlight(self):
        h = CircleOfFifths.highlight(["C", "E", "G"])
        active = [p for p in h if p["active"]]
        assert len(active) == 3

    def test_chord_major(self):
        pos = CircleOfFifths.chord_positions("C", "major")
        active = [p for p in pos if p["active"]]
        assert len(active) == 3


class TestChromaWheel:
    def test_map_chroma(self):
        chroma = np.zeros(12, dtype=np.float32)
        chroma[0] = 1.0  # C
        result = ChromaWheel.map_chroma(chroma)
        assert len(result) == 12
        assert result[0]["intensity"] == 1.0
        assert result[1]["intensity"] == 0.0

    def test_centroid_uniform(self):
        chroma = np.ones(12, dtype=np.float32)
        cx, cy = ChromaWheel.centroid(chroma)
        assert abs(cx) < 0.01
        assert abs(cy) < 0.01


class TestTonnetzProjection:
    def test_pitch_to_2d(self):
        x, y = TonnetzProjection.pitch_to_2d(0)
        assert isinstance(x, float)

    def test_chord_triangle(self):
        tri = TonnetzProjection.chord_triangle(0, "major")
        assert len(tri) == 3

    def test_project_chroma(self):
        chroma = np.ones(12, dtype=np.float32)
        proj = TonnetzProjection.project_chroma(chroma)
        assert len(proj) == 12


class TestKeyRegion:
    def test_c_major(self):
        region = KeyRegion.get_region("C", "major")
        in_key = [p for p in region if p["in_key"]]
        assert len(in_key) == 7

    def test_a_minor(self):
        region = KeyRegion.get_region("A", "minor")
        in_key = [p for p in region if p["in_key"]]
        assert len(in_key) == 7


# --- Piano roll ---

class TestNoteLabel:
    def test_middle_c(self):
        assert NoteLabel.from_midi(60) == "C4"

    def test_flats(self):
        label = NoteLabel.from_midi(61, use_flats=True)
        assert label == "Db4"

    def test_range(self):
        labels = NoteLabel.labels_for_range(60, 72)
        assert len(labels) == 13
        assert labels[0] == "C4"


class TestVelocityColorMap:
    def test_zero(self):
        r, g, b = VelocityColorMap.map_velocity(0)
        assert r == 0 and g == 0 and b == 0

    def test_max(self):
        r, g, b = VelocityColorMap.map_velocity(127)
        assert r > 0 or g > 0 or b > 0

    def test_gradient(self):
        grad = VelocityColorMap.gradient()
        assert grad.shape == (128, 3)


class TestPianoRollRenderer:
    def test_render(self):
        notes = [
            {"pitch": 60, "start": 0.0, "duration": 0.5, "velocity": 100},
            {"pitch": 64, "start": 0.5, "duration": 0.5, "velocity": 80},
        ]
        grid = PianoRollRenderer.render(notes, 1.0, 0.01)
        assert grid.shape[0] == 88  # 108 - 21 + 1
        assert grid.shape[1] == 100

    def test_to_rgb(self):
        grid = np.random.rand(20, 50).astype(np.float32)
        rgb = PianoRollRenderer.to_rgb(grid)
        assert rgb.shape == (20, 50, 3)


class TestPianoRollOverlay:
    def test_beat_grid(self):
        markers = PianoRollOverlay.beat_grid(1000, 0.01, 120.0)
        assert len(markers) == 1000
        assert markers[0] == 1.0

    def test_key_shading(self):
        shading = PianoRollOverlay.key_shading(88, root=0, pitch_low=21)
        assert len(shading) == 88
        assert np.any(shading == 1.0)
        assert np.any(shading < 1.0)


# --- Rhythm viz ---

class TestRhythmGrid:
    def test_create(self):
        grid = RhythmGrid.create(8, 4, [(0, 0), (2, 2)])
        assert grid.shape == (8, 4)
        assert grid[0, 0] == 1.0
        assert grid[2, 2] == 1.0
        assert grid[1, 1] == 0.0

    def test_density(self):
        grid = np.ones((4, 4), dtype=np.float32)
        assert RhythmGrid.density(grid) == 1.0

    def test_from_onsets(self):
        onsets = np.array([0.0, 0.5, 1.0, 1.5])
        grid = RhythmGrid.from_onsets(onsets, 120.0, 4, 4)
        assert grid.shape == (4, 4)


class TestOnsetPlot:
    def test_create(self):
        strengths = np.random.rand(100).astype(np.float32)
        result = OnsetPlot.create(strengths)
        assert len(result["times"]) == 100
        assert len(result["strengths"]) == 100

    def test_peak_onsets(self):
        s = np.array([0, 0.1, 0.5, 0.2, 0, 0, 0.8, 0.3, 0], dtype=np.float32)
        peaks = OnsetPlot.peak_onsets(s, 0.3)
        assert 2 in peaks
        assert 6 in peaks


class TestTempoGraph:
    def test_from_beats(self):
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = TempoGraph.from_beats(beats)
        assert len(result["bpm"]) == 4
        assert np.allclose(result["bpm"], 120.0)

    def test_stability_constant(self):
        beats = np.linspace(0, 4, 9)
        s = TempoGraph.tempo_stability(beats)
        assert s > 0.95


class TestGrooveMap:
    def test_compute(self):
        onsets = np.array([0.0, 0.51, 1.01, 1.49])
        gm = GrooveMap.compute(onsets, 120.0, 4, 1)
        assert gm.shape[1] == 4

    def test_mean_deviation(self):
        gm = np.array([[0.0, 5.0, -3.0, 2.0]], dtype=np.float32)
        md = GrooveMap.mean_deviation(gm)
        assert md > 0


# --- Analysis plots ---

class TestLoudnessCurve:
    def test_compute(self):
        sig = np.random.randn(44100).astype(np.float32) * 0.5
        result = LoudnessCurve.compute(sig, 44100, 400, 100)
        assert len(result["times"]) == len(result["lufs"])
        assert len(result["times"]) > 0

    def test_integrated(self):
        sig = np.sin(np.linspace(0, 100, 44100)).astype(np.float32)
        lufs = LoudnessCurve.integrated(sig)
        assert lufs < 0


class TestDynamicRangePlot:
    def test_compute(self):
        sig = np.random.randn(22050).astype(np.float32)
        result = DynamicRangePlot.compute(sig, 44100, 50)
        assert len(result["peak"]) == len(result["rms"])
        assert np.all(result["peak"] >= result["rms"])

    def test_crest(self):
        sig = np.sin(np.linspace(0, 50, 44100)).astype(np.float32)
        result = DynamicRangePlot.crest_factor_series(sig, 44100, 50)
        assert np.all(result["crest"] >= 1.0)


class TestFrequencyResponse:
    def test_compute(self):
        sig = np.random.randn(8192).astype(np.float32)
        result = FrequencyResponse.compute(sig, 44100, 4096)
        assert len(result["frequencies"]) == 2049
        assert len(result["magnitude_db"]) == 2049

    def test_smoothed(self):
        sig = np.random.randn(8192).astype(np.float32)
        result = FrequencyResponse.smoothed(sig, 44100, 4096)
        assert len(result["magnitude_db"]) == 2049


class TestStereoFieldPlot:
    def test_lissajous(self):
        stereo = np.random.randn(1000, 2).astype(np.float32)
        result = StereoFieldPlot.lissajous(stereo)
        assert len(result["x"]) == len(result["y"])

    def test_goniometer(self):
        stereo = np.random.randn(500, 2).astype(np.float32)
        result = StereoFieldPlot.goniometer(stereo)
        assert len(result["r"]) == len(result["theta"])

    def test_correlation_identical(self):
        mono = np.random.randn(1000).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        corr = StereoFieldPlot.correlation(stereo)
        assert abs(corr - 1.0) < 0.01

    def test_width_mono(self):
        mono = np.random.randn(1000).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        w = StereoFieldPlot.width_estimate(stereo)
        assert w < 0.05
