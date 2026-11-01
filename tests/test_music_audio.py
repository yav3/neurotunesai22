"""Tests for music_audio module: spatial, stems, MIDI, effects."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_audio.spatial import (
    SpatialPosition, SpatialPanner, AmbisonicsEncoder, RoomSimulator,
)
from models.music_audio.stems import (
    StemType, SpectralStemEstimator, StemRemixer,
)
from models.music_audio.midi import (
    MIDINote, MIDITrack, MIDISequence, PianoRollConverter,
)
from models.music_audio.effects import (
    AudioEffect, Reverb, Delay, Compressor, EQBand, EQ, EffectsChain,
)


def _tone(freq=440.0, dur=0.1, sr=22050):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------

class TestSpatialPanner:
    def test_stereo_center(self):
        sig = _tone()
        out = SpatialPanner.stereo_pan(sig, 0.0)
        assert out.shape == (len(sig), 2)
        np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-5)

    def test_stereo_hard_left(self):
        sig = _tone()
        out = SpatialPanner.stereo_pan(sig, -1.0)
        assert np.max(np.abs(out[:, 0])) > np.max(np.abs(out[:, 1]))

    def test_stereo_hard_right(self):
        sig = _tone()
        out = SpatialPanner.stereo_pan(sig, 1.0)
        assert np.max(np.abs(out[:, 1])) > np.max(np.abs(out[:, 0]))

    def test_binaural_shape(self):
        sig = _tone()
        out = SpatialPanner.binaural_pan(sig, 45.0)
        assert out.ndim == 2
        assert out.shape[1] == 2
        assert out.shape[0] >= len(sig)

    def test_surround_encode(self):
        sig = _tone()
        positions = [SpatialPosition(azimuth=0), SpatialPosition(azimuth=180)]
        out = SpatialPanner.surround_encode(sig, positions, n_channels=6)
        assert out.shape == (len(sig), 6)


class TestAmbisonicsEncoder:
    def test_bformat_keys(self):
        sig = _tone()
        bf = AmbisonicsEncoder.encode_bformat(sig, 0.0)
        assert set(bf.keys()) == {"W", "X", "Y", "Z"}
        for ch in bf.values():
            assert len(ch) == len(sig)

    def test_bformat_front(self):
        sig = _tone()
        bf = AmbisonicsEncoder.encode_bformat(sig, 0.0, 0.0)
        assert np.max(np.abs(bf["X"])) > 0.1
        assert np.max(np.abs(bf["Y"])) < 1e-5

    def test_bformat_side(self):
        sig = _tone()
        bf = AmbisonicsEncoder.encode_bformat(sig, 90.0, 0.0)
        assert np.max(np.abs(bf["Y"])) > 0.1


class TestRoomSimulator:
    def test_early_reflections_longer(self):
        sig = _tone(dur=0.05)
        out = RoomSimulator.early_reflections(sig, (5, 4, 3), (1, 1, 1), (3, 3, 1))
        assert len(out) > len(sig)

    def test_reverb_tail_length(self):
        sig = _tone(dur=0.05)
        out = RoomSimulator.reverb_tail(sig, rt60=0.5, sr=22050)
        expected_len = len(sig) + int(0.5 * 22050)
        assert len(out) == expected_len


# ---------------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------------

class TestSpectralStemEstimator:
    def test_separates_three_stems(self):
        sig = _tone(100, 0.1) + _tone(1000, 0.1) + _tone(8000, 0.1)
        est = SpectralStemEstimator()
        stems = est.separate(sig)
        assert StemType.BASS in stems
        assert StemType.VOCALS in stems
        assert StemType.OTHER in stems

    def test_bass_has_low_freq(self):
        sig = _tone(100, 0.2)
        est = SpectralStemEstimator()
        stems = est.separate(sig)
        assert np.max(np.abs(stems[StemType.BASS])) > 0.1

    def test_reconstruction(self):
        sig = _tone(440, 0.1)
        est = SpectralStemEstimator()
        stems = est.separate(sig)
        reconstructed = sum(stems.values())
        np.testing.assert_allclose(reconstructed, sig, atol=0.01)


class TestStemRemixer:
    def test_remix_unity(self):
        stems = {StemType.BASS: np.ones(100, dtype=np.float32), StemType.VOCALS: np.ones(100, dtype=np.float32)}
        out = StemRemixer.remix(stems)
        np.testing.assert_allclose(out, 2.0, atol=1e-5)

    def test_isolate(self):
        stems = {StemType.BASS: np.ones(100, dtype=np.float32), StemType.VOCALS: np.zeros(100, dtype=np.float32)}
        out = StemRemixer.isolate(stems, [StemType.BASS])
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_remove(self):
        stems = {StemType.BASS: np.ones(100, dtype=np.float32), StemType.VOCALS: np.ones(100, dtype=np.float32) * 2}
        out = StemRemixer.remove(stems, [StemType.BASS])
        np.testing.assert_allclose(out, 2.0, atol=1e-5)

    def test_empty_stems(self):
        out = StemRemixer.remix({})
        assert len(out) == 0


# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------

class TestMIDINote:
    def test_end_time(self):
        n = MIDINote(pitch=60, start_time=1.0, duration=0.5)
        assert n.end_time == 1.5

    def test_note_name(self):
        assert MIDINote(pitch=60).note_name == "C4"
        assert MIDINote(pitch=69).note_name == "A4"


class TestMIDITrack:
    def test_add_and_duration(self):
        t = MIDITrack()
        t.add_note(MIDINote(pitch=60, start_time=0.0, duration=1.0))
        t.add_note(MIDINote(pitch=64, start_time=0.5, duration=1.0))
        assert t.duration == 1.5

    def test_piano_roll_shape(self):
        t = MIDITrack()
        t.add_note(MIDINote(pitch=60, start_time=0.0, duration=0.5, velocity=100))
        roll = t.to_piano_roll(resolution=0.01)
        assert roll.shape[0] == 128
        assert roll.shape[1] == 50
        assert roll[60, 0] > 0

    def test_quantize(self):
        t = MIDITrack()
        t.add_note(MIDINote(pitch=60, start_time=0.13, duration=0.37))
        q = t.quantize(grid_size=0.25)
        assert q.notes[0].start_time == 0.25
        assert q.notes[0].duration == 0.25

    def test_empty_track(self):
        t = MIDITrack()
        assert t.duration == 0.0
        roll = t.to_piano_roll()
        assert roll.shape == (128, 0)


class TestMIDISequence:
    def test_merge_tracks(self):
        s = MIDISequence()
        t1 = MIDITrack()
        t1.add_note(MIDINote(pitch=60, start_time=0.0, duration=1.0))
        t2 = MIDITrack()
        t2.add_note(MIDINote(pitch=64, start_time=0.5, duration=1.0))
        s.add_track(t1)
        s.add_track(t2)
        merged = s.merge_tracks()
        assert len(merged.notes) == 2
        assert merged.notes[0].start_time <= merged.notes[1].start_time

    def test_transpose(self):
        s = MIDISequence()
        t = MIDITrack()
        t.add_note(MIDINote(pitch=60))
        s.add_track(t)
        up = s.transpose(5)
        assert up.tracks[0].notes[0].pitch == 65

    def test_transpose_clamps(self):
        s = MIDISequence()
        t = MIDITrack()
        t.add_note(MIDINote(pitch=126))
        s.add_track(t)
        up = s.transpose(10)
        assert up.tracks[0].notes[0].pitch == 127

    def test_tempo_scale(self):
        s = MIDISequence()
        t = MIDITrack(tempo=120)
        t.add_note(MIDINote(pitch=60, start_time=1.0, duration=1.0))
        s.add_track(t)
        fast = s.tempo_scale(2.0)
        assert fast.tracks[0].tempo == 240.0
        assert fast.tracks[0].notes[0].start_time == pytest.approx(0.5)


class TestPianoRollConverter:
    def test_roundtrip(self):
        notes = [MIDINote(pitch=60, velocity=100, start_time=0.0, duration=0.5)]
        roll = PianoRollConverter.notes_to_piano_roll(notes, resolution=0.01)
        recovered = PianoRollConverter.piano_roll_to_notes(roll, resolution=0.01)
        assert len(recovered) >= 1
        assert recovered[0].pitch == 60

    def test_empty(self):
        roll = PianoRollConverter.notes_to_piano_roll([])
        assert roll.shape == (128, 0)


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

class TestEffects:
    def test_passthrough(self):
        sig = _tone()
        out = AudioEffect().process(sig)
        np.testing.assert_allclose(out, sig, atol=1e-6)

    def test_reverb_not_silent(self):
        sig = _tone()
        out = Reverb(room_size=0.5, mix=0.5).process(sig)
        assert len(out) == len(sig)
        assert np.max(np.abs(out)) > 0

    def test_delay_shape(self):
        sig = _tone()
        out = Delay(delay_time=0.05, feedback=0.3, mix=0.5).process(sig)
        assert len(out) == len(sig)

    def test_compressor_reduces_peaks(self):
        sig = np.concatenate([np.ones(100) * 0.1, np.ones(100) * 1.0]).astype(np.float32)
        out = Compressor(threshold_db=-6, ratio=4).process(sig)
        assert np.mean(out[100:]) < np.mean(sig[100:])

    def test_eq_boost(self):
        sig = _tone(1000, 0.1)
        eq = EQ(bands=[EQBand(center_freq=1000, gain_db=6.0, q=2.0)])
        out = eq.process(sig)
        assert np.max(np.abs(out)) > np.max(np.abs(sig)) * 0.9

    def test_eq_no_bands(self):
        sig = _tone()
        out = EQ().process(sig)
        np.testing.assert_allclose(out, sig, atol=1e-5)

    def test_chain(self):
        sig = _tone()
        chain = EffectsChain()
        chain.add(Reverb(mix=0.2)).add(Delay(delay_time=0.01, mix=0.1))
        out = chain.process(sig)
        assert len(chain) == 2
        assert len(out) == len(sig)

    def test_chain_clear(self):
        chain = EffectsChain()
        chain.add(Reverb())
        chain.clear()
        assert len(chain) == 0
