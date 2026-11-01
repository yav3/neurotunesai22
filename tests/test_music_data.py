"""Tests for the music_data module: schema, catalog, loaders."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from models.music_data.schema import (
    TrackRecord,
    SpectralAnalysis,
    VADPoetryRecord,
    PlaylistRecord,
    TrackSchema,
)
from models.music_data.catalog import MusicCatalog, CatalogStats
from models.music_data.loaders import CSVTrackLoader, JSONTrackLoader


# ---------------------------------------------------------------------------
# TrackRecord
# ---------------------------------------------------------------------------

class TestTrackRecord:
    def test_defaults(self):
        t = TrackRecord()
        assert t.bpm == 0.0
        assert t.sr == 22050
        assert t.loudness_db == -14.0
        assert t.time_signature == "4/4"
        assert t.emotion_tags == []

    def test_to_feature_dict(self):
        t = TrackRecord(bpm=120.0, valence=0.8, energy=0.7)
        fd = t.to_feature_dict()
        assert fd["bpm"] == 120.0
        assert fd["valence"] == 0.8
        assert fd["energy"] == 0.7
        assert len(fd) == 16

    def test_to_transition_dict(self):
        t = TrackRecord(key_str="C", mode="major", bpm=128, energy=0.6)
        td = t.to_transition_dict()
        assert td["key"] == "C major"
        assert td["bpm"] == 128
        assert td["energy"] == 0.6

    def test_to_transition_dict_no_mode(self):
        t = TrackRecord(key_str="Am")
        td = t.to_transition_dict()
        assert td["key"] == "Am"

    def test_from_dict(self):
        d = {"id": "t1", "bpm": 140.0, "valence": 0.9, "unknown_field": "ignored"}
        t = TrackRecord.from_dict(d)
        assert t.id == "t1"
        assert t.bpm == 140.0
        assert t.valence == 0.9

    def test_from_dict_none_values(self):
        d = {"id": "t2", "bpm": None}
        t = TrackRecord.from_dict(d)
        assert t.id == "t2"
        assert t.bpm == 0.0  # None filtered out, default used


class TestTrackSchema:
    def test_feature_count(self):
        assert TrackSchema.feature_count() > 10

    def test_field_groups(self):
        groups = TrackSchema.field_groups()
        assert "identity" in groups
        assert "musical" in groups
        assert "emotional" in groups
        assert "spectral" in groups

    def test_all_numeric_features(self):
        for f in TrackSchema.ALL_NUMERIC_FEATURES:
            assert isinstance(f, str)


class TestSpectralAnalysis:
    def test_defaults(self):
        sa = SpectralAnalysis(track_id="t1")
        assert sa.spectral_entropy == 0.0
        assert sa.chroma_vector == []


class TestVADPoetryRecord:
    def test_defaults(self):
        r = VADPoetryRecord(id="p1", full_text="roses are red")
        assert r.language == "en"
        assert r.vad_valence_mean == 0.0


class TestPlaylistRecord:
    def test_defaults(self):
        p = PlaylistRecord(id="pl1", name="Chill")
        assert p.intensity == 0.5
        assert p.track_ids == []


# ---------------------------------------------------------------------------
# MusicCatalog
# ---------------------------------------------------------------------------

class TestMusicCatalog:
    def _make_tracks(self, n=5):
        return [
            TrackRecord(
                id=f"t{i}",
                title=f"Track {i}",
                artist=f"Artist {'AB'[i % 2]}",
                genre="Electronic" if i % 2 == 0 else "Pop",
                bpm=120.0 + i * 5,
                valence=0.3 + i * 0.1,
                arousal=0.4 + i * 0.05,
                energy=0.5 + i * 0.08,
                duration_s=180.0 + i * 30,
                key_str=["C", "G", "D", "A", "E", "F", "Bb", "Eb", "Ab", "Db"][i % 10],
                has_lyrics=i > 2,
                lyric_language="en" if i > 2 else "",
            )
            for i in range(n)
        ]

    def test_add_and_len(self):
        cat = MusicCatalog()
        cat.add(TrackRecord(id="t1"))
        assert len(cat) == 1

    def test_add_many(self):
        cat = MusicCatalog()
        n = cat.add_many(self._make_tracks())
        assert n == 5
        assert len(cat) == 5

    def test_add_no_key_raises(self):
        cat = MusicCatalog()
        with pytest.raises(ValueError):
            cat.add(TrackRecord())

    def test_get(self):
        cat = MusicCatalog()
        cat.add(TrackRecord(id="t1", title="Hello"))
        assert cat.get("t1").title == "Hello"
        assert cat.get("missing") is None

    def test_all(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks(3))
        assert len(cat.all()) == 3

    def test_search_by_genre(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks())
        results = cat.search(genre="Electronic")
        assert all("electronic" in r.genre.lower() for r in results)

    def test_search_by_bpm_range(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks())
        results = cat.search(bpm_range=(125, 140))
        assert all(125 <= r.bpm <= 140 for r in results)

    def test_search_by_artist(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks())
        results = cat.search(artist="Artist A")
        assert len(results) > 0

    def test_search_by_key(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks())
        results = cat.search(key="C")
        assert all(r.key_str.lower() == "c" for r in results)

    def test_search_limit(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks(20))
        results = cat.search(limit=3)
        assert len(results) <= 3

    def test_search_has_lyrics(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks())
        results = cat.search(has_lyrics=True)
        assert all(r.has_lyrics for r in results)

    def test_nearest_by_vad(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks())
        nearest = cat.nearest_by_vad(0.5, 0.5, 0.5, n=3)
        assert len(nearest) == 3
        dists = [d for _, d in nearest]
        assert dists == sorted(dists)

    def test_stats(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks())
        s = cat.stats()
        assert s.total_tracks == 5
        assert s.total_duration_hours > 0
        assert s.artists == 2
        assert s.bpm_mean > 0
        assert len(s.genre_distribution) == 2

    def test_stats_empty(self):
        cat = MusicCatalog()
        s = cat.stats()
        assert s.total_tracks == 0

    def test_to_feature_matrix(self):
        cat = MusicCatalog()
        cat.add_many(self._make_tracks())
        matrix, ids, feat_names = cat.to_feature_matrix()
        assert matrix.shape == (5, 16)
        assert len(ids) == 5
        assert len(feat_names) == 16

    def test_to_feature_matrix_empty(self):
        cat = MusicCatalog()
        matrix, ids, feat_names = cat.to_feature_matrix()
        assert matrix.shape == (0, 0)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class TestCSVTrackLoader:
    def test_loads_basic(self):
        csv_text = "id,title,bpm,valence\nt1,Song A,128,0.7\nt2,Song B,140,0.3\n"
        loader = CSVTrackLoader()
        tracks = loader.loads(csv_text)
        assert len(tracks) == 2
        assert tracks[0].id == "t1"
        assert tracks[0].bpm == 128.0
        assert tracks[1].valence == 0.3

    def test_loads_missing_fields(self):
        csv_text = "id,title\nt1,Song A\n"
        loader = CSVTrackLoader()
        tracks = loader.loads(csv_text)
        assert tracks[0].bpm == 0.0

    def test_load_file(self, tmp_path):
        p = tmp_path / "tracks.csv"
        p.write_text("id,title,bpm\nt1,Song,120\n")
        loader = CSVTrackLoader()
        tracks = loader.load(p)
        assert len(tracks) == 1
        assert tracks[0].bpm == 120.0

    def test_loads_safe_float(self):
        csv_text = "id,bpm,valence\nt1,not_a_number,\n"
        loader = CSVTrackLoader()
        tracks = loader.loads(csv_text)
        assert tracks[0].bpm == 0.0
        assert tracks[0].valence == 0.0


class TestJSONTrackLoader:
    def test_loads_list(self):
        data = [{"id": "t1", "bpm": 128}, {"id": "t2", "bpm": 140}]
        loader = JSONTrackLoader()
        tracks = loader.loads(json.dumps(data))
        assert len(tracks) == 2

    def test_loads_dict_with_tracks_key(self):
        data = {"tracks": [{"id": "t1", "bpm": 128}]}
        loader = JSONTrackLoader()
        tracks = loader.loads(json.dumps(data))
        assert len(tracks) == 1

    def test_loads_empty_dict(self):
        loader = JSONTrackLoader()
        tracks = loader.loads("{}")
        assert tracks == []

    def test_load_file(self, tmp_path):
        p = tmp_path / "tracks.json"
        p.write_text(json.dumps([{"id": "t1", "bpm": 128}]))
        loader = JSONTrackLoader()
        tracks = loader.load(p)
        assert len(tracks) == 1

    def test_emotion_tags_list(self):
        data = [{"id": "t1", "emotion_tags": ["happy", "energetic"]}]
        loader = JSONTrackLoader()
        tracks = loader.loads(json.dumps(data))
        assert tracks[0].emotion_tags == ["happy", "energetic"]

    def test_emotion_tags_csv_string(self):
        data = [{"id": "t1", "emotion_tags": "happy, sad"}]
        loader = JSONTrackLoader()
        tracks = loader.loads(json.dumps(data))
        assert tracks[0].emotion_tags == ["happy", "sad"]

    def test_mood_scores_dict(self):
        data = [{"id": "t1", "mood_scores": {"happy": 0.8, "calm": 0.3}}]
        loader = JSONTrackLoader()
        tracks = loader.loads(json.dumps(data))
        assert tracks[0].mood_scores["happy"] == 0.8

    def test_duration_alias(self):
        data = [{"id": "t1", "duration": 240.5}]
        loader = JSONTrackLoader()
        tracks = loader.loads(json.dumps(data))
        assert tracks[0].duration_s == 240.5

    def test_tempo_bpm_alias(self):
        data = [{"id": "t1", "tempo_bpm": 132}]
        loader = JSONTrackLoader()
        tracks = loader.loads(json.dumps(data))
        assert tracks[0].bpm == 132.0
