"""Tests for music_data.open_sources: dataset parsers, MultiSourceIngester, format detection."""
import pytest
import json


class TestKnownDatasets:
    def test_registry_populated(self):
        from models.music_data.open_sources import KNOWN_DATASETS
        assert len(KNOWN_DATASETS) >= 6
        assert "million_song" in KNOWN_DATASETS
        assert "musicbrainz" in KNOWN_DATASETS
        assert "fma" in KNOWN_DATASETS

    def test_dataset_meta_fields(self):
        from models.music_data.open_sources import KNOWN_DATASETS
        for key, ds in KNOWN_DATASETS.items():
            assert ds.name
            assert ds.licence
            assert ds.source_url


class TestMSDParser:
    def test_parse_msd_record(self):
        from models.music_data.open_sources import parse_msd_record
        rec = parse_msd_record({
            "track_id": "TRAAAAW128F429D538",
            "title": "Test Song",
            "artist_name": "Test Artist",
            "year": 2010,
            "duration": 180.5,
            "tempo": 120.0,
            "key": 0,
            "mode": 1,
            "loudness": -5.0,
        })
        assert rec.title == "Test Song"
        assert rec.source_bucket == "million_song"
        assert rec.bpm == 120.0
        assert rec.key_str == "C"
        assert rec.mode == "major"

    def test_parse_msd_csv(self):
        from models.music_data.open_sources import parse_msd_csv
        csv_text = "track_id,title,artist_name,year,duration,tempo,key,mode,loudness,song_hotttnesss\n"
        csv_text += "TR001,Song1,Artist1,2005,200.0,130.0,5,1,-8.0,0.6\n"
        csv_text += "TR002,Song2,Artist2,2012,250.0,90.0,7,0,-12.0,0.3\n"
        records = parse_msd_csv(csv_text)
        assert len(records) == 2
        assert records[0].title == "Song1"

    def test_key_mapping(self):
        from models.music_data.open_sources import parse_msd_record
        for idx, name in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]):
            rec = parse_msd_record({"track_id": f"t{idx}", "key": idx, "mode": 1})
            assert rec.key_str == name


class TestMusicBrainzParser:
    def test_parse_recording(self):
        from models.music_data.open_sources import parse_musicbrainz_recording
        rec = parse_musicbrainz_recording({
            "id": "12345678-1234-1234-1234-123456789012",
            "title": "MB Song",
            "artist-credit": [{"name": "MB Artist"}],
            "length": 240000,
            "tag-list": [{"name": "rock"}, {"name": "alternative"}],
        })
        assert rec.title == "MB Song"
        assert rec.source_bucket == "musicbrainz"
        assert rec.duration_s == 240.0

    def test_parse_json(self):
        from models.music_data.open_sources import parse_musicbrainz_json
        data = {
            "recordings": [
                {
                    "id": "aaaa-bbbb",
                    "title": "J Song",
                    "artist-credit": [{"name": "J Artist"}],
                    "length": 180000,
                    "tag-list": [{"name": "jazz"}],
                }
            ]
        }
        records = parse_musicbrainz_json(json.dumps(data))
        assert len(records) == 1
        assert records[0].title == "J Song"


class TestFMAParser:
    def test_parse_fma_record(self):
        from models.music_data.open_sources import parse_fma_record
        rec = parse_fma_record({
            "track_id": "12345",
            "track_title": "FMA Track",
            "artist_name": "FMA Artist",
            "track_genre_top": "Electronic",
            "track_duration": 300.0,
        })
        assert rec.source_bucket == "fma"
        assert rec.title == "FMA Track"

    def test_parse_fma_csv(self):
        from models.music_data.open_sources import parse_fma_csv
        csv_text = "track_id,track_title,artist_name,track_genre_top,track_duration\n"
        csv_text += "1,Track1,Art1,Pop,200.0\n"
        records = parse_fma_csv(csv_text)
        assert len(records) == 1
        assert records[0].genre == "Pop"


class TestAcousticBrainzParser:
    def test_parse_features(self):
        from models.music_data.open_sources import parse_acousticbrainz_features
        rec = parse_acousticbrainz_features({
            "lowlevel": {
                "spectral_centroid": {"mean": 2000.0},
                "average_loudness": -10.0,
            },
            "rhythm": {"bpm": 120.0, "danceability": 0.7},
            "tonal": {"key_edma": {"key": "C", "scale": "major", "strength": 0.9}},
        })
        assert rec.source_bucket == "acousticbrainz"
        assert rec.bpm == 120.0
        assert rec.key_str == "C"


class TestJamendoParser:
    def test_parse_record(self):
        from models.music_data.open_sources import parse_jamendo_record
        rec = parse_jamendo_record({
            "id": "jam_001",
            "name": "Jam Track",
            "artist_name": "Jam Artist",
            "duration": 200.0,
            "tags": ["chill", "lofi"],
        })
        assert rec.source_bucket == "jamendo"
        assert rec.title == "Jam Track"


class TestMIDIParser:
    def test_parse_metadata(self):
        from models.music_data.open_sources import parse_midi_metadata
        meta = parse_midi_metadata({
            "filename": "/data/midi/test.mid",
            "tempo_bpm": 140.0,
            "time_signature": [3, 4],
            "key": "D major",
            "n_tracks": 4,
            "instruments": [0, 24, 33, 48],
            "duration": 180.0,
        })
        assert meta.tempo_bpm == 140.0
        assert meta.time_signature == "3/4"
        assert len(meta.instruments) == 4
        assert meta.instruments[0] == "Acoustic Grand Piano"

    def test_to_track_record(self):
        from models.music_data.open_sources import parse_midi_metadata
        meta = parse_midi_metadata({"filename": "test.mid", "tempo_bpm": 120.0})
        rec = meta.to_track_record()
        assert rec.bpm == 120.0
        assert rec.source_bucket == "lakh_midi"


class TestMultiSourceIngester:
    def test_ingest_json_dedup(self):
        from models.music_data.open_sources import MultiSourceIngester
        ing = MultiSourceIngester()
        data = [
            {"song_id": "t1", "title": "Song A", "artist_name": "Art", "tempo": 120, "key": 0, "loudness": -5},
            {"song_id": "t1", "title": "Song A", "artist_name": "Art", "tempo": 120, "key": 0, "loudness": -5},
            {"song_id": "t2", "title": "Song B", "artist_name": "Art", "tempo": 90, "key": 3, "loudness": -8},
        ]
        added = ing.ingest_json(json.dumps(data), source="msd")
        assert added == 2
        assert ing.count == 2

    def test_statistics(self):
        from models.music_data.open_sources import MultiSourceIngester
        ing = MultiSourceIngester()
        data1 = [{"song_id": "a", "title": "S1", "tempo": 120, "key": 0, "loudness": -5}]
        ing.ingest_json(json.dumps(data1), source="msd")
        ing.ingest_csv("track_id,track_title,artist_name,track_genre_top\nb,S2,A,Rock\n", source="fma")
        stats = ing.statistics()
        assert stats["total"] >= 1


class TestFormatDetection:
    def test_detect_msd(self):
        from models.music_data.open_sources import detect_source_format
        assert detect_source_format({"song_id": "x", "tempo": 120, "key": 5, "loudness": -8}) == "msd"

    def test_detect_acousticbrainz(self):
        from models.music_data.open_sources import detect_source_format
        assert detect_source_format({"lowlevel": {}, "rhythm": {}, "tonal": {}}) == "acousticbrainz"

    def test_detect_musicbrainz(self):
        from models.music_data.open_sources import detect_source_format
        assert detect_source_format({"artist-credit": [], "title": "X"}) == "musicbrainz"

    def test_detect_unknown(self):
        from models.music_data.open_sources import detect_source_format
        assert detect_source_format({"random": "data"}) == "unknown"
