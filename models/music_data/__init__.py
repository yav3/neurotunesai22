from __future__ import annotations

from models.music_data.schema import (
    TrackRecord,
    SpectralAnalysis,
    VADPoetryRecord,
    PlaylistRecord,
    TrackSchema,
)
from models.music_data.catalog import (
    MusicCatalog,
    CatalogStats,
)
from models.music_data.loaders import (
    CSVTrackLoader,
    JSONTrackLoader,
)
from models.music_data.open_sources import (
    KNOWN_DATASETS,
    DatasetMeta,
    MultiSourceIngester,
    parse_msd_record,
    parse_msd_csv,
    parse_musicbrainz_recording,
    parse_musicbrainz_json,
    parse_fma_record,
    parse_fma_csv,
    parse_acousticbrainz_features,
    parse_acousticbrainz_json,
    parse_jamendo_record,
    parse_jamendo_json,
    parse_midi_metadata,
    MIDIMeta,
    detect_source_format,
    parse_record,
)

__all__ = [
    "TrackRecord",
    "SpectralAnalysis",
    "VADPoetryRecord",
    "PlaylistRecord",
    "TrackSchema",
    "MusicCatalog",
    "CatalogStats",
    "CSVTrackLoader",
    "JSONTrackLoader",
    "KNOWN_DATASETS",
    "DatasetMeta",
    "MultiSourceIngester",
    "parse_msd_record",
    "parse_msd_csv",
    "parse_musicbrainz_recording",
    "parse_musicbrainz_json",
    "parse_fma_record",
    "parse_fma_csv",
    "parse_acousticbrainz_features",
    "parse_acousticbrainz_json",
    "parse_jamendo_record",
    "parse_jamendo_json",
    "parse_midi_metadata",
    "MIDIMeta",
    "detect_source_format",
    "parse_record",
]
