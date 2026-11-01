"""Music metadata: tag schemas, MusicBrainz, ISRC/ISWC, auto-tagging."""

from .tag_schema import (
    TagStandard, TagField, ID3_FIELDS, VORBIS_FIELDS,
    normalize_field, validate_tag,
)
from .musicbrainz_schema import (
    MBID, MBEntity, MBRelationType, MBArtistCredit,
    MBRecording, MBRelease, MBWork, MBRelation,
    entity_url, parse_artist_credit, format_artist_credit,
)
from .isrc_iswc import (
    validate_isrc, parse_isrc, validate_iswc, parse_iswc,
    validate_ean, generate_check_digit,
)
from .auto_tagger import (
    AudioFeatureProfile, estimate_genre, estimate_mood,
    estimate_era, suggest_tags, tag_confidence,
)

__all__ = [
    "TagStandard", "TagField", "ID3_FIELDS", "VORBIS_FIELDS",
    "normalize_field", "validate_tag",
    "MBID", "MBEntity", "MBRelationType", "MBArtistCredit",
    "MBRecording", "MBRelease", "MBWork", "MBRelation",
    "entity_url", "parse_artist_credit", "format_artist_credit",
    "validate_isrc", "parse_isrc", "validate_iswc", "parse_iswc",
    "validate_ean", "generate_check_digit",
    "AudioFeatureProfile", "estimate_genre", "estimate_mood",
    "estimate_era", "suggest_tags", "tag_confidence",
    "Namespace", "Triple", "LinkedRecord",
    "build_track_record", "build_artist_record", "resolve_same_as",
    "triple_count", "filter_by_namespace", "merge_records",
    "FingerprintAlgorithm", "FingerprintMeta", "ChromaprintConfig",
    "LandmarkConfig", "FingerprintMatch",
    "fingerprint_similarity", "generate_spectral_hash",
    "landmark_hash", "decode_landmark_hash", "match_fingerprints",
]

from .linked_data import (
    Namespace, Triple, LinkedRecord,
    build_track_record, build_artist_record, resolve_same_as,
    triple_count, filter_by_namespace, merge_records,
)
from .audio_fingerprint_meta import (
    FingerprintAlgorithm, FingerprintMeta, ChromaprintConfig,
    LandmarkConfig, FingerprintMatch,
    fingerprint_similarity, generate_spectral_hash,
    landmark_hash, decode_landmark_hash, match_fingerprints,
)
