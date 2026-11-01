"""Music industry metadata standards, rights, and catalog management."""

from .isrc import ISRC, parse_isrc, format_isrc, validate_isrc, generate_isrc
from .iswc import ISWC, parse_iswc, validate_iswc, compute_check_digit
from .metadata import (
    TrackMetadata, validate_metadata, metadata_completeness,
    normalize_artist_name, detect_metadata_conflicts,
)
from .distribution import (
    DistributionChannel, RoyaltyModel, estimate_royalties,
    channel_revenue_split, breakeven_streams,
)
from .catalog import (
    CatalogEntry, catalog_summary, duplicate_detection,
    catalog_valuation_heuristic,
)

__all__ = [
    "ISRC", "parse_isrc", "format_isrc", "validate_isrc", "generate_isrc",
    "ISWC", "parse_iswc", "validate_iswc", "compute_check_digit",
    "TrackMetadata", "validate_metadata", "metadata_completeness",
    "normalize_artist_name", "detect_metadata_conflicts",
    "DistributionChannel", "RoyaltyModel", "estimate_royalties",
    "channel_revenue_split", "breakeven_streams",
    "CatalogEntry", "catalog_summary", "duplicate_detection",
    "catalog_valuation_heuristic",
]
