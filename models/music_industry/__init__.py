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
from .rights_management import (
    RightType, LicenseType, OwnershipShare, License, RightsBundle,
    validate_splits, detect_conflicts, RoyaltyWaterfall,
    compute_waterfall, license_value_estimate, territory_overlap,
)
from .analytics import (
    Platform, PER_STREAM_RATES, StreamingMetrics,
    engagement_score, estimated_revenue, artist_momentum_score,
    growth_projection, viral_coefficient, catalog_health,
    audience_concentration, listener_retention_curve, breakeven_analysis,
)
from .revenue_models import (
    RevenueStream, StreamingTier, StreamingRate, PLATFORM_RATES,
    streaming_revenue, mechanical_royalty, performance_royalty,
    RevenueProjection, catalog_valuation, breakeven_streams_for_cost, blended_rate,
)
from .market_analysis import (
    genre_market_share, herfindahl_index, genre_growth_rate,
    artist_market_position, trend_momentum, seasonal_index,
    market_saturation,
)
from .royalty_engine import (
    RightsHolder, RoyaltyConfig,
    validate_splits as validate_royalty_splits,
    allocate_royalties,
    streaming_payout, streaming_payout_curve, pro_rata_pool_share,
    mechanical_royalty as mechanical_royalty_statutory,
    performance_royalty_estimate,
    sync_fee_estimate,
    project_revenue, RevenueProjection as RoyaltyProjection,
    streams_to_breakeven, recoupment_schedule,
)
from .release_strategy import (
    ReleaseType, RolloutPhase, ReleaseWindow, STANDARD_ROLLOUT, ReleasePlan,
    optimal_release_day, single_vs_album_roi, playlist_pitch_score, waterfall_schedule,
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
    "RightType", "LicenseType", "OwnershipShare", "License", "RightsBundle",
    "validate_splits", "detect_conflicts", "RoyaltyWaterfall",
    "compute_waterfall", "license_value_estimate", "territory_overlap",
    "Platform", "PER_STREAM_RATES", "StreamingMetrics",
    "engagement_score", "estimated_revenue", "artist_momentum_score",
    "growth_projection", "viral_coefficient", "catalog_health",
    "audience_concentration", "listener_retention_curve", "breakeven_analysis",
    "RevenueStream", "StreamingTier", "StreamingRate", "PLATFORM_RATES",
    "streaming_revenue", "mechanical_royalty", "performance_royalty",
    "RevenueProjection", "catalog_valuation", "breakeven_streams_for_cost", "blended_rate",
    "genre_market_share", "herfindahl_index", "genre_growth_rate",
    "artist_market_position", "trend_momentum", "seasonal_index",
    "market_saturation",
    "ReleaseType", "RolloutPhase", "ReleaseWindow", "STANDARD_ROLLOUT", "ReleasePlan",
    "optimal_release_day", "single_vs_album_roi", "playlist_pitch_score", "waterfall_schedule",
    "RightsHolder", "RoyaltyConfig", "validate_royalty_splits", "allocate_royalties",
    "streaming_payout", "streaming_payout_curve", "pro_rata_pool_share",
    "mechanical_royalty_statutory", "performance_royalty_estimate",
    "sync_fee_estimate", "project_revenue", "RoyaltyProjection",
    "streams_to_breakeven", "recoupment_schedule",
    "ListenerSegment", "ListenerProfile", "AudienceMetrics",
    "classify_listener", "segment_audience", "compute_audience_metrics",
    "churn_risk", "lifetime_value_estimate", "genre_overlap_matrix",
]

from .audience_analytics import (
    ListenerSegment, ListenerProfile, AudienceMetrics,
    classify_listener, segment_audience, compute_audience_metrics,
    churn_risk, lifetime_value_estimate, genre_overlap_matrix,
)
