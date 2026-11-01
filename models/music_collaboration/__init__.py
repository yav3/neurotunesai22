"""Collaborative music-making: ensemble synchronization, co-composition, call-response, session management."""

from .ensemble_sync import (
    SyncMetric, compute_async, mutual_adaptation, granger_influence,
    entrainment_score, phase_correction_gain, ensemble_cohesion,
)
from .collaborative_composition import (
    ContributionType, Contribution, CompositionGraph,
    attribution_weights, style_consistency,
)
from .call_response import (
    TransformationType, CallResponsePair, classify_response,
    generate_response_constraints, dialogue_score,
)
from .session_state import (
    SessionRole, AudioQuality, Participant, SessionState,
    compute_latency_offset, sync_point, role_assignment,
    bandwidth_estimate, max_participants_for_bandwidth, jitter_buffer_size,
)

__all__ = [
    "SyncMetric", "compute_async", "mutual_adaptation", "granger_influence",
    "entrainment_score", "phase_correction_gain", "ensemble_cohesion",
    "ContributionType", "Contribution", "CompositionGraph",
    "attribution_weights", "style_consistency",
    "TransformationType", "CallResponsePair", "classify_response",
    "generate_response_constraints", "dialogue_score",
    "SessionRole", "AudioQuality", "Participant", "SessionState",
    "compute_latency_offset", "sync_point", "role_assignment",
    "bandwidth_estimate", "max_participants_for_bandwidth", "jitter_buffer_size",
]
