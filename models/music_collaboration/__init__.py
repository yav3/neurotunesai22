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
from .version_control import (
    DiffType, NoteEvent as VCNoteEvent, MusicalDiff, MergeConflict, MusicalRevision,
    compute_musical_diff, apply_diffs, merge_revisions, revision_distance, make_revision,
)
from .role_assignment import (
    CollaboratorRole, SkillProfile,
    role_compatibility, assign_roles, suggest_voice_ranges,
    workload_balance, ensemble_diversity,
)
from .conflict_resolution import (
    EditOp, regions_overlap, detect_conflicts, resolve_last_writer_wins,
    merge_non_overlapping, full_resolution_pipeline,
)
from .skill_matching import (
    MusicianProfile, EnsembleRequirement, genre_overlap, role_fitness,
    greedy_assign, ensemble_compatibility, skill_gap_analysis,
)

from .communication_protocol import (
    MessageType, VectorClock, Message, GCounter, PNCounter,
    LWWRegister, ORSet, SessionBus,
    latency_compensated_timestamp, merge_concurrent_edits,
    message_integrity_check,
)

__all__ = [
    "MessageType", "VectorClock", "Message", "GCounter", "PNCounter",
    "LWWRegister", "ORSet", "SessionBus",
    "latency_compensated_timestamp", "merge_concurrent_edits",
    "message_integrity_check",
    "SyncMetric", "compute_async", "mutual_adaptation", "granger_influence",
    "entrainment_score", "phase_correction_gain", "ensemble_cohesion",
    "ContributionType", "Contribution", "CompositionGraph",
    "attribution_weights", "style_consistency",
    "TransformationType", "CallResponsePair", "classify_response",
    "generate_response_constraints", "dialogue_score",
    "SessionRole", "AudioQuality", "Participant", "SessionState",
    "compute_latency_offset", "sync_point", "role_assignment",
    "bandwidth_estimate", "max_participants_for_bandwidth", "jitter_buffer_size",
    "DiffType", "VCNoteEvent", "MusicalDiff", "MergeConflict", "MusicalRevision",
    "compute_musical_diff", "apply_diffs", "merge_revisions", "revision_distance", "make_revision",
    "CollaboratorRole", "SkillProfile",
    "role_compatibility", "assign_roles", "suggest_voice_ranges",
    "workload_balance", "ensemble_diversity",
    "EditOp", "regions_overlap", "detect_conflicts", "resolve_last_writer_wins",
    "merge_non_overlapping", "full_resolution_pipeline",
    "MusicianProfile", "EnsembleRequirement", "genre_overlap", "role_fitness",
    "greedy_assign", "ensemble_compatibility", "skill_gap_analysis",
    "RehearsalFocus", "SectionDifficulty", "RehearsalPlan", "ProgressRecord",
    "estimate_section_difficulty", "plan_rehearsal", "learning_curve",
    "weakest_sections", "tempo_ramp_schedule", "rehearsal_efficiency",
    "EventType", "SessionEvent", "SessionRecording",
    "record_event", "events_in_range", "player_activity",
    "activity_density", "player_contribution", "interaction_matrix",
    "session_fingerprint", "merge_recordings", "tempo_changes", "note_events_only",
    "CodeLanguage", "SyncMode", "CodeSnippet", "PerformerState", "SharedClock", "LiveSession",
    "submit_code", "code_diff", "performer_activity", "session_entropy",
    "quantize_to_bar", "active_performers",
    "NetworkStats", "JitterBuffer", "SyncState",
    "estimate_rtt", "adaptive_buffer_size", "clock_drift_estimate",
    "compensate_drift", "packet_loss_concealment", "sync_score",
    "optimal_monitoring_latency", "jitter_buffer_push", "jitter_buffer_pull",
    "jitter_buffer_level",
    "ReviewCriterion", "ReviewScore", "PeerReview", "ReviewSummary",
    "DEFAULT_CRITERIA",
    "create_review", "aggregate_reviews", "reviewer_reliability",
    "inter_rater_agreement", "weighted_consensus",
    "feedback_quality_score", "review_bias_detection",
    "bayesian_score_estimate", "rank_submissions",
    "calibration_factor", "detect_outlier_reviews",
    "SyncStrategy", "ClockState", "JitterStats", "QuantizeConfig", "JamParticipant",
    "ntp_offset", "ptp_offset", "audio_beacon_offset",
    "optimal_buffer_depth", "jitter_buffer_delay", "jitter_statistics",
    "packet_loss_interpolation", "latency_aware_quantize", "beat_align_to_master",
    "round_trip_estimate", "session_latency_budget", "mix_remote_streams",
    "dj_consensus_tempo", "network_score",
]

from .live_coding import (
    CodeLanguage, SyncMode, CodeSnippet, PerformerState, SharedClock, LiveSession,
    submit_code, code_diff, performer_activity, session_entropy,
    quantize_to_bar, active_performers,
)
from .rehearsal_planning import (
    RehearsalFocus, SectionDifficulty, RehearsalPlan, ProgressRecord,
    estimate_section_difficulty, plan_rehearsal, learning_curve,
    weakest_sections, tempo_ramp_schedule, rehearsal_efficiency,
)
from .session_recording import (
    EventType, SessionEvent, SessionRecording,
    record_event, events_in_range, player_activity,
    activity_density, player_contribution, interaction_matrix,
    session_fingerprint, merge_recordings, tempo_changes, note_events_only,
)
from .ensemble_dynamics import (
    EnsembleSynchrony, compute_onset_synchrony, compute_tempo_agreement,
    entrainment_index, turn_taking_analysis, blend_score, analyze_ensemble,
)
from .score_annotation import (
    AnnotationCategory, AnnotationStatus, MusicAnchor, Annotation, AnnotationSet,
    make_annotation_id, annotation_density, annotation_coverage,
    category_distribution, author_activity, resolution_rate, priority_summary,
    merge_annotation_sets, hotspot_measures, unresolved_critical,
)
from .operational_transform import (
    OpType, NotePayload, Operation, transform as ot_transform,
    apply_op as ot_apply_op, apply_ops as ot_apply_ops,
    compose as ot_compose, invert as ot_invert,
    OTServer, verify_tp1, random_ops, convergence_test,
)
from .peer_review import (
    ReviewCriterion, ReviewScore, PeerReview, ReviewSummary,
    DEFAULT_CRITERIA,
    create_review, aggregate_reviews, reviewer_reliability,
    inter_rater_agreement, weighted_consensus,
    feedback_quality_score, review_bias_detection,
    bayesian_score_estimate, rank_submissions,
    calibration_factor, detect_outlier_reviews,
)
from .latency_compensation import (
    NetworkStats, JitterBuffer, SyncState,
    estimate_rtt, adaptive_buffer_size, clock_drift_estimate,
    compensate_drift, packet_loss_concealment, sync_score,
    optimal_monitoring_latency, jitter_buffer_push, jitter_buffer_pull,
    jitter_buffer_level,
)
from .distributed_jam import (
    SyncStrategy, ClockState, JitterStats, QuantizeConfig, JamParticipant,
    ntp_offset, ptp_offset, audio_beacon_offset,
    optimal_buffer_depth, jitter_buffer_delay, jitter_statistics,
    packet_loss_interpolation, latency_aware_quantize, beat_align_to_master,
    round_trip_estimate, session_latency_budget, mix_remote_streams,
    consensus_tempo as dj_consensus_tempo, network_score,
)
