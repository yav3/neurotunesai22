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
