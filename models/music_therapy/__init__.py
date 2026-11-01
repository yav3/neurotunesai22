"""Music therapy techniques module — musical (non-clinical) techniques."""

from .iso_principle import (
    IsoTrajectory,
    iso_match_tempo,
    iso_match_mode,
    emotional_trajectory,
    iso_energy_curve,
)
from .entrainment import (
    EntropicCoupling,
    phase_coupling_strength,
    entrainment_time,
    pulse_synchronization,
    kuramoto_order_parameter,
)
from .receptive_methods import (
    ListeningSession,
    relaxation_playlist_score,
    guided_imagery_tempo_profile,
    song_discussion_features,
    session_arc_quality,
)
from .group_dynamics import (
    GroupSession,
    rhythmic_cohesion,
    turn_taking_balance,
    collective_tempo_drift,
    synchronization_index,
)
from .session_protocol import (
    TherapyGoal,
    SessionPhase,
    SessionProtocol,
    InterventionType,
    Intervention,
    build_protocol,
    arousal_regulation_protocol,
    emotional_processing_protocol,
)
from .music_selection import (
    TherapeuticIntent, TrackProfile, TherapeuticPlaylist,
    intent_fitness, rank_tracks, build_therapeutic_playlist,
    iso_principle_sequence, session_energy_profile,
)

__all__ = [
    "IsoTrajectory",
    "iso_match_tempo",
    "iso_match_mode",
    "emotional_trajectory",
    "iso_energy_curve",
    "EntropicCoupling",
    "phase_coupling_strength",
    "entrainment_time",
    "pulse_synchronization",
    "kuramoto_order_parameter",
    "ListeningSession",
    "relaxation_playlist_score",
    "guided_imagery_tempo_profile",
    "song_discussion_features",
    "session_arc_quality",
    "GroupSession",
    "rhythmic_cohesion",
    "turn_taking_balance",
    "collective_tempo_drift",
    "synchronization_index",
    "TherapyGoal",
    "SessionPhase",
    "SessionProtocol",
    "InterventionType",
    "Intervention",
    "build_protocol",
    "arousal_regulation_protocol",
    "emotional_processing_protocol",
    "GIMPhase", "ProgramPiece", "GIMProgram", "BONNY_PROGRAMS",
    "ImageryScorecard", "program_arc", "match_client_readiness", "SessionPlan",
    "TherapeuticIntent", "TrackProfile", "TherapeuticPlaylist",
    "intent_fitness", "rank_tracks", "build_therapeutic_playlist",
    "iso_principle_sequence", "session_energy_profile",
    "WellnessGoal", "WellnessTrack", "WellnessSession",
    "goal_fitness", "rank_wellness_tracks", "build_wellness_session", "session_adherence_score",
    "RhythmPattern", "RhythmicProfile",
    "generate_entrainment_tempo_curve", "heartbeat_synchronization_score",
    "breathing_rhythm_profile", "DrumCircleConfig", "drum_circle_cohesion",
    "movement_tempo_matching", "rhythmic_complexity_index", "adaptive_tempo_recommendation",
    "BowlType", "BrainwaveState", "BRAINWAVE_RANGES", "CHAKRA_FREQUENCIES",
    "SingingBowl", "BinauralBeatConfig", "DroneLayer",
    "SoundBathPhase", "SoundBathSession",
    "binaural_beat_for_state", "isochronic_tone_frequency",
    "bowl_harmony_score", "drone_layer_from_fundamental",
    "design_relaxation_session", "session_frequency_map",
    "GoalDomain", "ProgressLevel", "TherapeuticGoal", "SessionObservation", "OutcomeMeasure",
    "ta_engagement_score", "session_quality_index", "compute_effect_size", "interpret_effect_size",
    "progress_trajectory", "trend_analysis", "sessions_to_target",
    "compute_outcome", "aggregate_progress",
    "reliable_change_index", "clinically_significant",
    "SessionReport", "generate_session_report",
    "EMEntrainmentState", "EntrainmentSession", "GaitTrainingProtocol",
    "kuramoto_oscillator", "em_plv",
    "auditory_motor_coupling", "em_entrainment_strength",
    "isochronous_sequence", "gradual_tempo_shift", "tempo_adaptation_sequence",
    "gait_training_protocol", "bilateral_entrainment_ratio",
]

from .sound_bath import (
    BowlType, BrainwaveState, BRAINWAVE_RANGES, CHAKRA_FREQUENCIES,
    SingingBowl, BinauralBeatConfig, DroneLayer,
    SoundBathPhase, SoundBathSession,
    binaural_beat_for_state, isochronic_tone_frequency,
    bowl_harmony_score, drone_layer_from_fundamental,
    design_relaxation_session, session_frequency_map,
)
from .guided_imagery import (
    GIMPhase, ProgramPiece, GIMProgram, BONNY_PROGRAMS,
    ImageryScorecard, program_arc, match_client_readiness, SessionPlan,
)
from .wellness_protocols import (
    WellnessGoal, WellnessTrack, WellnessSession,
    goal_fitness, rank_wellness_tracks, build_wellness_session, session_adherence_score,
)
from .rhythm_intervention import (
    RhythmPattern, RhythmicProfile,
    generate_entrainment_tempo_curve, heartbeat_synchronization_score,
    breathing_rhythm_profile, DrumCircleConfig, drum_circle_cohesion,
    movement_tempo_matching, rhythmic_complexity_index, adaptive_tempo_recommendation,
)
from .rhythmic_entrainment import (
    EntrainmentState, kuramoto_phase_coupling, phase_locking_value,
    adaptive_tempo_prescription, rhythmic_complexity_for_therapy,
    entrainment_trajectory, iso_principle_trajectory,
)
from .dose_response import (
    TherapyDose, DoseResponseCurve, fit_dose_response,
    therapeutic_threshold, optimal_dose, dose_efficiency,
    diminishing_returns_point, session_schedule, cumulative_response,
)
from .entrainment_models import (
    EntrainmentState as EMEntrainmentState, EntrainmentSession, GaitTrainingProtocol,
    kuramoto_oscillator, phase_locking_value as em_plv,
    auditory_motor_coupling, entrainment_strength as em_entrainment_strength,
    isochronous_sequence, gradual_tempo_shift, tempo_adaptation_sequence,
    gait_training_protocol, bilateral_entrainment_ratio,
)
from .therapeutic_assessment import (
    GoalDomain, ProgressLevel, TherapeuticGoal, SessionObservation, OutcomeMeasure,
    engagement_score as ta_engagement_score,
    session_quality_index, compute_effect_size, interpret_effect_size,
    progress_trajectory, trend_analysis, sessions_to_target,
    compute_outcome, aggregate_progress,
    reliable_change_index, clinically_significant,
    SessionReport, generate_session_report,
)
