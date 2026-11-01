"""Problem specification — typed, structured input to any solver.

Every problem has a type, parameters, and a query. The type selects
which solver runs; the parameters configure it; the query specifies
what to compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ProblemType(Enum):
    # Pharmacokinetics
    PK_CONCENTRATION = "pk_concentration"
    PK_STEADY_STATE = "pk_steady_state"
    PK_MULTI_DOSE = "pk_multi_dose"

    # Pharmacodynamics
    PD_EFFECT = "pd_effect"
    PD_BIOMARKER = "pd_biomarker"
    PD_INVERSE = "pd_inverse"
    PKPD_TRAJECTORY = "pkpd_trajectory"

    # Disease progression
    PROGRESSION_LINEAR = "progression_linear"
    PROGRESSION_EPISODIC = "progression_episodic"
    PROGRESSION_EVENT = "progression_event"
    MCI_CONVERSION = "mci_conversion"

    # GRN dynamics
    GRN_SIMULATE = "grn_simulate"
    GRN_IDENTIFY = "grn_identify_critical"

    # VADE geometry
    VADE_PROJECT = "vade_project"
    VADE_GEODESIC = "vade_geodesic"
    VADE_TRAJECTORY = "vade_trajectory"
    VADE_SCHEDULE = "vade_fibonacci_schedule"

    # Spectral
    SPECTRAL_BAND_POWER = "spectral_band_power"
    SPECTRAL_ENTROPY = "spectral_entropy"
    SPECTRAL_THERAPEUTIC = "spectral_therapeutic_score"
    BINAURAL_BEAT = "binaural_beat"

    # Audio analysis
    AUDIO_VAD_ESTIMATE = "audio_vad_estimate"
    AUDIO_MOOD_SCORES = "audio_mood_scores"
    AUDIO_CAMELOT_COMPAT = "audio_camelot_compatibility"

    # Therapeutic session
    THERAPEUTIC_TRACK_SCORE = "therapeutic_track_score"
    THERAPEUTIC_EIF = "therapeutic_eif"
    THERAPEUTIC_SESSION_PLAN = "therapeutic_session_plan"
    THERAPEUTIC_PHASE_SCORE = "therapeutic_phase_score"

    # EIF (Emotional Impact Factor)
    EIF_FROM_PARAMETERS = "eif_from_parameters"
    EIF_SPREADSHEET = "eif_spreadsheet"
    EIF_RECOMMEND = "eif_recommend"

    # Neural entrainment
    ENTRAINMENT_PLV = "entrainment_plv"
    ENTRAINMENT_ITC = "entrainment_itc"
    ENTRAINMENT_FFR = "entrainment_ffr"
    ENTRAINMENT_BEAT = "entrainment_beat_alignment"

    # CNS pharmacodynamics
    CNS_RTMS = "cns_rtms_modulation"
    CNS_SYNC_DISTANCE = "cns_sync_distance"
    CNS_AFFECT_PURITY = "cns_affect_purity"

    # RTP kernel (patent claims)
    RTP_DOMINANCE = "rtp_dominance"
    RTP_EVOLUTION = "rtp_evolution"
    RTP_SIMULATE = "rtp_simulate"
    RTP_IDENTIFY = "rtp_identify"

    # VADE Drift Estimator
    DRIFT_MONITOR = "drift_monitor"

    # GPCR bias
    GPCR_BIAS = "gpcr_bias"

    # GRN canonical
    GRN_PROPAGATE = "grn_propagate"
    GRN_DOMINANCE = "grn_dominance"
    GRN_SENSITIVITY = "grn_sensitivity"
    GRN_MOTIFS = "grn_motifs"

    # Structural PK/PD
    PKPD_STRUCTURAL = "pkpd_structural"
    PKPD_TOXICITY = "pkpd_toxicity"

    # Composite
    TREATMENT_PLAN = "treatment_plan"
    DISEASE_FORECAST = "disease_forecast"


@dataclass
class Problem:
    """A mathematical problem to be solved.

    Usage:
        p = Problem(
            type=ProblemType.PK_CONCENTRATION,
            params={"drug": "donepezil", "weight_kg": 70},
            query={"time_hr": 24.0},
        )
        answer = solver.solve(p)
    """
    type: ProblemType
    params: dict[str, Any] = field(default_factory=dict)
    query: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] | None = None

    def encode(self) -> np.ndarray:
        """Encode the problem as a numerical vector for the neural solver.
        Type → one-hot, params/query → concatenated values.
        """
        n_types = len(ProblemType)
        type_vec = np.zeros(n_types)
        type_vec[list(ProblemType).index(self.type)] = 1.0

        param_vals = []
        for v in self.params.values():
            if isinstance(v, (int, float)):
                param_vals.append(float(v))
            elif isinstance(v, np.ndarray):
                param_vals.extend(v.ravel().tolist()[:16])

        query_vals = []
        for v in self.query.values():
            if isinstance(v, (int, float)):
                query_vals.append(float(v))

        # Pad to fixed length
        param_vals = (param_vals + [0.0] * 32)[:32]
        query_vals = (query_vals + [0.0] * 16)[:16]

        return np.concatenate([type_vec, param_vals, query_vals]).astype(np.float32)

    @staticmethod
    def encoding_dim() -> int:
        return len(ProblemType) + 32 + 16


@dataclass
class Answer:
    """Structured answer from a solver."""
    values: dict[str, float | np.ndarray]
    problem_type: ProblemType
    exact: bool = True    # True if from math_core, False if from neural approx
    confidence: float = 1.0

    def primary_value(self) -> float | np.ndarray:
        """Return the main numerical answer."""
        if len(self.values) == 1:
            return next(iter(self.values.values()))
        for key in ("concentration", "effect", "biomarker", "distance",
                     "score", "value", "result"):
            if key in self.values:
                return self.values[key]
        return next(iter(self.values.values()))
