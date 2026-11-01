"""Product bridge — connects math kernels to the consumer-facing apps.

neurotunes (Flutter): therapeutic music streaming
  - Audio features: BPM, key, energy, valence, arousal, dominance
  - Brainwave entrainment: binaural beats for focus/relaxation/sleep
  - Therapeutic session tracking

neurotunescopy4privatesharing (React/TS): clinical/research product
  - Audio analysis pipeline (Supabase edge functions)
  - Brainwave streaming (SSE endpoint)
  - Data integrity validation

This bridge translates between:
  - NeuroTunesAI solver outputs (tensors) → product feature vectors (JSON)
  - Product audio features → NeuroTunesAI inputs (tensors for EIF, VADE, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AudioFeatures:
    """Audio features matching the product surface schema.

    Field names match neurotunescopy4privatesharing/supabase/functions/
    audio-analysis/index.ts AnalysisResult interface.
    """
    track_id: str
    bpm: float = 0.0
    key: str = ""
    camelot: str = ""
    energy_level: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    danceability: float = 0.0
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    loudness_lufs: float = 0.0
    mood_happy: float = 0.0
    mood_sad: float = 0.0
    mood_relaxed: float = 0.0
    mood_aggressive: float = 0.0

    def to_vad_tensor(self) -> torch.Tensor:
        """Convert to VAD (Valence-Arousal-Dominance) tensor for VADE solver."""
        return torch.tensor([self.valence, self.arousal, self.dominance])

    def to_eif_input(self) -> dict[str, float]:
        """Convert to EIF solver input format."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "energy": self.energy_level,
            "bpm": self.bpm,
            "spectral_centroid": self.spectral_centroid,
        }

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class BrainwaveParams:
    """Brainwave entrainment parameters matching the product's frequency bands.

    Maps to neurotunescopy4privatesharing/supabase/functions/
    brainwave-stream/index.ts FREQUENCY_BANDS and GOAL_FREQUENCIES.
    """
    band: str = "alpha"
    base_freq_hz: float = 10.0
    binaural_beat_hz: float = 3.0
    goal: str = "relaxation"
    duration_sec: float = 30.0

    BAND_RANGES: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
        "gamma": (30.0, 100.0),
    })


class ProductBridge:
    """Translates between NeuroTunesAI solvers and product surfaces.

    Product → Solver direction:
      Audio features → VADE projection input
      Audio features → EIF computation input
      Audio features → Therapeutic session planner input
      Brainwave params → Entrainment solver input

    Solver → Product direction:
      VADE trajectory → Session plan (sequence of audio feature targets)
      EIF scores → Track recommendation ranking
      Therapeutic constraint check → Session feasibility flag
    """

    @staticmethod
    def audio_to_solver_input(features: AudioFeatures, n_biomarkers: int = 96) -> torch.Tensor:
        """Convert product audio features to solver-compatible biomarker tensor.

        Maps the ~15 audio features into the expected biomarker space
        by zero-padding the remaining dimensions.
        """
        core = torch.tensor([
            features.bpm / 200.0,
            features.valence,
            features.arousal,
            features.dominance,
            features.energy_level,
            features.danceability,
            features.spectral_centroid / 8000.0,
            features.spectral_rolloff / 16000.0,
            features.loudness_lufs / -60.0 + 1.0,
            features.mood_happy,
            features.mood_sad,
            features.mood_relaxed,
            features.mood_aggressive,
        ])
        padded = torch.zeros(n_biomarkers)
        padded[:len(core)] = core
        return padded.unsqueeze(0)

    @staticmethod
    def vade_trajectory_to_session_plan(
        trajectory: torch.Tensor,
        doses: torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Convert VADE trajectory to a session plan for the product.

        Each step becomes a target audio feature vector that the
        recommendation engine should match.
        """
        n_steps = trajectory.shape[-1] if trajectory.dim() == 2 else trajectory.shape[0]
        steps = []
        for k in range(n_steps):
            if trajectory.dim() == 2:
                state = trajectory[:, k]
            else:
                state = trajectory[k]
            steps.append({
                "step": k,
                "target_valence": state[0].item() if state.numel() > 0 else 0.0,
                "target_arousal": state[1].item() if state.numel() > 1 else 0.0,
                "target_dominance": state[2].item() if state.numel() > 2 else 0.0,
                "dose_intensity": doses[k].item() if k < doses.numel() else 0.0,
            })
        return steps

    @staticmethod
    def eif_scores_to_ranking(
        track_ids: list[str],
        eif_total: torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Rank tracks by EIF score for product recommendation."""
        scores = eif_total.squeeze().tolist()
        if isinstance(scores, float):
            scores = [scores]
        ranked = sorted(
            zip(track_ids, scores),
            key=lambda x: -x[1],
        )
        return [{"track_id": tid, "eif_score": s, "rank": i + 1} for i, (tid, s) in enumerate(ranked)]

    @staticmethod
    def brainwave_to_entrainment_input(params: BrainwaveParams) -> dict[str, float]:
        """Convert product brainwave params to entrainment solver input."""
        lo, hi = params.BAND_RANGES.get(params.band, (8.0, 12.0))
        return {
            "target_freq_hz": params.base_freq_hz,
            "beat_freq_hz": params.binaural_beat_hz,
            "band_low_hz": lo,
            "band_high_hz": hi,
            "duration_sec": params.duration_sec,
        }

    @staticmethod
    def perturbation_to_product(
        prediction_result: Any,
        trial_id: str = "NT-001",
    ) -> dict[str, Any]:
        """Convert PerturbationInferenceService output to product-consumable format.

        Maps the cVAE's gene-level predictions into the product surface's
        PerturbationResult schema for the analysis pipeline.
        """
        return {
            "trialId": trial_id,
            "perturbationType": prediction_result.record.perturbation_type,
            "specHash": prediction_result.record.model_spec_hash,
            "regime": "data-driven",
            "species": "human",
            "confidence": 1.0 / (1.0 + prediction_result.latent_variance),
            "nGenes": len(prediction_result.gene_names),
            "topUpregulated": prediction_result.top_upregulated[:5],
            "topDownregulated": prediction_result.top_downregulated[:5],
            "predictionId": prediction_result.record.prediction_id,
            "predictionHash": prediction_result.record.prediction_hash,
        }
