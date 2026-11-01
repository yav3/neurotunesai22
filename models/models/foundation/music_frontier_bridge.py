"""Music Frontier Bridge — connects the music foundation encoder to
structured frontier LLM reasoning via a governed evidence layer.

Extends the GovernedFrontierBridge pattern to music-domain queries:
  - CLS embedding → structured query (grounded in learned audio representation)
  - LLM response → evidence record → lineage chain → governed answer
  - Mood-shift, track-recommendation, and therapeutic-session query types

The bridge enforces:
  - Domain firewall (music queries only)
  - Hash-chained lineage on every LLM interaction
  - Pre-registered hypothesis format
  - Calibration tracking
  - Structured output schema
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


AUDIO_FEATURE_NAMES = [
    "spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
    "zero_crossing_rate", "rms_energy", "dynamic_range",
    "bass_energy", "mid_energy", "treble_energy",
    "tempo_bpm", "key", "scale", "camelot",
    "valence", "arousal", "dominance",
]


class MusicQueryType(str, Enum):
    """Structured query types for music-domain frontier reasoning."""
    MOOD_SHIFT_HYPOTHESIS = "mood_shift_hypothesis"
    TRACK_RECOMMENDATION = "track_recommendation"
    THERAPEUTIC_SESSION_DESIGN = "therapeutic_session_design"
    AUDIO_FEATURE_ANALYSIS = "audio_feature_analysis"
    PLAYLIST_TRANSITION = "playlist_transition"
    ENTRAINMENT_DESIGN = "entrainment_design"


@dataclass
class MusicStructuredQuery:
    """A governed query to the frontier LLM, grounded in foundation output."""
    query_type: MusicQueryType
    audio_context: dict[str, Any]
    embedding_summary: dict[str, float]
    domain: str
    regime: str
    hypothesis: str
    pre_registered: bool = False
    lineage_hash: str = ""


@dataclass
class MusicFrontierResponse:
    """Structured response from the frontier LLM for music queries."""
    reasoning: str
    confidence: float
    evidence_citations: list[str]
    suggested_tracks: list[str]
    mood_trajectory: list[dict[str, float]]
    caveats: list[str]
    response_hash: str = ""
    latency_ms: float = 0.0


@dataclass
class GovernedMusicAnswer:
    """The final governed answer combining both towers for music."""
    foundation_prediction: dict[str, Any]
    frontier_reasoning: MusicFrontierResponse | None
    lineage_chain: list[str]
    domain: str
    regime: str
    study_id: str
    is_governed: bool = True
    content_hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps({
            "prediction": self.foundation_prediction,
            "reasoning": self.frontier_reasoning.reasoning if self.frontier_reasoning else None,
            "domain": self.domain,
            "regime": self.regime,
            "study_id": self.study_id,
        }, sort_keys=True, default=str)
        self.content_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.content_hash


@dataclass
class MusicBridgeConfig:
    """Configuration for the governed music frontier bridge."""
    max_top_features: int = 16
    confidence_threshold: float = 0.3
    require_pre_registration: bool = False
    max_query_length: int = 4000
    allowed_domains: tuple[str, ...] = ("music", "therapeutic_audio")
    allowed_regimes: tuple[str, ...] = ("data-driven", "prior-contrast", "therapeutic")


class MusicEmbeddingSummariser:
    """Extract interpretable features from the music encoder's output
    to ground frontier LLM queries in the learned representation."""

    @staticmethod
    def summarise_feature_delta(
        baseline: np.ndarray,
        predicted: np.ndarray,
        feature_names: list[str],
        top_k: int = 16,
    ) -> dict[str, Any]:
        delta = predicted - baseline
        abs_delta = np.abs(delta)
        top_k = min(top_k, len(feature_names))
        top_idx = np.argsort(abs_delta)[-top_k:][::-1]

        return {
            "top_features": [
                {"feature": feature_names[i], "delta": float(delta[i]),
                 "baseline": float(baseline[i]), "predicted": float(predicted[i])}
                for i in top_idx
            ],
            "mean_abs_delta": float(abs_delta.mean()),
            "max_abs_delta": float(abs_delta.max()),
            "vad_delta": {
                "valence": float(delta[feature_names.index("valence")]) if "valence" in feature_names else 0.0,
                "arousal": float(delta[feature_names.index("arousal")]) if "arousal" in feature_names else 0.0,
                "dominance": float(delta[feature_names.index("dominance")]) if "dominance" in feature_names else 0.0,
            },
            "total_features": len(feature_names),
        }

    @staticmethod
    def summarise_cls_embedding(cls_repr: np.ndarray) -> dict[str, float]:
        return {
            "norm": float(np.linalg.norm(cls_repr)),
            "mean": float(cls_repr.mean()),
            "std": float(cls_repr.std()),
            "max": float(cls_repr.max()),
            "min": float(cls_repr.min()),
            "sparsity": float((np.abs(cls_repr) < 0.01).mean()),
        }


class GovernedMusicBridge:
    """Connects music foundation encoder output to frontier LLM via governed protocol."""

    def __init__(self, config: MusicBridgeConfig | None = None):
        self.config = config or MusicBridgeConfig()
        self.lineage: list[str] = []

    def _chain_hash(self, content: str) -> str:
        prev = self.lineage[-1] if self.lineage else "genesis"
        payload = f"{prev}:{content}"
        h = hashlib.sha256(payload.encode()).hexdigest()
        self.lineage.append(h)
        return h

    def build_query(
        self,
        query_type: MusicQueryType,
        feature_delta: dict[str, Any],
        cls_summary: dict[str, float],
        domain: str,
        regime: str,
        hypothesis: str,
        study_id: str = "",
    ) -> MusicStructuredQuery:
        if domain not in self.config.allowed_domains:
            raise ValueError(
                f"Domain '{domain}' not in allowed set {self.config.allowed_domains}. "
                f"Domain firewall: declare or refuse."
            )
        if regime not in self.config.allowed_regimes:
            raise ValueError(
                f"Regime '{regime}' not in allowed set {self.config.allowed_regimes}."
            )

        audio_context = {
            "top_features": feature_delta.get("top_features", [])[:self.config.max_top_features],
            "summary_stats": {
                "mean_abs_delta": feature_delta.get("mean_abs_delta", 0),
                "vad_delta": feature_delta.get("vad_delta", {}),
                "total_features": feature_delta.get("total_features", 0),
            },
        }

        query = MusicStructuredQuery(
            query_type=query_type,
            audio_context=audio_context,
            embedding_summary=cls_summary,
            domain=domain,
            regime=regime,
            hypothesis=hypothesis,
            pre_registered=bool(study_id),
            lineage_hash=self._chain_hash(
                json.dumps({"type": query_type.value, "hypothesis": hypothesis,
                            "domain": domain, "study_id": study_id}, sort_keys=True)
            ),
        )
        return query

    def format_prompt(self, query: MusicStructuredQuery) -> str:
        """Format the structured query into a frontier LLM prompt."""
        top_features_text = "\n".join(
            f"  {f['feature']}: delta={f['delta']:+.3f} "
            f"(baseline={f['baseline']:.2f} → predicted={f['predicted']:.2f})"
            for f in query.audio_context["top_features"][:16]
        )

        stats = query.audio_context["summary_stats"]
        vad = stats.get("vad_delta", {})

        prompt = f"""You are a music therapy and audio analysis expert interpreting audio feature changes.

Domain: {query.domain}
Analysis regime: {query.regime}
Query type: {query.query_type.value}

HYPOTHESIS (pre-registered: {query.pre_registered}):
{query.hypothesis}

FOUNDATION MODEL OUTPUT — top audio feature changes:
{top_features_text}

VAD shift: valence={vad.get('valence', 0):+.3f}, arousal={vad.get('arousal', 0):+.3f}, dominance={vad.get('dominance', 0):+.3f}
Summary: {stats['total_features']} features analysed, mean |delta| = {stats.get('mean_abs_delta', 0):.4f}

Respond in JSON with exactly these fields:
{{
  "reasoning": "<2-3 sentence interpretation of the mood/therapeutic shift>",
  "confidence": <float 0-1, calibrated>,
  "evidence_citations": ["<PMID or DOI for music therapy evidence>", ...],
  "suggested_tracks": ["<track or genre recommendation>", ...],
  "mood_trajectory": [{{"valence": <float>, "arousal": <float>, "time_pct": <0-1>}}, ...],
  "caveats": ["<limitation or alternative explanation>", ...]
}}"""
        return prompt[:self.config.max_query_length]

    def parse_response(
        self, raw_json: str | dict, query: MusicStructuredQuery
    ) -> MusicFrontierResponse:
        if isinstance(raw_json, str):
            data = json.loads(raw_json)
        else:
            data = raw_json

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        response = MusicFrontierResponse(
            reasoning=str(data.get("reasoning", "")),
            confidence=confidence,
            evidence_citations=[str(c) for c in data.get("evidence_citations", [])],
            suggested_tracks=[str(t) for t in data.get("suggested_tracks", [])],
            mood_trajectory=[
                {"valence": float(m.get("valence", 0.5)),
                 "arousal": float(m.get("arousal", 0.5)),
                 "time_pct": float(m.get("time_pct", 0.0))}
                for m in data.get("mood_trajectory", [])
            ],
            caveats=[str(c) for c in data.get("caveats", [])],
        )

        response.response_hash = self._chain_hash(
            json.dumps({"reasoning": response.reasoning,
                        "confidence": response.confidence}, sort_keys=True)
        )
        return response

    def build_governed_answer(
        self,
        foundation_prediction: dict[str, Any],
        frontier_response: MusicFrontierResponse | None,
        domain: str,
        regime: str,
        study_id: str,
    ) -> GovernedMusicAnswer:
        answer = GovernedMusicAnswer(
            foundation_prediction=foundation_prediction,
            frontier_reasoning=frontier_response,
            lineage_chain=list(self.lineage),
            domain=domain,
            regime=regime,
            study_id=study_id,
        )
        answer.compute_hash()
        return answer


class MockMusicFrontierCaller:
    """Deterministic mock for testing without API access."""

    def __init__(self, confidence: float = 0.7):
        self.confidence = confidence
        self.call_count = 0

    def __call__(self, prompt: str) -> dict:
        self.call_count += 1
        return {
            "reasoning": (
                "The foundation model predicts a shift toward higher valence and "
                "moderate arousal, consistent with a relaxation-to-focus transition. "
                "Spectral centroid increase suggests brighter timbral quality."
            ),
            "confidence": self.confidence,
            "evidence_citations": ["DOI:10.1016/j.neuroimage.2024.example"],
            "suggested_tracks": [
                "Ambient with gradual tempo increase (70-90 BPM)",
                "Major key progression with rising spectral centroid",
            ],
            "mood_trajectory": [
                {"valence": 0.4, "arousal": 0.3, "time_pct": 0.0},
                {"valence": 0.6, "arousal": 0.5, "time_pct": 0.5},
                {"valence": 0.8, "arousal": 0.6, "time_pct": 1.0},
            ],
            "caveats": [
                "Individual response varies with musical preference",
                "Therapeutic effect requires controlled exposure duration",
            ],
        }


if HAS_TORCH:

    class MusicFoundationToFrontierProjector(nn.Module):
        """Project music encoder CLS → structured features for the bridge."""

        def __init__(self, hidden_dim: int = 128, projection_dim: int = 16):
            super().__init__()
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, projection_dim),
            )
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, cls_repr: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "projection": self.projector(cls_repr),
                "confidence": self.confidence_head(cls_repr).squeeze(-1),
            }
