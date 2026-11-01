"""Track schema matching the 86-column Supabase data model from App_Audio_Song_Inventory.

Columns organized by domain:
  - Identity: id, path, title, artist, album, genre, duration, sr
  - Musical: bpm, key_str, camelot, scale, mode, key_confidence, time_signature
  - Emotional/Affective: valence, arousal, dominance, emotion_tags, mood_scores
  - Timbre/Spectral: spectral_centroid, spectral_rolloff, spectral_bandwidth,
    brightness, zero_crossing_rate, rms_energy, dynamic_range, mfcc_coefficients
  - Band energies: bass_energy, mid_energy, treble_energy
  - Dynamics/Structure: loudness_db, dynamic_complexity, attack_density
  - Temporal affect: emotion_arc (per-section valence curve)
  - Lyrics/Notation: has_lyrics, lyric_language, lyric_sentiment
  - Therapeutic: therapeutic_tags, evidence_score
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackRecord:
    id: str = ""
    path: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    duration_s: float = 0.0
    sr: int = 22050

    bpm: float = 0.0
    key_str: str = ""
    camelot: str = ""
    scale: str = ""
    mode: str = ""
    key_confidence: float = 0.0
    time_signature: str = "4/4"

    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    emotion_tags: list[str] = field(default_factory=list)
    mood_scores: dict[str, float] = field(default_factory=dict)

    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_bandwidth: float = 0.0
    brightness: float = 0.0
    zero_crossing_rate: float = 0.0
    rms_energy: float = 0.0
    dynamic_range: float = 0.0
    mfcc_coefficients: list[float] = field(default_factory=list)

    bass_energy: float = 0.0
    mid_energy: float = 0.0
    treble_energy: float = 0.0

    loudness_db: float = -14.0
    dynamic_complexity: float = 0.0
    attack_density: float = 0.0

    emotion_arc: list[float] = field(default_factory=list)

    has_lyrics: bool = False
    lyric_language: str = ""
    lyric_sentiment: float = 0.0

    therapeutic_tags: list[str] = field(default_factory=list)
    evidence_score: float = 0.0

    danceability: float = 0.0
    energy: float = 0.0
    acousticness: float = 0.0
    instrumentalness: float = 0.0
    speechiness: float = 0.0
    liveness: float = 0.0

    tags: list[str] = field(default_factory=list)
    checksum: str = ""
    source_bucket: str = ""

    def to_feature_dict(self) -> dict[str, float]:
        return {
            "bpm": self.bpm,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "spectral_centroid": self.spectral_centroid,
            "spectral_rolloff": self.spectral_rolloff,
            "spectral_bandwidth": self.spectral_bandwidth,
            "brightness": self.brightness,
            "zero_crossing_rate": self.zero_crossing_rate,
            "rms_energy": self.rms_energy,
            "dynamic_range": self.dynamic_range,
            "bass_energy": self.bass_energy,
            "mid_energy": self.mid_energy,
            "treble_energy": self.treble_energy,
            "danceability": self.danceability,
            "energy": self.energy,
        }

    def to_transition_dict(self) -> dict[str, Any]:
        return {
            "key": f"{self.key_str} {self.mode}" if self.mode else self.key_str,
            "bpm": self.bpm,
            "energy": self.energy or self.rms_energy,
            "valence": self.valence,
            "arousal": self.arousal,
            "artist": self.artist,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known and v is not None}
        return cls(**filtered)


@dataclass
class SpectralAnalysis:
    track_id: str = ""
    eeg_band_scores: dict[str, float] = field(default_factory=dict)
    chroma_vector: list[float] = field(default_factory=list)
    mfcc_coefficients: list[float] = field(default_factory=list)
    spectral_entropy: float = 0.0
    harmonic_ratio: float = 0.0
    percussive_ratio: float = 0.0


@dataclass
class VADPoetryRecord:
    id: str = ""
    full_text: str = ""
    per_line_vad: list[dict[str, float]] = field(default_factory=list)
    vad_valence_mean: float = 0.0
    vad_valence_median: float = 0.0
    vad_arousal_mean: float = 0.0
    vad_arousal_median: float = 0.0
    vad_dominance_mean: float = 0.0
    vad_dominance_median: float = 0.0
    language: str = "en"
    source: str = ""


@dataclass
class PlaylistRecord:
    id: str = ""
    name: str = ""
    track_ids: list[str] = field(default_factory=list)
    vad_profile: dict[str, float] = field(default_factory=dict)
    camelot_data: dict[str, Any] = field(default_factory=dict)
    goal: str = ""
    intensity: float = 0.5
    direction: str = ""
    duration_min: float = 0.0


class TrackSchema:
    IDENTITY_FIELDS = ["id", "path", "title", "artist", "album", "genre", "duration_s", "sr"]
    MUSICAL_FIELDS = ["bpm", "key_str", "camelot", "scale", "mode", "key_confidence", "time_signature"]
    EMOTIONAL_FIELDS = ["valence", "arousal", "dominance", "emotion_tags", "mood_scores"]
    SPECTRAL_FIELDS = [
        "spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
        "brightness", "zero_crossing_rate", "rms_energy", "dynamic_range",
        "mfcc_coefficients",
    ]
    BAND_FIELDS = ["bass_energy", "mid_energy", "treble_energy"]
    DYNAMICS_FIELDS = ["loudness_db", "dynamic_complexity", "attack_density"]
    TEMPORAL_FIELDS = ["emotion_arc"]
    LYRICS_FIELDS = ["has_lyrics", "lyric_language", "lyric_sentiment"]
    AUDIO_DESCRIPTOR_FIELDS = [
        "danceability", "energy", "acousticness", "instrumentalness",
        "speechiness", "liveness",
    ]

    ALL_NUMERIC_FEATURES = (
        MUSICAL_FIELDS[:-2] + EMOTIONAL_FIELDS[:3] + SPECTRAL_FIELDS[:-1]
        + BAND_FIELDS + DYNAMICS_FIELDS + AUDIO_DESCRIPTOR_FIELDS
    )

    @classmethod
    def feature_count(cls) -> int:
        return len(cls.ALL_NUMERIC_FEATURES)

    @classmethod
    def field_groups(cls) -> dict[str, list[str]]:
        return {
            "identity": cls.IDENTITY_FIELDS,
            "musical": cls.MUSICAL_FIELDS,
            "emotional": cls.EMOTIONAL_FIELDS,
            "spectral": cls.SPECTRAL_FIELDS,
            "band_energy": cls.BAND_FIELDS,
            "dynamics": cls.DYNAMICS_FIELDS,
            "temporal": cls.TEMPORAL_FIELDS,
            "lyrics": cls.LYRICS_FIELDS,
            "audio_descriptors": cls.AUDIO_DESCRIPTOR_FIELDS,
        }
