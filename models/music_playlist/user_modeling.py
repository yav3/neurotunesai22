"""User taste modeling for playlist systems."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ListeningRecord:
    track_id: str
    timestamp: float
    duration_s: float
    total_duration_s: float
    skipped: bool = False


@dataclass
class TasteVector:
    genre_weights: np.ndarray
    genre_labels: List[str]
    freshness: float
    diversity_score: float


@dataclass
class UserProfile:
    user_id: str
    taste: TasteVector
    novelty_preference: float
    avg_session_length: float
    active_hours: np.ndarray


def completion_ratio(record: ListeningRecord) -> float:
    if record.total_duration_s <= 0:
        return 0.0
    return min(record.duration_s / record.total_duration_s, 1.0)


def implicit_rating(record: ListeningRecord, max_plays: int = 10) -> float:
    cr = completion_ratio(record)
    skip_penalty = 0.0 if not record.skipped else -0.3
    return np.clip(cr + skip_penalty, 0.0, 1.0)


def preference_decay(
    ratings: np.ndarray, timestamps: np.ndarray,
    half_life_days: float = 30.0, now: Optional[float] = None,
) -> np.ndarray:
    if now is None:
        now = timestamps[-1] if len(timestamps) > 0 else 0.0
    age_days = (now - timestamps) / 86400.0
    weights = np.exp(-np.log(2.0) * age_days / half_life_days)
    return ratings * weights


def estimate_taste_vector(
    genre_per_track: np.ndarray,
    ratings: np.ndarray,
    timestamps: np.ndarray,
    genre_labels: List[str],
    half_life_days: float = 30.0,
) -> TasteVector:
    decayed = preference_decay(ratings, timestamps, half_life_days)
    weighted = genre_per_track.T @ decayed
    total = np.sum(weighted)
    if total > 1e-12:
        genre_weights = weighted / total
    else:
        genre_weights = np.ones(len(genre_labels)) / len(genre_labels)
    entropy = -np.sum(genre_weights * np.log(genre_weights + 1e-12))
    max_entropy = np.log(len(genre_labels))
    diversity = entropy / max_entropy if max_entropy > 0 else 0.0
    freshness = float(np.mean(decayed) / (np.mean(ratings) + 1e-12))
    return TasteVector(
        genre_weights=genre_weights, genre_labels=genre_labels,
        freshness=freshness, diversity_score=float(diversity),
    )


def novelty_score(
    track_genre: np.ndarray, taste: TasteVector,
) -> float:
    familiar = float(np.dot(track_genre, taste.genre_weights))
    return 1.0 - familiar


def familiarity_score(
    track_genre: np.ndarray, taste: TasteVector,
) -> float:
    return float(np.dot(track_genre, taste.genre_weights))


def novelty_familiarity_balance(
    candidates_genre: np.ndarray, taste: TasteVector,
    alpha: float = 0.3,
) -> np.ndarray:
    fam = candidates_genre @ taste.genre_weights
    nov = 1.0 - fam
    return alpha * nov + (1.0 - alpha) * fam


def genre_affinity(taste: TasteVector, top_k: int = 3) -> List[Tuple[str, float]]:
    indices = np.argsort(taste.genre_weights)[::-1][:top_k]
    return [(taste.genre_labels[i], float(taste.genre_weights[i])) for i in indices]


def temporal_listening_pattern(
    timestamps: np.ndarray, n_bins: int = 24,
) -> np.ndarray:
    hours = (timestamps % 86400) / 3600.0
    hist, _ = np.histogram(hours, bins=n_bins, range=(0, 24))
    total = np.sum(hist)
    if total > 0:
        hist = hist / total
    return hist.astype(float)


def user_similarity_cosine(
    taste_a: TasteVector, taste_b: TasteVector,
) -> float:
    a = taste_a.genre_weights
    b = taste_b.genre_weights
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def user_similarity_jaccard(
    history_a: set, history_b: set,
) -> float:
    inter = len(history_a & history_b)
    union = len(history_a | history_b)
    return inter / union if union > 0 else 0.0


def collaborative_neighbors(
    target_taste: TasteVector,
    all_tastes: List[TasteVector],
    k: int = 10,
) -> List[Tuple[int, float]]:
    sims = []
    for i, t in enumerate(all_tastes):
        s = user_similarity_cosine(target_taste, t)
        sims.append((i, s))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


def session_length_stats(
    records: List[ListeningRecord],
) -> Tuple[float, float]:
    if not records:
        return 0.0, 0.0
    durations = np.array([r.duration_s for r in records])
    return float(np.mean(durations)), float(np.std(durations))


def skip_rate(records: List[ListeningRecord]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.skipped) / len(records)


def build_user_profile(
    user_id: str,
    records: List[ListeningRecord],
    genre_per_track: np.ndarray,
    genre_labels: List[str],
) -> UserProfile:
    ratings = np.array([implicit_rating(r) for r in records])
    timestamps = np.array([r.timestamp for r in records])
    taste = estimate_taste_vector(
        genre_per_track, ratings, timestamps, genre_labels,
    )
    nov_pref = 1.0 - taste.diversity_score
    avg_len, _ = session_length_stats(records)
    active = temporal_listening_pattern(timestamps)
    return UserProfile(
        user_id=user_id, taste=taste,
        novelty_preference=nov_pref,
        avg_session_length=avg_len,
        active_hours=active,
    )
