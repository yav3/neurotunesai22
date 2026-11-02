"""Recommendation explanation: feature attribution, similar-item
evidence, diversity justification, counterfactual reasoning.

Based on Tintarev & Masthoff (2007) explanation taxonomy,
Herlocker et al. (2000) collaborative filtering explanations,
and Vig et al. (2009) tag-based explanations. Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class ExplanationStyle(Enum):
    FEATURE_BASED = "feature_based"
    ITEM_BASED = "item_based"
    USER_BASED = "user_based"
    TAG_BASED = "tag_based"
    COUNTERFACTUAL = "counterfactual"


class RecommendationReason(Enum):
    SIMILAR_TO_LIKED = "similar_to_liked"
    POPULAR_IN_GENRE = "popular_in_genre"
    DIVERSITY = "diversity"
    NOVELTY = "novelty"
    MOOD_MATCH = "mood_match"
    COLLABORATIVE = "collaborative"


@dataclass
class FeatureContribution:
    feature_name: str
    contribution: float
    direction: str


@dataclass
class ItemEvidence:
    reference_item_id: int
    similarity: float
    shared_features: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    item_id: int
    score: float
    style: ExplanationStyle
    reason: RecommendationReason
    contributions: List[FeatureContribution] = field(default_factory=list)
    evidence: List[ItemEvidence] = field(default_factory=list)
    text: str = ""


def feature_attribution(item_features: np.ndarray,
                        user_preferences: np.ndarray,
                        feature_names: List[str]) -> List[FeatureContribution]:
    contributions = item_features * user_preferences
    result = []
    for i, name in enumerate(feature_names):
        c = float(contributions[i])
        direction = "positive" if c > 0 else "negative" if c < 0 else "neutral"
        result.append(FeatureContribution(name, abs(c), direction))
    result.sort(key=lambda x: x.contribution, reverse=True)
    return result


def top_k_features(item_features: np.ndarray,
                   user_preferences: np.ndarray,
                   feature_names: List[str],
                   k: int = 3) -> List[FeatureContribution]:
    all_contribs = feature_attribution(item_features, user_preferences, feature_names)
    return all_contribs[:k]


def similar_item_evidence(target_features: np.ndarray,
                          history_features: np.ndarray,
                          history_ids: np.ndarray,
                          feature_names: List[str],
                          top_k: int = 3,
                          threshold: float = 0.5) -> List[ItemEvidence]:
    if len(history_features) == 0:
        return []
    norms_t = np.linalg.norm(target_features)
    norms_h = np.linalg.norm(history_features, axis=1)
    if norms_t < 1e-10:
        return []
    dots = history_features @ target_features
    sims = dots / (norms_t * norms_h + 1e-10)
    top_indices = np.argsort(-sims)[:top_k]
    evidence = []
    for idx in top_indices:
        sim = float(sims[idx])
        if sim < threshold:
            continue
        shared = []
        for j, name in enumerate(feature_names):
            if abs(target_features[j] - history_features[idx, j]) < 0.1:
                shared.append(name)
        evidence.append(ItemEvidence(int(history_ids[idx]), sim, shared[:5]))
    return evidence


def diversity_justification(item_features: np.ndarray,
                            list_features: np.ndarray) -> Tuple[float, str]:
    if len(list_features) == 0:
        return 0.0, "first item in list"
    centroid = np.mean(list_features, axis=0)
    dist = float(np.linalg.norm(item_features - centroid))
    mean_dist = float(np.mean(np.linalg.norm(list_features - centroid, axis=1)))
    if dist > mean_dist * 1.5:
        reason = "adds variety to the recommendation list"
    elif dist > mean_dist:
        reason = "moderately different from current selections"
    else:
        reason = "consistent with current listening direction"
    return dist, reason


def novelty_score(item_popularity: float,
                  max_popularity: float) -> float:
    if max_popularity < 1e-10:
        return 1.0
    return 1.0 - (item_popularity / max_popularity)


def counterfactual_explanation(item_features: np.ndarray,
                               user_preferences: np.ndarray,
                               threshold: float,
                               feature_names: List[str]
                               ) -> List[Tuple[str, float]]:
    score = float(np.dot(item_features, user_preferences))
    if score <= threshold:
        return []
    contributions = item_features * user_preferences
    sorted_indices = np.argsort(contributions)[::-1]
    counterfactuals = []
    running = score
    for idx in sorted_indices:
        if running <= threshold:
            break
        removed = float(contributions[idx])
        if removed > 0:
            counterfactuals.append((feature_names[idx], removed))
            running -= removed
    return counterfactuals


def tag_relevance(item_tags: Dict[str, float],
                  user_tag_preferences: Dict[str, float],
                  top_k: int = 5) -> List[Tuple[str, float]]:
    relevance = []
    for tag, item_weight in item_tags.items():
        user_weight = user_tag_preferences.get(tag, 0.0)
        score = item_weight * user_weight
        if score > 0:
            relevance.append((tag, score))
    relevance.sort(key=lambda x: x[1], reverse=True)
    return relevance[:top_k]


def explain_recommendation(item_id: int,
                           item_features: np.ndarray,
                           user_preferences: np.ndarray,
                           feature_names: List[str],
                           history_features: Optional[np.ndarray] = None,
                           history_ids: Optional[np.ndarray] = None,
                           ) -> Explanation:
    score = float(np.dot(item_features, user_preferences))
    contribs = top_k_features(item_features, user_preferences, feature_names, k=3)
    evidence = []
    if history_features is not None and history_ids is not None:
        evidence = similar_item_evidence(
            item_features, history_features, history_ids, feature_names
        )
    if evidence:
        reason = RecommendationReason.SIMILAR_TO_LIKED
        style = ExplanationStyle.ITEM_BASED
    else:
        reason = RecommendationReason.MOOD_MATCH
        style = ExplanationStyle.FEATURE_BASED
    top_feat = contribs[0].feature_name if contribs else "overall match"
    text = f"Recommended because of {top_feat}"
    if evidence:
        text += f" (similar to item {evidence[0].reference_item_id})"
    return Explanation(item_id, score, style, reason, contribs, evidence, text)


def explanation_fidelity(explanation: Explanation,
                         item_features: np.ndarray,
                         user_preferences: np.ndarray) -> float:
    if not explanation.contributions:
        return 0.0
    explained_score = sum(c.contribution for c in explanation.contributions)
    actual_score = abs(float(np.dot(item_features, user_preferences)))
    if actual_score < 1e-10:
        return 1.0 if explained_score < 1e-10 else 0.0
    return float(min(explained_score / actual_score, 1.0))


def batch_explain(item_ids: np.ndarray,
                  item_features_matrix: np.ndarray,
                  user_preferences: np.ndarray,
                  feature_names: List[str]) -> List[Explanation]:
    explanations = []
    for i, item_id in enumerate(item_ids):
        exp = explain_recommendation(
            int(item_id), item_features_matrix[i], user_preferences, feature_names
        )
        explanations.append(exp)
    return explanations
