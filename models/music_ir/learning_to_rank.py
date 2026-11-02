"""Learning to rank: pairwise/listwise ranking for music retrieval."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np


@dataclass
class RankingFeatures:
    query_id: str
    doc_id: str
    features: np.ndarray
    relevance: float = 0.0


@dataclass
class RankingModel:
    weights: np.ndarray
    bias: float = 0.0
    learning_rate: float = 0.001
    regularization: float = 0.0001


@dataclass
class RankingResult:
    doc_id: str
    score: float
    rank: int


def score_document(model: RankingModel, features: np.ndarray) -> float:
    return float(np.dot(model.weights, features) + model.bias)


def rank_documents(model: RankingModel,
                   docs: List[RankingFeatures]) -> List[RankingResult]:
    scored = [(d, score_document(model, d.features)) for d in docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        RankingResult(doc_id=d.doc_id, score=s, rank=i + 1)
        for i, (d, s) in enumerate(scored)
    ]


def pairwise_loss(s_i: float, s_j: float, label: float) -> float:
    diff = s_i - s_j
    return math.log1p(math.exp(-label * diff))


def pairwise_gradient(s_i: float, s_j: float, label: float) -> float:
    diff = label * (s_i - s_j)
    sigma = 1.0 / (1.0 + math.exp(diff))
    return -label * sigma


def listwise_softmax_loss(scores: np.ndarray,
                          relevances: np.ndarray) -> float:
    exp_scores = np.exp(scores - np.max(scores))
    softmax = exp_scores / (np.sum(exp_scores) + 1e-10)
    exp_rel = np.exp(relevances - np.max(relevances))
    target = exp_rel / (np.sum(exp_rel) + 1e-10)
    return -float(np.sum(target * np.log(softmax + 1e-10)))


def listwise_gradient(scores: np.ndarray,
                      relevances: np.ndarray) -> np.ndarray:
    exp_scores = np.exp(scores - np.max(scores))
    softmax = exp_scores / (np.sum(exp_scores) + 1e-10)
    exp_rel = np.exp(relevances - np.max(relevances))
    target = exp_rel / (np.sum(exp_rel) + 1e-10)
    return softmax - target


def ndcg_gain(relevance: float) -> float:
    return 2.0 ** relevance - 1.0


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = relevances[:k]
    discounts = np.log2(np.arange(len(rel)) + 2.0)
    gains = np.array([ndcg_gain(r) for r in rel])
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(np.sort(relevances)[::-1], k)
    if ideal < 1e-10:
        return 0.0
    return actual / ideal


def lambda_weight(rank_i: int, rank_j: int,
                  relevances: np.ndarray, k: int) -> float:
    ndcg_before = ndcg_at_k(relevances, k)
    swapped = relevances.copy()
    swapped[rank_i], swapped[rank_j] = swapped[rank_j], swapped[rank_i]
    ndcg_after = ndcg_at_k(swapped, k)
    return abs(ndcg_after - ndcg_before)


def lambdarank_gradients(model: RankingModel,
                         docs: List[RankingFeatures],
                         k: int = 10) -> np.ndarray:
    n = len(docs)
    scores = np.array([score_document(model, d.features) for d in docs])
    order = np.argsort(-scores)
    relevances = np.array([docs[i].relevance for i in order])
    grad = np.zeros_like(model.weights)
    for i in range(n):
        for j in range(i + 1, n):
            if docs[order[i]].relevance == docs[order[j]].relevance:
                continue
            s_i = scores[order[i]]
            s_j = scores[order[j]]
            label = 1.0 if docs[order[i]].relevance > docs[order[j]].relevance else -1.0
            lam = lambda_weight(i, j, relevances, k)
            rho = pairwise_gradient(s_i, s_j, label) * lam
            grad += rho * (docs[order[i]].features - docs[order[j]].features)
    return grad


def train_step(model: RankingModel,
               docs: List[RankingFeatures],
               k: int = 10) -> float:
    grad = lambdarank_gradients(model, docs, k)
    grad += model.regularization * model.weights
    model.weights -= model.learning_rate * grad
    scores = np.array([score_document(model, d.features) for d in docs])
    order = np.argsort(-scores)
    rels = np.array([docs[i].relevance for i in order])
    return ndcg_at_k(rels, k)


def init_model(n_features: int, lr: float = 0.001,
               reg: float = 0.0001) -> RankingModel:
    return RankingModel(
        weights=np.zeros(n_features),
        bias=0.0,
        learning_rate=lr,
        regularization=reg,
    )


def feature_importance(model: RankingModel) -> np.ndarray:
    abs_w = np.abs(model.weights)
    total = abs_w.sum()
    if total < 1e-10:
        return np.ones_like(abs_w) / len(abs_w)
    return abs_w / total


def mean_reciprocal_rank(ranked_results: List[List[RankingResult]],
                         relevant_ids: List[Set]) -> float:
    rr_sum = 0.0
    for results, rel in zip(ranked_results, relevant_ids):
        for r in results:
            if r.doc_id in rel:
                rr_sum += 1.0 / r.rank
                break
    return rr_sum / max(len(ranked_results), 1)


def precision_at_k(results: List[RankingResult],
                   relevant: set, k: int) -> float:
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.doc_id in relevant)
    return hits / max(k, 1)


def recall_at_k(results: List[RankingResult],
                relevant: set, k: int) -> float:
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.doc_id in relevant)
    return hits / max(len(relevant), 1)


def average_precision(results: List[RankingResult],
                      relevant: set) -> float:
    hits = 0
    sum_prec = 0.0
    for i, r in enumerate(results):
        if r.doc_id in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / max(len(relevant), 1)
