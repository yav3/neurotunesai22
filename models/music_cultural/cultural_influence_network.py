"""Cultural influence network: modeling how musical styles propagate across cultures.

Influence graph construction, diffusion dynamics, cultural distance metrics,
and transmission pathway analysis. Based on Granovetter (1978) threshold
diffusion, Toiviainen & Eerola (2006) cultural MDS, and Savage et al. (2015)
CantoCore cross-cultural classification.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class TransmissionMode(Enum):
    DIRECT_CONTACT = "direct_contact"
    TRADE_ROUTE = "trade_route"
    COLONIAL = "colonial"
    MEDIA = "media"
    DIASPORA = "diaspora"
    RELIGIOUS = "religious"


@dataclass
class CulturalNode:
    name: str
    region: str
    features: np.ndarray
    population_weight: float = 1.0
    openness: float = 0.5


@dataclass
class InfluenceEdge:
    source: int
    target: int
    weight: float
    mode: TransmissionMode = TransmissionMode.DIRECT_CONTACT
    year_start: Optional[int] = None
    year_end: Optional[int] = None


@dataclass
class DiffusionState:
    adoption: np.ndarray
    time_step: int
    newly_adopted: List[int] = field(default_factory=list)


def build_influence_graph(nodes: List[CulturalNode],
                          edges: List[InfluenceEdge]) -> np.ndarray:
    n = len(nodes)
    adj = np.zeros((n, n))
    for e in edges:
        adj[e.source, e.target] = e.weight
    return adj


def cultural_feature_distance(a: CulturalNode, b: CulturalNode) -> float:
    diff = a.features - b.features
    return float(np.sqrt(np.sum(diff ** 2)))


def cultural_distance_matrix(nodes: List[CulturalNode]) -> np.ndarray:
    n = len(nodes)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = cultural_feature_distance(nodes[i], nodes[j])
            D[i, j] = D[j, i] = d
    return D


def proximity_influence_graph(nodes: List[CulturalNode],
                              sigma: float = 1.0) -> np.ndarray:
    D = cultural_distance_matrix(nodes)
    W = np.exp(-D ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def threshold_diffusion(adj: np.ndarray, seeds: List[int],
                        thresholds: np.ndarray,
                        max_steps: int = 100) -> List[DiffusionState]:
    n = adj.shape[0]
    adopted = np.zeros(n, dtype=bool)
    for s in seeds:
        adopted[s] = True
    history = [DiffusionState(adopted.astype(float).copy(), 0, list(seeds))]

    for t in range(1, max_steps + 1):
        new_adopted = []
        for i in range(n):
            if adopted[i]:
                continue
            neighbor_influence = np.sum(adj[:, i] * adopted)
            total_possible = np.sum(adj[:, i])
            if total_possible > 0 and neighbor_influence / total_possible >= thresholds[i]:
                new_adopted.append(i)
        if not new_adopted:
            break
        for i in new_adopted:
            adopted[i] = True
        history.append(DiffusionState(adopted.astype(float).copy(), t, new_adopted))
    return history


def continuous_diffusion(adj: np.ndarray, initial: np.ndarray,
                         beta: float = 0.1, gamma: float = 0.01,
                         steps: int = 50) -> np.ndarray:
    n = adj.shape[0]
    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    T = adj / row_sums[:, None]
    trajectory = np.zeros((steps + 1, n))
    trajectory[0] = initial.copy()
    for t in range(steps):
        x = trajectory[t]
        dx = beta * (T @ x - x) - gamma * x
        trajectory[t + 1] = np.clip(x + dx, 0, 1)
    return trajectory


def influence_centrality(adj: np.ndarray, alpha: float = 0.85,
                         max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    n = adj.shape[0]
    col_sums = adj.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    M = adj / col_sums[None, :]
    v = np.ones(n) / n
    for _ in range(max_iter):
        v_new = alpha * M @ v + (1 - alpha) / n
        v_new /= np.sum(v_new)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    return v


def transmission_pathways(adj: np.ndarray, source: int, target: int,
                          max_hops: int = 5) -> List[List[int]]:
    paths = []
    queue = [[source]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == target and len(path) > 1:
            paths.append(path)
            continue
        if len(path) > max_hops:
            continue
        for neighbor in range(adj.shape[0]):
            if adj[node, neighbor] > 0 and neighbor not in path:
                queue.append(path + [neighbor])
    return paths


def pathway_strength(adj: np.ndarray, path: List[int]) -> float:
    strength = 1.0
    for i in range(len(path) - 1):
        strength *= adj[path[i], path[i + 1]]
    return strength


def dominant_pathway(adj: np.ndarray, source: int, target: int,
                     max_hops: int = 5) -> Tuple[List[int], float]:
    paths = transmission_pathways(adj, source, target, max_hops)
    if not paths:
        return [], 0.0
    best_path = max(paths, key=lambda p: pathway_strength(adj, p))
    return best_path, pathway_strength(adj, best_path)


def cultural_mds(distance_matrix: np.ndarray, dims: int = 2) -> np.ndarray:
    n = distance_matrix.shape[0]
    D2 = distance_matrix ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:dims]
    eigvecs = eigvecs[:, idx][:, :dims]
    return eigvecs * np.sqrt(np.maximum(eigvals, 0))


def innovation_rate(trajectory: np.ndarray) -> np.ndarray:
    if trajectory.shape[0] < 2:
        return np.zeros(trajectory.shape[1])
    deltas = np.diff(trajectory, axis=0)
    return np.mean(np.abs(deltas), axis=0)


def cultural_entropy(features: np.ndarray) -> float:
    n = features.shape[0]
    if n < 2:
        return 0.0
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((features[i] - features[j]) ** 2))
            D[i, j] = D[j, i] = d
    mean_d = np.mean(D[np.triu_indices(n, k=1)])
    if mean_d < 1e-15:
        return 0.0
    normed = D / mean_d
    flat = normed[np.triu_indices(n, k=1)]
    flat = flat[flat > 0]
    if len(flat) == 0:
        return 0.0
    p = flat / np.sum(flat)
    return float(-np.sum(p * np.log(p + 1e-30)))


def syncretism_score(node: CulturalNode,
                     traditions: List[CulturalNode]) -> float:
    if not traditions:
        return 0.0
    distances = [cultural_feature_distance(node, t) for t in traditions]
    d_arr = np.array(distances)
    if np.max(d_arr) < 1e-15:
        return 1.0
    similarities = 1.0 - d_arr / np.max(d_arr)
    sorted_sim = np.sort(similarities)[::-1]
    if len(sorted_sim) < 2:
        return 0.0
    return float(sorted_sim[1] / (sorted_sim[0] + 1e-10))


def influence_asymmetry(adj: np.ndarray) -> np.ndarray:
    return adj - adj.T


def net_influence(adj: np.ndarray) -> np.ndarray:
    out_influence = adj.sum(axis=1)
    in_influence = adj.sum(axis=0)
    return out_influence - in_influence
