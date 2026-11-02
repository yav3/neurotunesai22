"""Network graph visualization: artist similarity, genre relationships, chord graphs.

Provides graph layout algorithms and rendering data structures for music
relationship networks — artist similarity, genre taxonomy, chord transition
graphs, and instrument co-occurrence networks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraphNode:
    node_id: str
    label: str
    x: float = 0.0
    y: float = 0.0
    size: float = 1.0
    color: str = "#4488cc"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    weight: float = 1.0
    label: str = ""


@dataclass
class NetworkGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def adjacency_matrix(self) -> np.ndarray:
        idx = {n.node_id: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        mat = np.zeros((n, n))
        for e in self.edges:
            i, j = idx.get(e.source), idx.get(e.target)
            if i is not None and j is not None:
                mat[i, j] = e.weight
                mat[j, i] = e.weight
        return mat

    def degree(self, node_id: str) -> float:
        return sum(e.weight for e in self.edges if e.source == node_id or e.target == node_id)


def force_directed_layout(
    graph: NetworkGraph,
    iterations: int = 100,
    k: float = 1.0,
    cooling: float = 0.95,
) -> None:
    n = len(graph.nodes)
    if n == 0:
        return
    pos = np.random.RandomState(42).randn(n, 2)
    idx = {node.node_id: i for i, node in enumerate(graph.nodes)}
    temp = k * math.sqrt(n)
    for _ in range(iterations):
        disp = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist = max(np.linalg.norm(delta), 0.01)
                repulsion = k * k / dist
                direction = delta / dist
                disp[i] += direction * repulsion
                disp[j] -= direction * repulsion
        for edge in graph.edges:
            i = idx.get(edge.source)
            j = idx.get(edge.target)
            if i is None or j is None:
                continue
            delta = pos[j] - pos[i]
            dist = max(np.linalg.norm(delta), 0.01)
            attraction = dist * dist / k * edge.weight
            direction = delta / dist
            disp[i] += direction * attraction
            disp[j] -= direction * attraction
        lengths = np.linalg.norm(disp, axis=1, keepdims=True).clip(0.01)
        pos += disp / lengths * min(temp, lengths)
        temp *= cooling
    for i, node in enumerate(graph.nodes):
        node.x = float(pos[i, 0])
        node.y = float(pos[i, 1])


def chord_transition_graph(
    transitions: list[tuple[str, str]],
) -> NetworkGraph:
    counts: dict[tuple[str, str], int] = {}
    chord_set: set[str] = set()
    for a, b in transitions:
        counts[(a, b)] = counts.get((a, b), 0) + 1
        chord_set.add(a)
        chord_set.add(b)
    nodes = [GraphNode(node_id=c, label=c) for c in sorted(chord_set)]
    max_count = max(counts.values()) if counts else 1
    edges = [
        GraphEdge(source=a, target=b, weight=c / max_count)
        for (a, b), c in counts.items()
    ]
    return NetworkGraph(nodes=nodes, edges=edges)


def genre_taxonomy_graph(
    taxonomy: dict[str, list[str]],
) -> NetworkGraph:
    nodes_dict: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    for parent, children in taxonomy.items():
        if parent not in nodes_dict:
            nodes_dict[parent] = GraphNode(node_id=parent, label=parent, size=2.0)
        for child in children:
            if child not in nodes_dict:
                nodes_dict[child] = GraphNode(node_id=child, label=child, size=1.0)
            edges.append(GraphEdge(source=parent, target=child))
    return NetworkGraph(nodes=list(nodes_dict.values()), edges=edges)


def artist_similarity_graph(
    artists: list[str],
    similarity_matrix: np.ndarray,
    threshold: float = 0.3,
) -> NetworkGraph:
    nodes = [GraphNode(node_id=a, label=a) for a in artists]
    edges: list[GraphEdge] = []
    n = len(artists)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            if sim >= threshold:
                edges.append(GraphEdge(source=artists[i], target=artists[j], weight=sim))
    return NetworkGraph(nodes=nodes, edges=edges)


def community_detection_louvain(
    graph: NetworkGraph,
    resolution: float = 1.0,
) -> dict[str, int]:
    adj = graph.adjacency_matrix()
    n = len(graph.nodes)
    if n == 0:
        return {}
    communities = list(range(n))
    m = float(np.sum(adj)) / 2.0
    if m < 1e-10:
        return {graph.nodes[i].node_id: 0 for i in range(n)}
    improved = True
    while improved:
        improved = False
        for i in range(n):
            best_comm = communities[i]
            best_gain = 0.0
            ki = float(np.sum(adj[i]))
            old_comm = communities[i]
            neighbor_comms = set(communities[j] for j in range(n) if adj[i, j] > 0)
            neighbor_comms.add(old_comm)
            for c in neighbor_comms:
                members = [j for j in range(n) if communities[j] == c and j != i]
                ki_in = sum(float(adj[i, j]) for j in members)
                sigma_tot = sum(float(np.sum(adj[j])) for j in members)
                gain = ki_in - resolution * ki * sigma_tot / (2.0 * m + 1e-12)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = c
            if best_comm != old_comm:
                communities[i] = best_comm
                improved = True
    return {graph.nodes[i].node_id: communities[i] for i in range(n)}
