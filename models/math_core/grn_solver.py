"""Generative Relational Network (GRN) solver.

Implements the D/C/E node dynamics from US 12,651,165:

    D(t) = α · I(t) + β · F(t)          (dominance-factor node)
    C(t) = C(t-1) + D(t)                (cumulative-change node)
    E(t) = E(t-1) + γ · C(t)            (evolution-of-internal-component)

Where:
    I(t) = change intensity at time t
    F(t) = change frequency at time t
    α, β, γ = learned or specified coefficients

The GRN models relationships between entities (e.g., neurotransmitters)
as a directed graph. Each edge carries D/C/E dynamics. The simulate →
eliminate → identify pipeline discovers which interactions drive a
target outcome.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class GRNNode:
    """A node in the GRN representing an entity (e.g., a neurotransmitter)."""
    name: str
    baseline: float = 0.0
    state: float = 0.0


@dataclass
class GRNEdge:
    """A directed edge with D/C/E dynamics."""
    source: str
    target: str
    alpha: float = 1.0    # intensity weight
    beta: float = 1.0     # frequency weight
    gamma: float = 0.1    # evolution rate
    D: float = 0.0        # current dominance-factor
    C: float = 0.0        # cumulative change
    E: float = 0.0        # evolution state


class GRNSolver:
    """Simulates GRN dynamics over discrete time steps."""

    def __init__(self, nodes: list[GRNNode], edges: list[GRNEdge]):
        self.nodes = {n.name: n for n in nodes}
        self.edges = edges
        self._edge_map: dict[str, list[GRNEdge]] = {}
        for e in edges:
            self._edge_map.setdefault(e.target, []).append(e)

    def step(
        self,
        intensities: dict[str, float],
        frequencies: dict[str, float],
    ) -> dict[str, float]:
        """Advance one time step.

        Args:
            intensities: I(t) per edge source
            frequencies: F(t) per edge source

        Returns:
            Updated node states.
        """
        for edge in self.edges:
            I_t = intensities.get(edge.source, 0.0)
            F_t = frequencies.get(edge.source, 0.0)

            # D(t) = α·I(t) + β·F(t)
            edge.D = edge.alpha * I_t + edge.beta * F_t

            # C(t) = C(t-1) + D(t)
            edge.C = edge.C + edge.D

            # E(t) = E(t-1) + γ·C(t)
            edge.E = edge.E + edge.gamma * edge.C

        # Update target node states from incoming edges
        for name, node in self.nodes.items():
            incoming = self._edge_map.get(name, [])
            if incoming:
                total_effect = sum(e.E for e in incoming)
                node.state = node.baseline + total_effect

        return {n: node.state for n, node in self.nodes.items()}

    def simulate(
        self,
        intensities_seq: list[dict[str, float]],
        frequencies_seq: list[dict[str, float]],
    ) -> dict[str, np.ndarray]:
        """Simulate over T time steps.

        Returns:
            Dict mapping node name → array of states (T,).
        """
        T = len(intensities_seq)
        history: dict[str, list[float]] = {n: [] for n in self.nodes}

        self.reset()
        for t in range(T):
            states = self.step(intensities_seq[t], frequencies_seq[t])
            for n, val in states.items():
                history[n].append(val)

        return {n: np.array(v) for n, v in history.items()}

    def eliminate_edge(self, source: str, target: str) -> "GRNSolver":
        """Create a copy with one edge removed (simulate→eliminate pipeline)."""
        new_edges = [e for e in self.edges
                     if not (e.source == source and e.target == target)]
        new_nodes = [GRNNode(n.name, n.baseline) for n in self.nodes.values()]
        return GRNSolver(new_nodes, [
            GRNEdge(e.source, e.target, e.alpha, e.beta, e.gamma)
            for e in new_edges
        ])

    def identify_critical_edges(
        self,
        intensities_seq: list[dict[str, float]],
        frequencies_seq: list[dict[str, float]],
        target_node: str,
    ) -> list[tuple[str, str, float]]:
        """Simulate→eliminate→identify: find edges whose removal most
        changes the target node's final state.

        Returns list of (source, target, |Δ_state|) sorted by impact.
        """
        full_history = self.simulate(intensities_seq, frequencies_seq)
        full_final = full_history[target_node][-1]

        impacts = []
        for edge in self.edges:
            if edge.target != target_node and edge.source != target_node:
                continue
            ablated = self.eliminate_edge(edge.source, edge.target)
            ablated_history = ablated.simulate(intensities_seq, frequencies_seq)
            ablated_final = ablated_history[target_node][-1]
            delta = abs(full_final - ablated_final)
            impacts.append((edge.source, edge.target, delta))

        impacts.sort(key=lambda x: x[2], reverse=True)
        return impacts

    def reset(self):
        for node in self.nodes.values():
            node.state = node.baseline
        for edge in self.edges:
            edge.D = 0.0
            edge.C = 0.0
            edge.E = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Canonical GRN methods (from methods/grn.py production engine)
# ═══════════════════════════════════════════════════════════════════════


def propagate_canonical(
    adjacency: np.ndarray,
    perturbations: np.ndarray,
    damping: float = 0.85,
    max_passes: int = 20,
    convergence: float = 0.001,
) -> dict:
    """state ← perturbations + tanh(damping · Aᵀ · state)"""
    adj = np.asarray(adjacency, dtype=float)
    state = np.asarray(perturbations, dtype=float).copy()
    n = len(state)
    for p in range(max_passes):
        new_state = perturbations.copy()
        for i in range(n):
            incoming = 0.0
            for j in range(n):
                if adj[j, i] != 0 and state[j] != 0:
                    incoming += adj[j, i] * state[j] * damping
            new_state[i] += np.tanh(incoming)
        delta = float(np.max(np.abs(new_state - state)))
        state = new_state
        if delta < convergence:
            return {"state": state, "passes": p + 1, "converged": True}
    return {"state": state, "passes": max_passes, "converged": False}


def compute_dominance_scores(
    adjacency: np.ndarray,
    state: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> np.ndarray:
    """D_i = tanh(α·I_i + β·|state_i|) where I_i = Σ_j |A_ji|·|state_j|."""
    adj = np.asarray(adjacency, dtype=float)
    s = np.asarray(state, dtype=float)
    n = len(s)
    influence = np.array([
        float(np.dot(np.abs(adj[:, i]), np.abs(s))) for i in range(n)
    ])
    return np.tanh(alpha * influence + beta * np.abs(s))


def sensitivity_analysis(
    adjacency: np.ndarray,
    perturbation_magnitude: float = 0.1,
    damping: float = 0.85,
    max_passes: int = 20,
) -> np.ndarray:
    """Finite-difference Jacobian: J[i,j] = ∂state_j / ∂pert_i."""
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    baseline = propagate_canonical(adj, np.zeros(n), damping, max_passes)["state"]
    jac = np.zeros((n, n))
    for i in range(n):
        pert = np.zeros(n)
        pert[i] = perturbation_magnitude
        perturbed = propagate_canonical(adj, pert, damping, max_passes)["state"]
        jac[i] = (perturbed - baseline) / perturbation_magnitude
    return jac


def network_motifs(adjacency: np.ndarray, threshold: float = 1e-6) -> dict:
    adj = (np.abs(np.asarray(adjacency, dtype=float)) > threshold).astype(int)
    n = adj.shape[0]
    ffls, fbls, bifans = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j or not adj[i, j]:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                # Feed-forward: i→j, i→k, j→k
                if adj[i, k] and adj[j, k]:
                    ffls.append((i, j, k))
                # 3-cycle: i→j→k→i
                if adj[j, k] and adj[k, i]:
                    fbls.append((i, j, k))
    # Bifan: i→k, i→l, j→k, j→l
    for i in range(n):
        targets_i = set(np.where(adj[i] > 0)[0])
        for j in range(i + 1, n):
            targets_j = set(np.where(adj[j] > 0)[0])
            common = targets_i & targets_j - {i, j}
            if len(common) >= 2:
                common_list = sorted(common)
                for ci in range(len(common_list)):
                    for cj in range(ci + 1, len(common_list)):
                        bifans.append((i, j, common_list[ci], common_list[cj]))
    return {
        "feed_forward_loops": ffls,
        "feedback_loops": fbls,
        "bifan_motifs": bifans,
        "counts": {
            "feed_forward_loops": len(ffls),
            "feedback_loops": len(fbls),
            "bifan_motifs": len(bifans),
        },
    }
