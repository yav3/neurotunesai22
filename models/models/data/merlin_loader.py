"""Load expression datasets from Merlin's data registry for training.

Reads CSV expression matrices committed in Merlin's backend/data/expression/
directory, applies species firewall validation, and produces tensors suitable
for the PerturbationResponseModel and other math-first modules.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.interface.merlin_bridge import (
    MerlinDataRegistry,
    DataRegistryEntry,
    validate_species_boundary,
    select_regime,
    Species,
    MERLIN_DATA,
)


def load_expression_csv(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Load a Merlin-format expression CSV.

    Returns (gene_names, sample_ids, expression_matrix[genes, samples]).
    """
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_ids = header[1:]
        genes = []
        rows = []
        for row in reader:
            genes.append(row[0])
            rows.append([float(x) if x else 0.0 for x in row[1:]])
    return genes, sample_ids, np.array(rows, dtype=np.float32)


def load_from_registry(
    registry: MerlinDataRegistry,
    entry_id: str,
    declared_species: Species | None = None,
) -> dict:
    """Load an expression dataset by registry entry ID with species validation."""
    entry = registry.get(entry_id)
    if entry is None:
        raise KeyError(f"Entry {entry_id!r} not in Merlin data registry")
    if entry.kind != "expression":
        raise ValueError(f"Entry {entry_id!r} is kind={entry.kind!r}, not 'expression'")

    path = registry.resolve_path(entry)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    genes, samples, matrix = load_expression_csv(path)

    species = declared_species or entry.species
    validate_species_boundary(species, genes)

    regime = select_regime(n_samples=len(samples))

    return {
        "genes": genes,
        "samples": samples,
        "matrix": matrix,
        "species": species,
        "regime": regime,
        "entry": entry,
        "n_genes": len(genes),
        "n_samples": len(samples),
    }


def load_all_expression(
    registry: MerlinDataRegistry,
    species: Species | None = None,
    min_samples: int = 1,
) -> list[dict]:
    """Load all expression datasets matching criteria."""
    entries = registry.list_expression_data(species=species)
    results = []
    for entry in entries:
        path = registry.resolve_path(entry)
        if not path.exists():
            continue
        try:
            data = load_from_registry(registry, entry.entry_id)
            if data["n_samples"] >= min_samples:
                results.append(data)
        except (ValueError, KeyError):
            continue
    return results


class MerlinExpressionDataset(Dataset):
    """PyTorch Dataset wrapping Merlin expression data for model training.

    Each item is a dict with:
      - baseline: (n_genes,) tensor — log1p-transformed expression
      - sample_idx: int — index in the original matrix
      - dataset_id: str — Merlin registry entry ID
    """

    def __init__(
        self,
        registry: MerlinDataRegistry,
        entry_ids: list[str] | None = None,
        species: Species | None = None,
        log1p: bool = True,
    ):
        self.log1p = log1p
        self._items: list[tuple[torch.Tensor, str]] = []

        if entry_ids is None:
            entries = registry.list_expression_data(species=species)
            entry_ids = [e.entry_id for e in entries]

        for eid in entry_ids:
            try:
                data = load_from_registry(registry, eid)
            except (KeyError, FileNotFoundError, ValueError):
                continue
            mat = data["matrix"]
            if log1p:
                mat = np.log1p(np.clip(mat, 0, None))
            tensor = torch.from_numpy(mat.T)
            for i in range(tensor.shape[0]):
                self._items.append((tensor[i], eid))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        expression, dataset_id = self._items[idx]
        return {
            "baseline": expression,
            "sample_idx": idx,
            "dataset_id": dataset_id,
        }


class PairedConditionDataset(Dataset):
    """Dataset of (baseline, target) pairs from two-condition experiments.

    For perturbation-response training: control samples are baselines,
    treated samples are targets. Pairs are formed by matching within
    the same experiment.
    """

    def __init__(
        self,
        control_matrix: np.ndarray,
        treated_matrix: np.ndarray,
        gene_names: list[str],
        species: Species,
        log1p: bool = True,
    ):
        validate_species_boundary(species, gene_names)
        if control_matrix.shape[0] != treated_matrix.shape[0]:
            raise ValueError("Gene count mismatch between control and treated")

        self.gene_names = gene_names
        self.species = species
        self.n_genes = control_matrix.shape[0]

        ctrl = control_matrix.astype(np.float32)
        treat = treated_matrix.astype(np.float32)
        if log1p:
            ctrl = np.log1p(np.clip(ctrl, 0, None))
            treat = np.log1p(np.clip(treat, 0, None))

        self._control = torch.from_numpy(ctrl.T)
        self._treated = torch.from_numpy(treat.T)

        n_ctrl = self._control.shape[0]
        n_treat = self._treated.shape[0]
        self._pairs = [
            (i % n_ctrl, j % n_treat)
            for i in range(max(n_ctrl, n_treat))
            for j in range(max(n_ctrl, n_treat))
            if i < n_ctrl and j < n_treat
        ]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict:
        ci, ti = self._pairs[idx]
        return {
            "baseline": self._control[ci],
            "target": self._treated[ti],
            "control_idx": ci,
            "treated_idx": ti,
        }
