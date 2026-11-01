"""Molecule loader — quantum chemistry datasets from Merlin.

Loads Merlin's committed molecular datasets (QCSchema JSON format inside
compressed archives) into structured objects for NeuroTunesAI. Two dataset
families:

  - APB-E0: atomization energy benchmark (QM5/QM9 train/test splits, XYZ format)
  - MSR-ACC-TAE25: total atomization energy from Microsoft Research
    (QCSchema JSON, multiple basis sets + oracle + validation splits)

These provide the computational chemistry substrate for any future
molecular-property prediction or drug-likeness scoring pipeline.

Species: not_applicable (small-molecule quantum chemistry).
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MoleculeRecord:
    """A single molecule from a QCSchema JSON file."""
    name: str
    symbols: list[str]
    atomic_numbers: list[int]
    geometry: list[float]  # flattened xyz (n_atoms * 3)
    molecular_charge: float = 0.0
    molecular_multiplicity: int = 1
    file_id: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def formula_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.symbols:
            counts[s] = counts.get(s, 0) + 1
        return counts


@dataclass
class MoleculeDataset:
    """A collection of molecules from a single archive."""
    name: str
    source_path: str
    n_molecules: int
    n_files_in_archive: int
    unique_elements: set[str]
    atom_count_range: tuple[int, int]  # (min, max) atoms per molecule
    archive_format: str  # "zip" or "tar.xz"
    content_hash: str = ""
    molecules: list[MoleculeRecord] = field(default_factory=list)

    @property
    def element_list(self) -> list[str]:
        return sorted(self.unique_elements)

    def atom_count_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        """Histogram of atom counts across molecules."""
        counts = np.array([m.n_atoms for m in self.molecules], dtype=np.int32)
        if len(counts) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        bins = np.arange(counts.min(), counts.max() + 2)
        hist, edges = np.histogram(counts, bins=bins)
        return hist, edges[:-1]


def _parse_qcschema_molecule(data: dict) -> MoleculeRecord | None:
    """Parse a QCSchema molecule JSON dict."""
    if data.get("schema_name") != "qcschema_molecule":
        return None
    return MoleculeRecord(
        name=data.get("name", ""),
        symbols=data.get("symbols", []),
        atomic_numbers=data.get("atomic_numbers", []),
        geometry=data.get("geometry", []),
        molecular_charge=data.get("molecular_charge", 0.0),
        molecular_multiplicity=data.get("molecular_multiplicity", 1),
    )


def _parse_xyz_content(content: str) -> MoleculeRecord | None:
    """Parse XYZ format molecule file content."""
    lines = content.strip().split("\n")
    if len(lines) < 3:
        return None
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None
    comment = lines[1].strip()
    symbols: list[str] = []
    geometry: list[float] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        geometry.extend(float(x) for x in parts[1:4])
    if len(symbols) != n_atoms:
        return None
    ELEMENT_TO_Z = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
        "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
        "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    }
    return MoleculeRecord(
        name=comment,
        symbols=symbols,
        atomic_numbers=[ELEMENT_TO_Z.get(s, 0) for s in symbols],
        geometry=geometry,
    )


def load_molecule_dataset(
    archive_path: str | Path,
    max_molecules: int | None = None,
) -> MoleculeDataset:
    """Load a molecular dataset from a compressed archive.

    Args:
        archive_path: Path to .zip or .tar.xz archive.
        max_molecules: Cap on molecules to parse (None = all).
    """
    archive_path = Path(archive_path)
    molecules: list[MoleculeRecord] = []
    n_files = 0
    unique_elements: set[str] = set()
    min_atoms = float("inf")
    max_atoms = 0

    h = hashlib.sha256()
    h.update(archive_path.name.encode())

    if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
        fmt = "zip"
        with zipfile.ZipFile(archive_path) as z:
            names = sorted(z.namelist())
            n_files = len(names)
            h.update(str(n_files).encode())
            for name in names:
                if max_molecules is not None and len(molecules) >= max_molecules:
                    break
                if name.endswith(".json"):
                    try:
                        data = json.loads(z.read(name))
                        mol = _parse_qcschema_molecule(data)
                        if mol:
                            mol.file_id = Path(name).stem
                            molecules.append(mol)
                            unique_elements.update(mol.symbols)
                            min_atoms = min(min_atoms, mol.n_atoms)
                            max_atoms = max(max_atoms, mol.n_atoms)
                    except Exception:
                        continue
    elif archive_path.name.endswith(".tar.xz"):
        fmt = "tar.xz"
        with tarfile.open(archive_path, "r:xz") as t:
            members = sorted(t.getnames())
            n_files = len(members)
            h.update(str(n_files).encode())
            for name in members:
                if max_molecules is not None and len(molecules) >= max_molecules:
                    break
                if name.endswith(".xyz"):
                    try:
                        member = t.getmember(name)
                        f = t.extractfile(member)
                        if f is None:
                            continue
                        content = f.read().decode("utf-8")
                        mol = _parse_xyz_content(content)
                        if mol:
                            mol.file_id = Path(name).stem
                            molecules.append(mol)
                            unique_elements.update(mol.symbols)
                            min_atoms = min(min_atoms, mol.n_atoms)
                            max_atoms = max(max_atoms, mol.n_atoms)
                    except Exception:
                        continue
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    if not molecules:
        min_atoms = 0

    h.update(str(len(molecules)).encode())
    for mol in molecules[:100]:
        h.update(mol.name.encode())

    return MoleculeDataset(
        name=archive_path.stem,
        source_path=str(archive_path),
        n_molecules=len(molecules),
        n_files_in_archive=n_files,
        unique_elements=unique_elements,
        atom_count_range=(int(min_atoms) if min_atoms != float("inf") else 0, max_atoms),
        archive_format=fmt,
        content_hash=h.hexdigest(),
        molecules=molecules,
    )


def discover_molecule_datasets(
    merlin_molecules_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Discover molecular dataset archives in Merlin's data directory."""
    if merlin_molecules_dir is None:
        merlin_molecules_dir = (
            Path(__file__).parent.parent.parent.parent
            / "merlinaineuralplus2" / "backend" / "data" / "molecules"
        )
    else:
        merlin_molecules_dir = Path(merlin_molecules_dir)

    if not merlin_molecules_dir.exists():
        return []

    results: list[dict[str, Any]] = []
    for archive in sorted(merlin_molecules_dir.iterdir()):
        if not (archive.name.endswith(".zip") or archive.name.endswith(".tar.xz")):
            continue
        try:
            ds = load_molecule_dataset(archive, max_molecules=50)
            results.append({
                "source_path": str(archive),
                "name": archive.stem,
                "n_molecules": ds.n_molecules,
                "n_files_in_archive": ds.n_files_in_archive,
                "unique_elements": sorted(ds.unique_elements),
                "atom_count_range": list(ds.atom_count_range),
                "archive_format": ds.archive_format,
                "content_hash": ds.content_hash,
                "modality": "molecule",
            })
        except Exception:
            continue

    return results
