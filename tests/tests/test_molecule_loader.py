"""Tests for molecule loader."""
from __future__ import annotations

import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MERLIN_MOLECULES = Path("/home/user/merlinaineuralplus2/backend/data/molecules")


class TestMoleculeLoaderUnit:
    """Unit tests on synthetic molecule archives."""

    @pytest.fixture
    def zip_archive(self, tmp_path):
        mol1 = {
            "schema_name": "qcschema_molecule",
            "schema_version": 2,
            "name": "H2O",
            "symbols": ["O", "H", "H"],
            "atomic_numbers": [8, 1, 1],
            "geometry": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "molecular_charge": 0.0,
            "molecular_multiplicity": 1,
            "validated": True,
        }
        mol2 = {
            "schema_name": "qcschema_molecule",
            "schema_version": 2,
            "name": "CH4",
            "symbols": ["C", "H", "H", "H", "H"],
            "atomic_numbers": [6, 1, 1, 1, 1],
            "geometry": list(range(15)),
            "molecular_charge": 0.0,
            "molecular_multiplicity": 1,
            "validated": True,
        }
        archive = tmp_path / "test_molecules.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("mol/water.json", json.dumps(mol1))
            z.writestr("mol/methane.json", json.dumps(mol2))
        return archive

    @pytest.fixture
    def tar_archive(self, tmp_path):
        xyz1 = "3\nwater\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n"
        xyz2 = "2\nH2\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n"
        archive = tmp_path / "test_molecules.tar.xz"
        import io
        import lzma
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:xz") as t:
            for name, content in [("mol/water.xyz", xyz1), ("mol/h2.xyz", xyz2)]:
                data = content.encode("utf-8")
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        archive.write_bytes(buf.getvalue())
        return archive

    def test_load_zip(self, zip_archive):
        from models.data.molecule_loader import load_molecule_dataset
        ds = load_molecule_dataset(zip_archive)
        assert ds.n_molecules == 2
        assert ds.archive_format == "zip"
        assert "H" in ds.unique_elements
        assert "O" in ds.unique_elements
        assert "C" in ds.unique_elements
        assert ds.atom_count_range == (3, 5)

    def test_load_tar(self, tar_archive):
        from models.data.molecule_loader import load_molecule_dataset
        ds = load_molecule_dataset(tar_archive)
        assert ds.n_molecules == 2
        assert ds.archive_format == "tar.xz"
        assert "H" in ds.unique_elements
        assert "O" in ds.unique_elements

    def test_molecule_properties(self, zip_archive):
        from models.data.molecule_loader import load_molecule_dataset
        ds = load_molecule_dataset(zip_archive)
        water = [m for m in ds.molecules if m.name == "H2O"][0]
        assert water.n_atoms == 3
        assert water.formula_counts == {"O": 1, "H": 2}
        methane = [m for m in ds.molecules if m.name == "CH4"][0]
        assert methane.n_atoms == 5
        assert methane.formula_counts == {"C": 1, "H": 4}

    def test_max_molecules_cap(self, zip_archive):
        from models.data.molecule_loader import load_molecule_dataset
        ds = load_molecule_dataset(zip_archive, max_molecules=1)
        assert ds.n_molecules == 1

    def test_atom_count_histogram(self, zip_archive):
        from models.data.molecule_loader import load_molecule_dataset
        ds = load_molecule_dataset(zip_archive)
        hist, bins = ds.atom_count_histogram()
        assert len(hist) > 0
        assert hist.sum() == 2

    def test_content_hash_stable(self, zip_archive):
        from models.data.molecule_loader import load_molecule_dataset
        ds1 = load_molecule_dataset(zip_archive)
        ds2 = load_molecule_dataset(zip_archive)
        assert ds1.content_hash == ds2.content_hash
        assert len(ds1.content_hash) == 64

    def test_element_list_sorted(self, zip_archive):
        from models.data.molecule_loader import load_molecule_dataset
        ds = load_molecule_dataset(zip_archive)
        assert ds.element_list == sorted(ds.element_list)


class TestMoleculeDiscovery:

    def test_discover_finds_archives(self, tmp_path):
        from models.data.molecule_loader import discover_molecule_datasets
        mol = {
            "schema_name": "qcschema_molecule",
            "schema_version": 2,
            "name": "H2",
            "symbols": ["H", "H"],
            "atomic_numbers": [1, 1],
            "geometry": [0.0, 0.0, 0.0, 0.74, 0.0, 0.0],
            "molecular_charge": 0.0,
            "molecular_multiplicity": 1,
            "validated": True,
        }
        archive = tmp_path / "test_mols.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("h2.json", json.dumps(mol))
        results = discover_molecule_datasets(tmp_path)
        assert len(results) == 1
        assert results[0]["modality"] == "molecule"
        assert results[0]["n_molecules"] >= 1

    def test_discover_empty(self, tmp_path):
        from models.data.molecule_loader import discover_molecule_datasets
        assert discover_molecule_datasets(tmp_path) == []


@pytest.mark.skipif(
    not MERLIN_MOLECULES.exists(),
    reason="Merlin molecule data not available",
)
class TestRealMoleculeDatasets:

    def test_discover_molecules(self):
        from models.data.molecule_loader import discover_molecule_datasets
        results = discover_molecule_datasets(MERLIN_MOLECULES)
        assert len(results) >= 5
        formats = {r["archive_format"] for r in results}
        assert "zip" in formats

    def test_load_msr_sample(self):
        from models.data.molecule_loader import load_molecule_dataset
        archive = MERLIN_MOLECULES / "msr_acc_tae25_train_smallbasis28.zip"
        if not archive.exists():
            pytest.skip("MSR smallbasis28 not available")
        ds = load_molecule_dataset(archive, max_molecules=10)
        assert ds.n_molecules >= 1
        assert ds.archive_format == "zip"
        for mol in ds.molecules:
            assert mol.n_atoms > 0
            assert len(mol.geometry) == mol.n_atoms * 3

    def test_modality_bridge_includes_molecules(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(Path("/home/user/merlinaineuralplus2"))
        mol = [e for e in entries if e.modality.value == "molecule"]
        assert len(mol) >= 1
        assert mol[0].species == "not_applicable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
