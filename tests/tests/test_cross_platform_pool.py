"""Tests for cross_platform_pool — probe-to-gene mapping + multi-platform pooling.

Unit tests use synthetic data; integration tests use real Merlin GEO CSVs
and cached platform mappings (GPL570, GPL6255, GPL2700).
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.data.cross_platform_pool import (
    MappedGEODataset,
    _detect_platform,
    load_and_map_geo,
    pool_cross_platform,
    load_and_pool_all_merlin,
)
from models.data.probe_mapper import IDFormat

MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")
EXPR_DIR = MERLIN_ROOT / "backend" / "data" / "expression" / "human"


# ── Synthetic helpers ────────────────────────────────────────────────────

def _make_symbol_csv(path: Path, genes: list[str], n_samples: int = 4) -> None:
    """Write a fake gene-symbol-indexed CSV."""
    rng = np.random.default_rng(42)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        samples = [f"ctrl_{i}" if i < n_samples // 2 else f"treated_{i}"
                   for i in range(n_samples)]
        w.writerow(["gene"] + samples)
        for g in genes:
            w.writerow([g] + [f"{v:.4f}" for v in rng.random(n_samples)])


def _make_mapped_dataset(
    genes: list[str],
    n_samples: int = 4,
    species: str = "human",
    accession: str = "GSE_SYNTH",
    platform: str = "identity",
) -> MappedGEODataset:
    rng = np.random.default_rng(0)
    expr = rng.random((n_samples, len(genes)), dtype=np.float32)
    conditions = ["control"] * (n_samples // 2) + ["treated"] * (n_samples - n_samples // 2)
    return MappedGEODataset(
        expression=expr,
        gene_names=genes,
        conditions=conditions,
        species=species,
        accession=accession,
        id_format=IDFormat.GENE_SYMBOL,
        n_original_ids=len(genes),
        n_mapped=len(genes),
        n_unmapped=0,
        n_collapsed=0,
        platform=platform,
    )


# ── Unit tests ───────────────────────────────────────────────────────────

class TestDetectPlatform(unittest.TestCase):
    def test_known_accession(self):
        self.assertEqual(_detect_platform([], "GSE5281"), "GPL570")
        self.assertEqual(_detect_platform([], "GSE26927"), "GPL6255")
        self.assertEqual(_detect_platform([], "GSE15222"), "GPL2700")

    def test_affy_probes(self):
        ids = ["1007_s_at", "1053_at", "117_at", "121_at"]
        self.assertEqual(_detect_platform(ids), "GPL570")

    def test_ilmn_probes(self):
        ids = ["ILMN_1762337", "ILMN_2055271", "ILMN_1736007"]
        self.assertEqual(_detect_platform(ids), "GPL6255")

    def test_gi_probes(self):
        ids = ["GI_4507670", "GI_13325066", "GI_6912586"]
        self.assertEqual(_detect_platform(ids), "GPL2700")

    def test_gene_symbols_return_none(self):
        ids = ["BRCA1", "TP53", "EGFR"]
        self.assertIsNone(_detect_platform(ids))

    def test_accession_overrides_probe_detection(self):
        affy_ids = ["1007_s_at", "1053_at"]
        self.assertEqual(_detect_platform(affy_ids, "GSE26927"), "GPL6255")


class TestLoadAndMapGeoSynthetic(unittest.TestCase):
    def test_symbol_passthrough(self):
        genes = ["BRCA1", "TP53", "EGFR", "MYC", "KRAS"]
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "GSE99999_test.csv"
            _make_symbol_csv(csv_path, genes, n_samples=6)
            d = load_and_map_geo(csv_path, species="human")

        self.assertEqual(d.accession, "GSE99999")
        self.assertEqual(d.id_format, IDFormat.GENE_SYMBOL)
        self.assertEqual(d.platform, "identity")
        self.assertEqual(d.n_unmapped, 0)
        self.assertEqual(d.n_collapsed, 0)
        self.assertEqual(len(d.gene_names), 5)
        self.assertEqual(d.expression.shape, (6, 5))

    def test_accession_extraction_from_filename(self):
        genes = ["A", "B"]
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "gse12345_some_dataset.csv"
            _make_symbol_csv(csv_path, genes)
            d = load_and_map_geo(csv_path)
        self.assertEqual(d.accession, "GSE12345")

    def test_log1p_applied(self):
        genes = ["BRCA1"]
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "test.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["gene", "ctrl_1", "treated_1"])
                w.writerow(["BRCA1", "100.0", "200.0"])
            d = load_and_map_geo(csv_path, log1p=True)
        self.assertAlmostEqual(d.expression[0, 0], np.log1p(100.0), places=3)

    def test_log1p_disabled(self):
        genes = ["BRCA1"]
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "test.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["gene", "ctrl_1", "treated_1"])
                w.writerow(["BRCA1", "100.0", "200.0"])
            d = load_and_map_geo(csv_path, log1p=False)
        self.assertAlmostEqual(d.expression[0, 0], 100.0, places=3)


class TestPoolCrossPlatform(unittest.TestCase):
    def test_basic_pooling(self):
        d1 = _make_mapped_dataset(["A", "B", "C"], n_samples=4, accession="GSE1")
        d2 = _make_mapped_dataset(["B", "C", "D"], n_samples=6, accession="GSE2")
        pooled, meta = pool_cross_platform([d1, d2])

        self.assertEqual(meta["n_common_genes"], 2)
        self.assertEqual(sorted(pooled.gene_names), ["B", "C"])
        self.assertEqual(pooled.expression.shape, (10, 2))
        self.assertEqual(meta["n_samples_total"], 10)
        self.assertEqual(len(pooled.conditions), 10)
        self.assertEqual(pooled.accession, "GSE1+GSE2")

    def test_species_firewall(self):
        d1 = _make_mapped_dataset(["A", "B"], species="human", accession="GSE1")
        d2 = _make_mapped_dataset(["A", "B"], species="mouse", accession="GSE2")
        with self.assertRaises(ValueError, msg="Cross-species"):
            pool_cross_platform([d1, d2])

    def test_empty_datasets_raises(self):
        with self.assertRaises(ValueError):
            pool_cross_platform([])

    def test_no_common_genes_raises(self):
        d1 = _make_mapped_dataset(["A", "B"], accession="GSE1")
        d2 = _make_mapped_dataset(["C", "D"], accession="GSE2")
        with self.assertRaises(ValueError, msg="No common genes"):
            pool_cross_platform([d1, d2])

    def test_max_genes_selection(self):
        genes = [f"G{i}" for i in range(50)]
        d1 = _make_mapped_dataset(genes, n_samples=10, accession="GSE1")
        d2 = _make_mapped_dataset(genes, n_samples=8, accession="GSE2")
        pooled, meta = pool_cross_platform([d1, d2], max_genes=10)

        self.assertEqual(meta["n_common_genes"], 10)
        self.assertEqual(pooled.expression.shape, (18, 10))

    def test_metadata_per_dataset(self):
        d1 = _make_mapped_dataset(["A", "B", "C"], accession="GSE1", platform="GPL570")
        d2 = _make_mapped_dataset(["B", "C", "D"], accession="GSE2", platform="GPL6255")
        _, meta = pool_cross_platform([d1, d2])

        self.assertEqual(len(meta["per_dataset"]), 2)
        self.assertEqual(meta["per_dataset"][0]["accession"], "GSE1")
        self.assertEqual(meta["per_dataset"][1]["accession"], "GSE2")
        self.assertIn("GPL570", meta["platforms"])
        self.assertIn("GPL6255", meta["platforms"])

    def test_all_finite_values(self):
        d1 = _make_mapped_dataset(["A", "B", "C"], n_samples=4, accession="GSE1")
        d2 = _make_mapped_dataset(["A", "B", "C"], n_samples=6, accession="GSE2")
        pooled, _ = pool_cross_platform([d1, d2])
        self.assertTrue(np.all(np.isfinite(pooled.expression)))


# ── Integration tests (real data) ────────────────────────────────────────

@unittest.skipUnless(
    EXPR_DIR.exists(),
    "Merlin expression data not available",
)
class TestLoadAndMapRealGEO(unittest.TestCase):
    """Load individual real GEO CSVs with probe mapping."""

    def test_affy_gse5281(self):
        csv_path = EXPR_DIR / "gse5281_neurodegeneration_full.csv"
        if not csv_path.exists():
            self.skipTest("GSE5281 not available")
        d = load_and_map_geo(csv_path, species="human")
        self.assertEqual(d.id_format, IDFormat.AFFYMETRIX)
        self.assertEqual(d.platform, "GPL570")
        self.assertGreater(d.n_mapped, 10000)
        self.assertGreater(len(d.gene_names), 5000)
        self.assertEqual(d.expression.shape[1], len(d.gene_names))

    def test_ilmn_gse26927(self):
        csv_path = EXPR_DIR / "gse26927_neurodegeneration_full.csv"
        if not csv_path.exists():
            self.skipTest("GSE26927 not available")
        d = load_and_map_geo(csv_path, species="human")
        self.assertEqual(d.id_format, IDFormat.ILLUMINA_ILMN)
        self.assertEqual(d.platform, "GPL6255")
        self.assertGreater(d.n_mapped, 5000)
        self.assertGreater(len(d.gene_names), 5000)

    def test_gi_gse15222(self):
        csv_path = EXPR_DIR / "gse15222_neurodegeneration_full.csv"
        if not csv_path.exists():
            self.skipTest("GSE15222 not available")
        d = load_and_map_geo(csv_path, species="human")
        self.assertEqual(d.id_format, IDFormat.ILLUMINA_GI)
        self.assertEqual(d.platform, "GPL2700")
        self.assertGreater(d.n_mapped, 5000)
        self.assertGreater(len(d.gene_names), 5000)

    def test_symbol_gse119262(self):
        csv_path = EXPR_DIR / "gse119262_autophagy_full.csv"
        if not csv_path.exists():
            self.skipTest("GSE119262 not available")
        d = load_and_map_geo(csv_path, species="human")
        self.assertEqual(d.id_format, IDFormat.GENE_SYMBOL)
        self.assertEqual(d.platform, "identity")
        self.assertEqual(d.n_unmapped, 0)


@unittest.skipUnless(
    EXPR_DIR.exists(),
    "Merlin expression data not available",
)
class TestCrossPlatformPoolIntegration(unittest.TestCase):
    """Pool real datasets across different microarray platforms."""

    def test_three_platform_pool(self):
        """Pool Affymetrix + Illumina ILMN + GI across 3 platforms."""
        paths = [
            ("gse5281_neurodegeneration_full.csv", "GPL570"),
            ("gse26927_neurodegeneration_full.csv", "GPL6255"),
            ("gse15222_neurodegeneration_full.csv", "GPL2700"),
        ]
        datasets = []
        for fname, expected_platform in paths:
            p = EXPR_DIR / fname
            if not p.exists():
                self.skipTest(f"{fname} not available")
            d = load_and_map_geo(p, species="human")
            self.assertEqual(d.platform, expected_platform)
            datasets.append(d)

        pooled, meta = pool_cross_platform(datasets)

        self.assertGreater(meta["n_common_genes"], 5000)
        self.assertEqual(meta["n_datasets"], 3)
        total = sum(d.expression.shape[0] for d in datasets)
        self.assertEqual(pooled.expression.shape[0], total)
        self.assertEqual(pooled.expression.shape[1], meta["n_common_genes"])
        self.assertTrue(np.all(np.isfinite(pooled.expression)))
        self.assertEqual(len(meta["platforms"]), 3)

    def test_affy_plus_symbols(self):
        """Pool Affymetrix probe-mapped + gene-symbol datasets."""
        affy = EXPR_DIR / "gse5281_neurodegeneration_full.csv"
        sym = EXPR_DIR / "gse119262_autophagy_full.csv"
        if not affy.exists() or not sym.exists():
            self.skipTest("Required CSVs not available")

        d_affy = load_and_map_geo(affy, species="human")
        d_sym = load_and_map_geo(sym, species="human")
        pooled, meta = pool_cross_platform([d_affy, d_sym])

        self.assertGreater(meta["n_common_genes"], 100)
        self.assertEqual(pooled.expression.shape[0],
                         d_affy.expression.shape[0] + d_sym.expression.shape[0])
        self.assertTrue(np.all(np.isfinite(pooled.expression)))

    def test_max_genes_variance_selection(self):
        """Variance-based gene selection on real data."""
        paths = [
            EXPR_DIR / "gse5281_neurodegeneration_full.csv",
            EXPR_DIR / "gse26927_neurodegeneration_full.csv",
        ]
        datasets = []
        for p in paths:
            if not p.exists():
                self.skipTest(f"{p.name} not available")
            datasets.append(load_and_map_geo(p, species="human"))

        pooled, meta = pool_cross_platform(datasets, max_genes=2000)
        self.assertEqual(meta["n_common_genes"], 2000)
        self.assertEqual(pooled.expression.shape[1], 2000)

    def test_gene_retention_stats(self):
        """Per-dataset gene retention is plausible (30-80%)."""
        paths = [
            EXPR_DIR / "gse5281_neurodegeneration_full.csv",
            EXPR_DIR / "gse26927_neurodegeneration_full.csv",
            EXPR_DIR / "gse15222_neurodegeneration_full.csv",
        ]
        datasets = []
        for p in paths:
            if not p.exists():
                self.skipTest(f"{p.name} not available")
            datasets.append(load_and_map_geo(p, species="human"))

        _, meta = pool_cross_platform(datasets)
        for ds_stat in meta["per_dataset"]:
            retention = ds_stat["gene_retention"]
            self.assertGreater(retention, 0.30, f"{ds_stat['accession']} retention too low")
            self.assertLess(retention, 0.80, f"{ds_stat['accession']} retention suspiciously high")


@unittest.skipUnless(
    EXPR_DIR.exists(),
    "Merlin expression data not available",
)
class TestLoadAndPoolAllMerlin(unittest.TestCase):
    """End-to-end test of the top-level multi-program pooling function."""

    def test_pool_all_human(self):
        pooled, meta = load_and_pool_all_merlin(
            MERLIN_ROOT, species="human", max_genes=2000,
        )
        self.assertGreaterEqual(meta["n_datasets"], 2)
        self.assertGreater(meta["n_samples_total"], 50)
        self.assertLessEqual(pooled.expression.shape[1], 2000)
        self.assertGreater(pooled.expression.shape[1], 0)
        self.assertTrue(np.all(np.isfinite(pooled.expression)))
        self.assertEqual(pooled.species, "human")
        self.assertIn("+", pooled.accession)


if __name__ == "__main__":
    unittest.main()
