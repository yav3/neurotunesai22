"""Tests for cross-platform probe-to-gene mapper."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MERLIN_EXPRESSION = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")


class TestIDFormatDetection:

    def test_gene_symbols(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        ids = ["BRCA1", "TP53", "EGFR", "MYC", "KRAS", "PTEN", "APC"]
        assert detect_id_format(ids) == IDFormat.GENE_SYMBOL

    def test_affymetrix_probes(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        ids = ["1007_s_at", "1053_at", "117_at", "1255_g_at", "1294_at",
               "1552256_a_at", "1552257_a_at", "200000_s_at"]
        assert detect_id_format(ids) == IDFormat.AFFYMETRIX

    def test_illumina_ilmn(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        ids = ["ILMN_10000", "ILMN_10001", "ILMN_10002", "ILMN_10004",
               "ILMN_1001", "ILMN_10010"]
        assert detect_id_format(ids) == IDFormat.ILLUMINA_ILMN

    def test_illumina_gi(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        ids = ["GI_10047089-S", "GI_10047091-S", "GI_10047093-S",
               "GI_10047095-S", "GI_10047097-A"]
        assert detect_id_format(ids) == IDFormat.ILLUMINA_GI

    def test_ensembl(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        ids = ["ENSG00000141510", "ENSG00000012048", "ENSG00000146648"]
        assert detect_id_format(ids) == IDFormat.ENSEMBL

    def test_empty(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        assert detect_id_format([]) == IDFormat.UNKNOWN

    def test_mixed_defaults_unknown(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        ids = ["foo", "bar123!!!!", "---", "", "###"]
        assert detect_id_format(ids) == IDFormat.UNKNOWN


class TestPlatformMapping:

    @pytest.fixture
    def affy_mapping(self):
        from models.data.probe_mapper import PlatformMapping
        return PlatformMapping(
            platform="GPL570",
            probe_to_gene={
                "1007_s_at": "DDR1",
                "1053_at": "RFC2",
                "117_at": "HSPA6",
                "121_at": "PAX8",
                "1255_g_at": "GUCA1A",
                "200000_s_at": "PRPF31",
                "200001_at": "CAPNS1",
            },
        )

    def test_mapping_counts(self, affy_mapping):
        assert affy_mapping.n_probes == 7
        assert affy_mapping.n_genes == 7

    def test_map_known(self, affy_mapping):
        assert affy_mapping.map_id("1007_s_at") == "DDR1"
        assert affy_mapping.map_id("121_at") == "PAX8"

    def test_map_unknown(self, affy_mapping):
        assert affy_mapping.map_id("NONEXISTENT_at") is None

    def test_many_to_one(self):
        from models.data.probe_mapper import PlatformMapping
        m = PlatformMapping(
            platform="test",
            probe_to_gene={"p1": "GENE_A", "p2": "GENE_A", "p3": "GENE_B"},
        )
        assert m.n_probes == 3
        assert m.n_genes == 2


class TestSOFTAnnotationParsing:

    def test_parse_soft(self):
        from models.data.probe_mapper import parse_geo_soft_annotation
        soft = (
            "^PLATFORM = GPL570\n"
            "!Platform_title = [HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array\n"
            "!platform_table_begin\n"
            "ID\tGB_ACC\tGene Symbol\n"
            "1007_s_at\tU48705\tDDR1 /// MIR4640\n"
            "1053_at\tM87338\tRFC2\n"
            "117_at\tX51757\tHSPA6\n"
            "121_at\tX69699\tPAX8\n"
            "empty_at\tZ00000\t---\n"
            "!platform_table_end\n"
        )
        mapping = parse_geo_soft_annotation(soft, platform="GPL570")
        assert mapping.platform == "GPL570"
        assert mapping.n_probes == 4
        assert mapping.probe_to_gene["1007_s_at"] == "DDR1"
        assert mapping.probe_to_gene["1053_at"] == "RFC2"
        assert "empty_at" not in mapping.probe_to_gene

    def test_multi_symbol_takes_first(self):
        from models.data.probe_mapper import parse_geo_soft_annotation
        soft = (
            "!platform_table_begin\n"
            "ID\tGene Symbol\n"
            "probe1\tGENE1 /// GENE2 /// GENE3\n"
            "!platform_table_end\n"
        )
        mapping = parse_geo_soft_annotation(soft)
        assert mapping.probe_to_gene["probe1"] == "GENE1"


class TestMappingCacheAndCSV:

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        from models.data import probe_mapper
        from models.data.probe_mapper import PlatformMapping, save_mapping, load_cached_mapping
        monkeypatch.setattr(probe_mapper, "_MAPPING_CACHE_DIR", tmp_path)
        m = PlatformMapping(
            platform="test_plat",
            probe_to_gene={"p1": "G1", "p2": "G2"},
        )
        save_mapping(m)
        loaded = load_cached_mapping("test_plat")
        assert loaded is not None
        assert loaded.n_probes == 2
        assert loaded.probe_to_gene == m.probe_to_gene

    def test_cache_miss(self, tmp_path, monkeypatch):
        from models.data import probe_mapper
        from models.data.probe_mapper import load_cached_mapping
        monkeypatch.setattr(probe_mapper, "_MAPPING_CACHE_DIR", tmp_path)
        assert load_cached_mapping("nonexistent") is None

    def test_build_from_csv(self, tmp_path):
        from models.data.probe_mapper import build_mapping_from_csv
        csv_file = tmp_path / "annot.tsv"
        csv_file.write_text("ID\tSymbol\n1007_s_at\tDDR1\n1053_at\tRFC2\n")
        m = build_mapping_from_csv(csv_file, platform="GPL570")
        assert m.n_probes == 2
        assert m.probe_to_gene["1007_s_at"] == "DDR1"


class TestProbeToGeneMapping:

    def test_map_probes_mean(self):
        from models.data.probe_mapper import PlatformMapping, map_probes_to_genes
        mapping = PlatformMapping(
            platform="test",
            probe_to_gene={"p1": "GENE_A", "p2": "GENE_A", "p3": "GENE_B"},
        )
        probe_ids = ["p1", "p2", "p3", "p4_unmapped"]
        expression = np.array([
            [1.0, 3.0, 5.0, 99.0],
            [2.0, 4.0, 6.0, 99.0],
        ], dtype=np.float32)
        result = map_probes_to_genes(probe_ids, expression, mapping, collapse="mean")
        assert result.gene_names == ["GENE_A", "GENE_B"]
        assert result.expression.shape == (2, 2)
        np.testing.assert_allclose(result.expression[:, 0], [2.0, 3.0])  # mean of p1,p2
        np.testing.assert_allclose(result.expression[:, 1], [5.0, 6.0])  # p3 alone
        assert result.n_probes_mapped == 3
        assert result.n_probes_unmapped == 1
        assert result.n_genes_collapsed == 1

    def test_map_probes_max(self):
        from models.data.probe_mapper import PlatformMapping, map_probes_to_genes
        mapping = PlatformMapping(
            platform="test",
            probe_to_gene={"p1": "GENE_A", "p2": "GENE_A"},
        )
        probe_ids = ["p1", "p2"]
        expression = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        result = map_probes_to_genes(probe_ids, expression, mapping, collapse="max")
        np.testing.assert_allclose(result.expression[:, 0], [5.0, 3.0])

    def test_passthrough_for_symbols(self):
        from models.data.probe_mapper import PlatformMapping, map_probes_to_genes
        symbols = ["BRCA1", "TP53", "EGFR"]
        identity_map = {s: s for s in symbols}
        mapping = PlatformMapping(platform="identity", probe_to_gene=identity_map)
        expression = np.eye(3, dtype=np.float32)
        result = map_probes_to_genes(symbols, expression, mapping)
        assert result.gene_names == sorted(symbols)
        assert result.n_probes_unmapped == 0
        assert result.n_genes_collapsed == 0


class TestCrossPlatformAlignment:

    def test_common_genes(self):
        from models.data.probe_mapper import find_common_genes
        assert find_common_genes(["A", "B", "C"], ["B", "C", "D"], ["C", "B"]) == ["B", "C"]

    def test_common_genes_empty(self):
        from models.data.probe_mapper import find_common_genes
        assert find_common_genes(["A"], ["B"]) == []
        assert find_common_genes() == []

    def test_align_datasets(self):
        from models.data.probe_mapper import (
            MappedExpression, align_to_common_genes, IDFormat,
        )
        ds1 = MappedExpression(
            gene_names=["A", "B", "C"],
            expression=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            original_ids=["A", "B", "C"],
            id_format=IDFormat.GENE_SYMBOL,
            n_probes_mapped=3, n_probes_unmapped=0, n_genes_collapsed=0,
            mapping_platform="identity",
        )
        ds2 = MappedExpression(
            gene_names=["B", "C", "D"],
            expression=np.array([[7, 8, 9]], dtype=np.float32),
            original_ids=["B", "C", "D"],
            id_format=IDFormat.GENE_SYMBOL,
            n_probes_mapped=3, n_probes_unmapped=0, n_genes_collapsed=0,
            mapping_platform="identity",
        )
        pool = align_to_common_genes([ds1, ds2], accessions=["GSE1", "GSE2"])
        assert pool.common_genes == ["B", "C"]
        assert pool.n_common_genes == 2
        assert pool.total_samples == 3
        assert pool.expressions[0].shape == (2, 2)
        assert pool.expressions[1].shape == (1, 2)
        np.testing.assert_allclose(pool.expressions[0][0], [2, 3])
        np.testing.assert_allclose(pool.expressions[1][0], [7, 8])
        pooled = pool.pooled_expression()
        assert pooled.shape == (3, 2)

    def test_per_dataset_stats(self):
        from models.data.probe_mapper import (
            MappedExpression, align_to_common_genes, IDFormat,
        )
        ds = MappedExpression(
            gene_names=["X", "Y"],
            expression=np.ones((5, 2), dtype=np.float32),
            original_ids=["X", "Y"],
            id_format=IDFormat.GENE_SYMBOL,
            n_probes_mapped=2, n_probes_unmapped=0, n_genes_collapsed=0,
            mapping_platform="identity",
        )
        pool = align_to_common_genes([ds, ds], accessions=["A", "B"])
        assert len(pool.per_dataset_stats) == 2
        assert pool.per_dataset_stats[0]["gene_retention"] == 1.0


@pytest.mark.skipif(
    not MERLIN_EXPRESSION.exists(),
    reason="Merlin expression data not available",
)
class TestRealDataFormatDetection:

    def test_detect_gse5281_affymetrix(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        import csv
        ids = []
        with open(MERLIN_EXPRESSION / "gse5281_neurodegeneration_full.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if i >= 50:
                    break
                ids.append(row[0])
        assert detect_id_format(ids) == IDFormat.AFFYMETRIX

    def test_detect_gse26927_illumina(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        import csv
        ids = []
        with open(MERLIN_EXPRESSION / "gse26927_neurodegeneration_full.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if i >= 50:
                    break
                ids.append(row[0])
        assert detect_id_format(ids) == IDFormat.ILLUMINA_ILMN

    def test_detect_gse15222_gi(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        import csv
        ids = []
        with open(MERLIN_EXPRESSION / "gse15222_neurodegeneration_full.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if i >= 50:
                    break
                ids.append(row[0])
        assert detect_id_format(ids) == IDFormat.ILLUMINA_GI

    def test_detect_gse119262_symbols(self):
        from models.data.probe_mapper import detect_id_format, IDFormat
        import csv
        ids = []
        with open(MERLIN_EXPRESSION / "gse119262_autophagy_full.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if i >= 50:
                    break
                ids.append(row[0])
        assert detect_id_format(ids) == IDFormat.GENE_SYMBOL


@pytest.mark.skipif(
    not MERLIN_EXPRESSION.exists(),
    reason="Merlin expression data not available",
)
class TestRealCrossPlatformPooling:
    """Integration test: pool real Affymetrix + Illumina + GI datasets via probe mapping."""

    def _load_and_map(self, csv_name, platform_id):
        from models.data.probe_mapper import (
            detect_id_format, IDFormat, PlatformMapping,
            load_cached_mapping, map_probes_to_genes,
        )
        import csv
        path = MERLIN_EXPRESSION / csv_name
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            ids, rows = [], []
            for row in reader:
                ids.append(row[0])
                rows.append([float(x) if x else 0.0 for x in row[1:]])
        expr = np.array(rows, dtype=np.float32).T  # (samples, genes)
        fmt = detect_id_format(ids)

        if fmt == IDFormat.GENE_SYMBOL:
            mapping = PlatformMapping(platform="identity", probe_to_gene={s: s for s in ids})
        else:
            mapping = load_cached_mapping(platform_id)
            assert mapping is not None, f"No cached mapping for {platform_id}"

        return map_probes_to_genes(ids, expr, mapping)

    def test_pool_three_platforms(self):
        from models.data.probe_mapper import align_to_common_genes
        ds_affy = self._load_and_map("gse5281_neurodegeneration_full.csv", "GPL570")
        ds_ilmn = self._load_and_map("gse26927_neurodegeneration_full.csv", "GPL6255")
        ds_gi = self._load_and_map("gse15222_neurodegeneration_full.csv", "GPL2700")

        assert ds_affy.n_probes_mapped > 10000
        assert ds_ilmn.n_probes_mapped > 10000
        assert ds_gi.n_probes_mapped > 10000

        pool = align_to_common_genes(
            [ds_affy, ds_ilmn, ds_gi],
            accessions=["GSE5281", "GSE26927", "GSE15222"],
        )
        assert pool.n_common_genes > 5000, f"Expected >5000 common genes, got {pool.n_common_genes}"
        assert pool.total_samples == 161 + 118 + 363  # 642
        pooled = pool.pooled_expression()
        assert pooled.shape == (642, pool.n_common_genes)
        assert np.isfinite(pooled).all()

    def test_pool_affy_plus_symbols(self):
        from models.data.probe_mapper import align_to_common_genes
        ds_affy = self._load_and_map("gse5281_neurodegeneration_full.csv", "GPL570")
        ds_sym = self._load_and_map("gse119262_autophagy_full.csv", "identity")

        pool = align_to_common_genes(
            [ds_affy, ds_sym],
            accessions=["GSE5281", "GSE119262"],
        )
        assert pool.n_common_genes > 100
        assert pool.total_samples == 161 + 44


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
